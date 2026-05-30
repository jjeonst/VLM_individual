"""Prismatic VLM adapter for prompt-conditioned hidden-state extraction."""

from __future__ import annotations

from pathlib import Path

from configs.schema import VLMConfig
from encoders.base import VLMEncoder


class PrismaticEncoder(VLMEncoder):
    """Load a frozen Prismatic VLM and expose prompt-conditioned features."""

    def __init__(self, config: VLMConfig):
        self.config = config
        self._torch = None
        self._vlm = None
        self._device = None
        self._dtype = None
        self._load_model()

    @property
    def output_dim(self) -> int:
        return self.config.output_dim

    def encode_image_goal(self, image: object, goal_text: str):
        torch = self._torch
        vlm = self._vlm
        prompt = self._build_prompt(goal_text)
        input_ids, attention_mask, pixel_values = self._prepare_inputs(image, prompt)
        autocast_device = "cuda" if self._device.type == "cuda" else "cpu"
        with torch.inference_mode():
            with torch.autocast(
                autocast_device, dtype=self._dtype, enabled=self._device.type == "cuda"
            ):
                output, _ = self._forward_hidden_states(input_ids, attention_mask, pixel_values)
        hidden_states = output.hidden_states
        if hidden_states is None:
            raise RuntimeError("Prismatic forward did not return hidden states.")
        feature = hidden_states[-1][:, -1, :].detach().float().cpu().numpy()[0]
        if feature.shape[0] != self.config.output_dim:
            raise ValueError(
                f"Expected feature dim {self.config.output_dim}, observed {feature.shape[0]}"
            )
        return feature

    def encode_image_goal_tokens(self, image: object, goal_text: str) -> dict[str, object]:
        torch = self._torch
        prompt = self._build_prompt(goal_text)
        generated_text = None
        if self.config.include_generated_text:
            generated_text = self._generate_answer(image, prompt)
            prompt = self._build_prompt(goal_text, generated_text=generated_text)
        input_ids, attention_mask, pixel_values = self._prepare_inputs(image, prompt)

        autocast_device = "cuda" if self._device.type == "cuda" else "cpu"
        with torch.inference_mode():
            with torch.autocast(
                autocast_device, dtype=self._dtype, enabled=self._device.type == "cuda"
            ):
                output, visual_token_count = self._forward_hidden_states(
                    input_ids, attention_mask, pixel_values
                )
        hidden_states = output.hidden_states
        if hidden_states is None:
            raise RuntimeError("Prismatic forward did not return hidden states.")

        layer_tokens = []
        for layer_index in self.config.hidden_layer_indices:
            layer = hidden_states[layer_index][0, 1 : 1 + visual_token_count, :]
            layer_tokens.append(
                _pool_visual_tokens(
                    layer.float(),
                    pool_grid=self.config.visual_pool_grid,
                    bank_reduction=self.config.visual_bank_reduction,
                    torch=torch,
                )
            )
        tokens = torch.cat(layer_tokens, dim=-1).detach().cpu().numpy()
        if self.config.projection == "none" and tokens.shape[-1] != self.config.output_dim:
            raise ValueError(
                f"Expected token dim {self.config.output_dim}, observed {tokens.shape[-1]}"
            )
        return {
            "tokens": tokens,
            "prompt": prompt,
            "generated_text": generated_text,
            "hidden_layer_indices": list(self.config.hidden_layer_indices),
            "visual_token_count": int(visual_token_count),
            "pooled_token_count": int(tokens.shape[0]),
        }

    def _load_model(self) -> None:
        try:
            import torch
            from prismatic import load
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PrismaticEncoder requires torch and prismatic-vlms in the runtime environment."
            ) from exc

        self._torch = torch
        self._device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self._dtype = _resolve_dtype(torch, self.config.dtype)
        hf_token = _read_token(self.config.hf_token_path)
        load_id = _resolve_load_id(self.config.model_id, self.config.weights_path)
        vlm = load(load_id, hf_token=hf_token)
        vlm.to(self._device, dtype=self._dtype)
        if self.config.frozen:
            vlm.requires_grad_(False)
            vlm.eval()
        self._vlm = vlm

    def _build_prompt(self, goal_text: str, generated_text: str | None = None) -> str:
        message = self.config.prompt_template.format(goal_text=goal_text)
        prompt_builder = self._vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=message)
        if generated_text is not None:
            prompt_builder.add_turn(role="gpt", message=generated_text)
        return prompt_builder.get_prompt()

    def _generate_answer(self, image: object, prompt: str) -> str:
        self._torch.manual_seed(self.config.generation_seed)
        if self._device.type == "cuda":
            self._torch.cuda.manual_seed_all(self.config.generation_seed)
        return self._vlm.generate(
            image,
            prompt,
            do_sample=True,
            temperature=self.config.generation_temperature,
            max_new_tokens=self.config.max_new_tokens,
        )

    def _prepare_inputs(self, image: object, prompt: str):
        torch = self._torch
        vlm = self._vlm
        image_transform = vlm.vision_backbone.image_transform
        tokenizer = vlm.llm_backbone.tokenizer
        tokenized = tokenizer(prompt, truncation=True, return_tensors="pt")
        input_ids = tokenized.input_ids.to(self._device)
        attention_mask = torch.ones_like(input_ids)
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self._device)
        elif isinstance(pixel_values, dict):
            pixel_values = {
                key: value[None, ...].to(self._device) for key, value in pixel_values.items()
            }
        else:
            raise ValueError(f"Unsupported Prismatic pixel_values type: {type(pixel_values)}")
        return input_ids, attention_mask, pixel_values

    def _forward_hidden_states(self, input_ids, attention_mask, pixel_values):
        torch = self._torch
        vlm = self._vlm
        with torch.set_grad_enabled(vlm.vision_backbone_requires_grad):
            patch_features = vlm.vision_backbone(pixel_values)
        projected_patch_embeddings = vlm.projector(patch_features)
        patch_attention_mask = torch.full(
            (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
            True,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        input_embeddings = vlm.llm_backbone.embed_input_ids(input_ids)
        fused_embeddings = torch.cat(
            [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]],
            dim=1,
        )
        fused_attention_mask = torch.cat(
            [attention_mask[:, :1], patch_attention_mask, attention_mask[:, 1:]],
            dim=1,
        )
        output = vlm.llm_backbone(
            input_ids=None,
            attention_mask=fused_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=fused_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        return output, int(projected_patch_embeddings.shape[1])


def _resolve_dtype(torch, dtype_name: str):
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _read_token(token_path: str | None) -> str | None:
    if token_path is None:
        return None
    path = Path(token_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8").strip()


def _resolve_load_id(model_id: str, weights_path: str) -> str:
    path = Path(weights_path)
    if path.exists():
        return str(path)
    return model_id


def _pool_visual_tokens(tokens, *, pool_grid: int, bank_reduction: str, torch):
    banks = _split_visual_banks(tokens, torch=torch)
    pooled = [_adaptive_pool_bank(bank, pool_grid=pool_grid, torch=torch) for bank in banks]
    if bank_reduction == "mean":
        return torch.stack(pooled, dim=0).mean(dim=0)
    if bank_reduction == "concat":
        return torch.cat(pooled, dim=0)
    raise ValueError(f"Unsupported visual_bank_reduction: {bank_reduction}")


def _split_visual_banks(tokens, *, torch):
    token_count = int(tokens.shape[0])
    side = _square_side(token_count)
    if side is not None:
        return [tokens]
    side = _square_side(token_count - 1)
    if side is not None:
        return [tokens[1:]]
    if token_count % 2 == 0:
        half = token_count // 2
        side = _square_side(half)
        if side is not None:
            return [tokens[:half], tokens[half:]]
        side = _square_side(half - 1)
        if side is not None:
            return [tokens[1:half], tokens[half + 1 :]]
    raise ValueError(f"Cannot infer square visual token grid from {token_count} tokens.")


def _adaptive_pool_bank(tokens, *, pool_grid: int, torch):
    side = _square_side(int(tokens.shape[0]))
    if side is None:
        raise ValueError(f"Visual token bank is not square: {tokens.shape[0]}")
    if side == pool_grid:
        return tokens
    grid = tokens.reshape(side, side, tokens.shape[-1]).permute(2, 0, 1).unsqueeze(0)
    pooled = torch.nn.functional.adaptive_avg_pool2d(grid, (pool_grid, pool_grid))
    return pooled.squeeze(0).permute(1, 2, 0).reshape(pool_grid * pool_grid, tokens.shape[-1])


def _square_side(token_count: int) -> int | None:
    if token_count <= 0:
        return None
    side = int(token_count**0.5)
    if side * side == token_count:
        return side
    return None
