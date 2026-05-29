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
        image_transform = vlm.vision_backbone.image_transform
        tokenizer = vlm.llm_backbone.tokenizer

        tokenized = tokenizer(prompt, truncation=True, return_tensors="pt")
        input_ids = tokenized.input_ids.to(self._device)
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self._device)
        elif isinstance(pixel_values, dict):
            pixel_values = {
                key: value[None, ...].to(self._device) for key, value in pixel_values.items()
            }
        else:
            raise ValueError(f"Unsupported Prismatic pixel_values type: {type(pixel_values)}")

        autocast_device = "cuda" if self._device.type == "cuda" else "cpu"
        with torch.inference_mode():
            with torch.autocast(
                autocast_device, dtype=self._dtype, enabled=self._device.type == "cuda"
            ):
                output = vlm(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True,
                )
        hidden_states = output.hidden_states
        if hidden_states is None:
            raise RuntimeError("Prismatic forward did not return hidden states.")
        feature = hidden_states[-1][:, -1, :].detach().float().cpu().numpy()[0]
        if feature.shape[0] != self.config.output_dim:
            raise ValueError(
                f"Expected feature dim {self.config.output_dim}, observed {feature.shape[0]}"
            )
        return feature

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

    def _build_prompt(self, goal_text: str) -> str:
        message = self.config.prompt_template.format(goal_text=goal_text)
        prompt_builder = self._vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=message)
        return prompt_builder.get_prompt()


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
