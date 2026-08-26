"""Is position 0 of the prefill the BOS token or the first image patch?

Causal masking makes the answer measurable. Position 0 attends to nothing but itself, so if it
is BOS its hidden state cannot depend on the image, and if it is a patch it must. Two different
images through the same prompt settle it.
"""
import numpy as np, torch
from vlm_features import LAYERS, VISUAL_GRID, build_prompt, load_vlm, _stack_pixel_values

vlm = load_vlm()
device = next(vlm.parameters()).device
tok = vlm.llm_backbone.tokenizer

rng = np.random.default_rng(0)
a = rng.integers(0, 255, (1, 480, 640, 3), dtype=np.uint8)
b = rng.integers(0, 255, (1, 480, 640, 3), dtype=np.uint8)

prompt = build_prompt(vlm, "chair", with_cot=True)
ids = tok(prompt, truncation=True, return_tensors="pt").input_ids.to(device)
print("첫 토큰 id", int(ids[0, 0]), "| BOS id", tok.bos_token_id, "| 질문 길이", ids.shape[1])

def prefill(frames):
    pv = _stack_pixel_values(vlm.vision_backbone.image_transform, frames, device)
    with torch.autocast("cuda", dtype=vlm.llm_backbone.half_precision_dtype):
        out = vlm(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pv,
                  output_hidden_states=True, use_cache=False, return_dict=True)
    return torch.stack([out.hidden_states[l] for l in LAYERS], dim=2).float().cpu()

pa, pb = prefill(a), prefill(b)
print("prefill 길이", pa.shape[1], "= 256 +", pa.shape[1] - 256)
for pos in (0, 1, 2, 255, 256, 257):
    d = float((pa[0, pos] - pb[0, pos]).abs().max())
    print(f"  위치 {pos:3d}: 두 이미지 간 최대 차이 {d:.3e}  {'← 이미지와 무관' if d < 1e-5 else ''}")
