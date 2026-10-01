"""Does the 32-token minimum turn the no-CoT condition into a chain-of-thought one?

Appendix C.7 found the two conditions within 0.8 %p of each other (47.4 % against 46.6 %) where
the paper reports 14.1 %p between them, and the stored sample answers suggested why: asked only
"Would a bed be found here?", the model still goes on to describe the room, because
`vlm_features.py` forbids the end-of-sequence token until 32 tokens have been produced. If the
paper let the no-CoT answer stop where the model wanted to, its representation would carry a
"Yes." and little else.

This is the cheap check before re-encoding 1.2 M frames. The same frames are asked four ways --
{with, without} "Why or why not?" x {minimum 0, minimum 32} -- with the same per-frame seed, so a
difference between settings is the setting and not the draw. What it reports is how long the
answers run and whether they reason, not what the policy would do with them.

**It runs on two rtx2080 cards, in float16, which is not how the embeddings were made.** Turing
has no bfloat16 and the 7B model does not fit one 11 GB card, so the language model's layers are
split across two with `accelerate`. Precision can nudge individual draws; it is unlikely to
change whether an answer stops after one word or runs to thirty, and all four settings share it.
The min-32 settings are also the control: their answers should read like the stored ones.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_features import (MAX_NEW_TOKENS, MIN_NEW_TOKENS, MODEL_DIR, TEMPERATURE,  # noqa: E402
                          _stack_pixel_values, build_prompt, frame_seed)

HABITAT_ROOT = Path("/data/topovlm/habitat")
MANIFEST = HABITAT_ROOT / "episodes" / "pr2l_habitat_web_hd" / "train" / "manifest.jsonl"
SETTINGS = (("nocot", False, 0), ("nocot", False, MIN_NEW_TOKENS),
            ("cot", True, 0), ("cot", True, MIN_NEW_TOKENS))
REASON = re.compile(r"\b(because|since|as it|as there|due to|typically|usually|common)\b", re.I)


def load_split_vlm(layers_on_first: int, model_dir: str = MODEL_DIR):
    """The Prismatic model in float16, its language model split across cuda:0 and cuda:1.

    The vision backbone, the projector and the token embedding stay on cuda:0 because Prismatic
    concatenates projected patches with embedded prompt tokens before the language model sees
    either, and that concatenation needs both on one device.
    """
    from accelerate import dispatch_model
    from prismatic import load

    vlm = load(model_dir).to(dtype=torch.float16).eval()
    for parameter in vlm.parameters():
        parameter.requires_grad_(False)
    vlm.vision_backbone.to("cuda:0")
    vlm.projector.to("cuda:0")

    llm = vlm.llm_backbone.llm
    layers = llm.config.num_hidden_layers
    device_map = {"model.embed_tokens": 0, "model.norm": 1, "lm_head": 1}
    for index in range(layers):
        device_map[f"model.layers.{index}"] = 0 if index < layers_on_first else 1
    # Buffers registered on the model itself (4.38's `causal_mask`) are not inside any layer.
    for name, _ in llm.model.named_buffers(recurse=False):
        device_map[f"model.{name}"] = 0
    # `skip_keys`: the hook on the top-level model sends every tensor argument to cuda:0, the
    # cache included, so on the second token the eighteen layers on cuda:1 found their own keys
    # moved away and failed to concatenate (job 16075). `from_pretrained` avoids this through
    # `_skip_keys_device_placement`; calling `dispatch_model` directly has to say it.
    dispatch_model(llm, device_map=device_map, main_device=0,
                   skip_keys=getattr(llm, "_skip_keys_device_placement", "past_key_values"))
    for device in (0, 1):
        print(f"[load] cuda:{device} {torch.cuda.memory_allocated(device) / 1e9:.2f} GB", flush=True)
    return vlm


def sample(logits: torch.Tensor, generators: list[torch.Generator], forbid: torch.Tensor,
           eos_id: int) -> tuple[torch.Tensor, int]:
    logits = logits.float().cpu()
    logits[forbid, eos_id] = float("-inf")
    probabilities = torch.softmax(logits / TEMPERATURE, dim=-1)
    broken = 0
    drawn = []
    for i, generator in enumerate(generators):
        row = probabilities[i]
        if not torch.isfinite(row).all():
            broken += 1
            drawn.append(torch.tensor([eos_id]))
            continue
        drawn.append(torch.multinomial(row, 1, generator=generator))
    return torch.cat(drawn), broken


@torch.inference_mode()
def answer(vlm, frames: np.ndarray, goal: str, owner: str, indices: list[int], with_cot: bool,
           minimum: int) -> tuple[list[str], list[int], int]:
    tokenizer = vlm.llm_backbone.tokenizer
    eos_id = tokenizer.eos_token_id
    batch = len(indices)
    prompt = build_prompt(vlm, goal, with_cot=with_cot)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
    input_ids = input_ids.expand(batch, -1).contiguous()
    pixels = _stack_pixel_values(vlm.vision_backbone.image_transform, frames, "cuda:0")
    pixels = ({k: v.half() for k, v in pixels.items()} if isinstance(pixels, dict)
              else pixels.half())
    generators = [torch.Generator().manual_seed(frame_seed(owner, i)) for i in indices]

    output = vlm(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                 pixel_values=pixels, use_cache=True, return_dict=True)
    past = output.past_key_values
    finished = torch.zeros(batch, dtype=torch.bool)
    lengths = torch.zeros(batch, dtype=torch.long)
    # As in `encode_batch`: the first token may never be end-of-sequence.
    token, broken = sample(output.logits[:, -1], generators, ~finished, eos_id)
    del output
    drawn = []
    for _ in range(MAX_NEW_TOKENS):
        drawn.append(token)
        lengths += (~finished).long()
        finished |= token == eos_id
        if bool(finished.all()):
            break
        output = vlm(input_ids=token[:, None].to("cuda:0"), past_key_values=past,
                     use_cache=True, return_dict=True)
        past = output.past_key_values
        token, more = sample(output.logits[:, -1], generators, (lengths < minimum) | finished,
                             eos_id)
        broken += more
        del output

    ids = torch.stack(drawn, dim=1)
    texts, counts = [], []
    for i in range(batch):
        row = ids[i, :int(lengths[i])].tolist()
        if row and row[-1] == eos_id:
            row = row[:-1]
        counts.append(len(row))
        texts.append(tokenizer.decode(row, skip_special_tokens=True).strip())
    return texts, counts, broken


def pick(records: list[dict], trajectories: int, per_trajectory: int) -> list[tuple[dict, list[int]]]:
    """Evenly spaced trajectories, evenly spaced frames inside each -- no randomness to argue over."""
    stride = max(len(records) // trajectories, 1)
    chosen = []
    for record in records[::stride][:trajectories]:
        steps = int(record["steps"])
        if steps < per_trajectory:
            continue
        indices = np.linspace(0, steps - 1, per_trajectory + 2)[1:-1].round().astype(int).tolist()
        chosen.append((record, sorted(set(indices))))
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=int, default=40)
    parser.add_argument("--per-trajectory", type=int, default=5)
    parser.add_argument("--layers-on-first", type=int, default=14)
    parser.add_argument("--out", type=Path, default=Path("/data/topovlm/logs/probe_answer_length.json"))
    parser.add_argument("--model-dir", default=MODEL_DIR,
                        help="Prismatic checkpoint directory. Appendix C.9 compares ours against "
                             "prism-dinosiglip-224px-controlled+7b, trained on LLaVA v1.5 alone")
    args = parser.parse_args()

    if torch.cuda.device_count() < 2:
        print(f"GPU {torch.cuda.device_count()}장 — 두 장이 필요하다")
        return 1
    print(" | ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())))

    records = [json.loads(line) for line in MANIFEST.open()]
    chosen = pick(records, args.trajectories, args.per_trajectory)
    frames_total = sum(len(i) for _, i in chosen)
    print(f"[probe] 궤적 {len(chosen)}개, 프레임 {frames_total}장, 설정 {len(SETTINGS)}개", flush=True)

    started = time.time()
    print(f"[probe] 체크포인트 {args.model_dir}", flush=True)
    vlm = load_split_vlm(args.layers_on_first, args.model_dir)
    print(f"[load] {time.time() - started:.0f}s", flush=True)

    results: dict[str, list[dict]] = defaultdict(list)
    broken_total = defaultdict(int)
    for position, (record, indices) in enumerate(chosen, start=1):
        frames = np.load(HABITAT_ROOT / record["rgb_path"], mmap_mode="r")[indices]
        frames = np.ascontiguousarray(frames)
        for name, with_cot, minimum in SETTINGS:
            key = f"{name}_min{minimum}"
            texts, counts, broken = answer(vlm, frames, record["object_category"],
                                           record["episode_id"], indices, with_cot, minimum)
            broken_total[key] += broken
            for index, text, count in zip(indices, texts, counts):
                results[key].append({"episode_id": record["episode_id"], "frame": index,
                                     "goal": record["object_category"], "tokens": count,
                                     "text": text})
        if position % 5 == 0 or position == len(chosen):
            print(f"[probe] {position}/{len(chosen)} 궤적, {time.time() - started:.0f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"\n[probe] 저장: {args.out}")

    print("\n=== 답변 길이 (토큰) ===")
    print(f"  {'설정':12s} {'평균':>5s} {'중앙':>5s} {'≤3':>6s} {'≤10':>6s} {'<32':>6s} {'=48':>6s} "
          f"{'이유어휘':>7s} {'2문장+':>7s} {'NaN':>4s}")
    for name, _, minimum in SETTINGS:
        key = f"{name}_min{minimum}"
        rows = results[key]
        n = np.array([r["tokens"] for r in rows])
        reason = np.mean([bool(REASON.search(r["text"])) for r in rows])
        multi = np.mean([len([s for s in re.split(r"[.!?]", r["text"]) if s.strip()]) >= 2
                         for r in rows])
        print(f"  {key:12s} {n.mean():5.1f} {np.median(n):5.0f} {100 * np.mean(n <= 3):5.1f}% "
              f"{100 * np.mean(n <= 10):5.1f}% {100 * np.mean(n < 32):5.1f}% "
              f"{100 * np.mean(n >= MAX_NEW_TOKENS):5.1f}% {100 * reason:6.1f}% {100 * multi:6.1f}% "
              f"{broken_total[key]:4d}")

    print("\n=== 같은 프레임, 네 설정 ===")
    for j in range(0, len(results["nocot_min0"]), max(len(results["nocot_min0"]) // 6, 1))[:6]:
        head = results["nocot_min0"][j]
        print(f"\n[{head['goal']}] {head['episode_id'][:20]} 프레임 {head['frame']}")
        for name, _, minimum in SETTINGS:
            row = results[f"{name}_min{minimum}"][j]
            print(f"  {name + '_min' + str(minimum):12s} ({row['tokens']:2d}) {row['text'][:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
