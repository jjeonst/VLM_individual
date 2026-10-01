"""Appendix C.8's probe: three candidate causes of the no-CoT gap, on one model load.

  ① top-k 50. The paper's extraction code calls HF `generate`, which truncates sampling to the
    50 most likely tokens unless told otherwise; `vlm_features._sample` draws from the whole
    vocabulary. The candidate only survives if truncation makes the no-CoT answers *worse* --
    repetitive, emptier of the scene -- while the CoT answers keep describing. If truncation just
    cleans away garbage and no-CoT still describes the room, it is rejected (C.8 ①, direction).
  ② a shared PCA basis. If the paper reduced no-CoT tokens with a basis fitted on CoT outputs,
    no-CoT would lose more of its variance. Measured directly: this run's no-CoT and CoT tokens,
    before reduction, are projected with both stored bases.
  ③ the prompt wrapper. Prismatic's pure Llama was trained on `In: ...\\nOut: `; the paper's
    listing passes a bare string (for a different model). Asked bare, does no-CoT stop answering?

Every setting keeps the paper's 32-48 new tokens and temperature 0.4, and shares the per-frame seed
with `probe_answer_length.py` and the same 200 frames, so its `full` settings should reproduce that
run's min-32 rows -- a check that the harness did not drift.

Same hardware caveat as the earlier probe: two rtx2080 cards, float16, language model split.
Bases were fitted on bfloat16 activations, so ② is read as a comparison between the two bases on
the same tokens, not as an absolute retention figure.
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
import pca  # noqa: E402
from probe_answer_length import HABITAT_ROOT, MANIFEST, load_split_vlm, pick  # noqa: E402
from vlm_features import (LAYERS, MAX_NEW_TOKENS, MIN_NEW_TOKENS,  # noqa: E402
                          PROMPT_WITH_COT, PROMPT_WITHOUT_COT, TEMPERATURE, _assemble,
                          _stack_pixel_values, build_prompt, frame_seed)

EMBEDDINGS = HABITAT_ROOT / "embeddings"
# name, with_cot, top_k, wrapped, capture hidden states for ②
SETTINGS = (
    ("nocot_full", False, None, True, True),
    ("nocot_top50", False, 50, True, False),
    ("cot_full", True, None, True, True),
    ("cot_top50", True, 50, True, False),
    ("nocot_bare", False, None, False, False),
    ("cot_bare", True, None, False, False),
)

REASON = re.compile(r"\b(because|since|due to|typically|usually|common(ly)?|likely)\b", re.I)
GARBAGE = re.compile(r"[=_{};<>\[\]#]|\(\s*\d|\d+\s*,\s*\d+|\.\w+\(")
SCENE = re.compile(
    r"\b(bed(room)?|sofa|couch|chairs?|table|kitchen|living room|bathroom|toilet|tv|television|"
    r"monitor|plants?|window|door(way)?|floor|walls?|cabinets?|sink|lamp|mirror|shelf|shelves|"
    r"rug|carpet|hallway|stairs?|staircase|counter|desk|picture|painting|pillows?|fireplace|"
    r"ceiling|curtains?|dresser|closet|bathtub|shower|refrigerator|fridge|oven|stove|bench|"
    r"room|dining|office|entryway|corridor|balcony|furniture|decor(ation)?s?)\b", re.I)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def describe(text: str) -> dict:
    parts = sentences(text)
    rest = " ".join(parts[1:])
    words = re.findall(r"\w+", text.lower())
    grams = [tuple(words[i:i + 4]) for i in range(max(len(words) - 3, 0))]
    normalised = [re.sub(r"\W+", " ", s.lower()).strip() for s in parts]
    long_enough = [s for s in normalised if len(s.split()) >= 3]
    return {
        "repeat4": 1 - len(set(grams)) / len(grams) if grams else 0.0,
        "repeated_sentence": len(long_enough) != len(set(long_enough)),
        "garbage": bool(GARBAGE.search(text)),
        "reason": bool(REASON.search(text)),
        "scene_after_first": len({m.group(0).lower() for m in SCENE.finditer(rest)}),
        "sentences": len(parts),
    }


def sample(logits: torch.Tensor, generators: list[torch.Generator], forbid: torch.Tensor,
           eos_id: int, top_k: int | None) -> tuple[torch.Tensor, int]:
    """HF's order: minimum length masks EOS, temperature divides, top-k keeps the k largest."""
    logits = logits.float().cpu()
    logits[forbid, eos_id] = float("-inf")
    logits = logits / TEMPERATURE
    if top_k:
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probabilities = torch.softmax(logits, dim=-1)
    drawn, broken = [], 0
    for i, generator in enumerate(generators):
        if not torch.isfinite(probabilities[i]).all():
            broken += 1
            drawn.append(torch.tensor([eos_id]))
            continue
        drawn.append(torch.multinomial(probabilities[i], 1, generator=generator))
    return torch.cat(drawn), broken


@torch.inference_mode()
def run(vlm, frames: np.ndarray, goal: str, owner: str, indices: list[int], with_cot: bool,
        top_k: int | None, wrapped: bool, capture: bool):
    """Mirrors `encode_batch`'s loop order so captured tokens line up with the stored ones."""
    tokenizer = vlm.llm_backbone.tokenizer
    eos_id = tokenizer.eos_token_id
    batch = len(indices)
    template = PROMPT_WITH_COT if with_cot else PROMPT_WITHOUT_COT
    prompt = build_prompt(vlm, goal, with_cot=with_cot) if wrapped else template.format(goal=goal)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
    prompt_length = input_ids.shape[1]
    input_ids = input_ids.expand(batch, -1).contiguous()
    pixels = _stack_pixel_values(vlm.vision_backbone.image_transform, frames, "cuda:0")
    pixels = ({k: v.half() for k, v in pixels.items()} if isinstance(pixels, dict)
              else pixels.half())
    generators = [torch.Generator().manual_seed(frame_seed(owner, i)) for i in indices]

    output = vlm(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                 pixel_values=pixels, use_cache=True, output_hidden_states=capture,
                 return_dict=True)
    prefix = (torch.stack([output.hidden_states[layer] for layer in LAYERS], dim=2)
              if capture else None)
    past = output.past_key_values
    finished = torch.zeros(batch, dtype=torch.bool)
    lengths = torch.zeros(batch, dtype=torch.long)
    token, broken = sample(output.logits[:, -1], generators, ~finished, eos_id, top_k)
    del output

    hidden, ids = [], []
    for _ in range(MAX_NEW_TOKENS):
        output = vlm(input_ids=token[:, None].to("cuda:0"), past_key_values=past,
                     use_cache=True, output_hidden_states=capture, return_dict=True)
        past = output.past_key_values
        if capture:
            hidden.append(torch.stack([output.hidden_states[layer][:, 0] for layer in LAYERS],
                                      dim=1))
        ids.append(token)
        lengths += (~finished).long()
        finished |= token == eos_id
        if bool(finished.all()):
            break
        token, more = sample(output.logits[:, -1], generators,
                             (lengths < MIN_NEW_TOKENS) | finished, eos_id, top_k)
        broken += more
        del output

    features = None
    if capture:
        device = prefix.device
        features = _assemble(prefix, [h.to(device) for h in hidden],
                             [t.to(device) for t in ids], lengths.to(device), prompt_length,
                             tokenizer)
    stacked = torch.stack(ids, dim=1)
    texts, counts = [], []
    for i in range(batch):
        row = stacked[i, :int(lengths[i])].tolist()
        if row and row[-1] == eos_id:
            row = row[:-1]
        counts.append(len(row))
        texts.append(tokenizer.decode(row, skip_special_tokens=True).strip())
    return texts, counts, broken, features


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=int, default=40)
    parser.add_argument("--per-trajectory", type=int, default=5)
    parser.add_argument("--layers-on-first", type=int, default=14)
    parser.add_argument("--out", type=Path, default=Path("/data/topovlm/logs/probe_decoding.json"))
    args = parser.parse_args()

    if torch.cuda.device_count() < 2:
        print(f"GPU {torch.cuda.device_count()}장 — 두 장이 필요하다")
        return 1
    bases = {name: pca.load(EMBEDDINGS / f"pr2l_habitat_web_hd_{name}" / "pca.npz")
             for name in ("cot_fixed", "nocot")}

    records = [json.loads(line) for line in MANIFEST.open()]
    chosen = pick(records, args.trajectories, args.per_trajectory)
    print(f"[probe] 궤적 {len(chosen)}개, 프레임 {sum(len(i) for _, i in chosen)}장, "
          f"설정 {len(SETTINGS)}개", flush=True)
    started = time.time()
    vlm = load_split_vlm(args.layers_on_first)
    print(f"[load] {time.time() - started:.0f}s", flush=True)

    rows: dict[str, list[dict]] = defaultdict(list)
    broken_total: dict[str, int] = defaultdict(int)
    moments = {(tokens, layer): pca.Moments() for tokens in ("nocot", "cot") for layer in LAYERS}
    for position, (record, indices) in enumerate(chosen, start=1):
        frames = np.ascontiguousarray(
            np.load(HABITAT_ROOT / record["rgb_path"], mmap_mode="r")[indices])
        for name, with_cot, top_k, wrapped, capture in SETTINGS:
            texts, counts, broken, features = run(vlm, frames, record["object_category"],
                                                  record["episode_id"], indices, with_cot,
                                                  top_k, wrapped, capture)
            broken_total[name] += broken
            for index, text, count in zip(indices, texts, counts):
                rows[name].append({"episode_id": record["episode_id"], "frame": index,
                                   "goal": record["object_category"], "tokens": count,
                                   "text": text, **describe(text)})
            if features:
                group = "cot" if with_cot else "nocot"
                for slot, layer in enumerate(LAYERS):
                    moments[(group, layer)].update(
                        np.concatenate([item.tokens[:, slot] for item in features], axis=0))
        if position % 5 == 0 or position == len(chosen):
            print(f"[probe] {position}/{len(chosen)} 궤적, {time.time() - started:.0f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"\n[probe] 저장: {args.out}")

    print("\n=== ①·③ 답의 성격 (전부 최소 32 · 최대 48 · temperature 0.4) ===")
    print(f"  {'설정':12s} {'토큰':>5s} {'4gram반복':>9s} {'반복문장':>8s} {'쓰레기':>6s} "
          f"{'이유어휘':>8s} {'첫문장뒤 장면명사':>14s} {'NaN':>4s}")
    for name, *_ in SETTINGS:
        r = rows[name]
        print(f"  {name:12s} {np.mean([x['tokens'] for x in r]):5.1f} "
              f"{100 * np.mean([x['repeat4'] for x in r]):8.1f}% "
              f"{100 * np.mean([x['repeated_sentence'] for x in r]):7.1f}% "
              f"{100 * np.mean([x['garbage'] for x in r]):5.1f}% "
              f"{100 * np.mean([x['reason'] for x in r]):7.1f}% "
              f"{np.mean([x['scene_after_first'] for x in r]):13.2f} {broken_total[name]:4d}")

    print("\n=== ② 기저 교차 적용 — 보존 분산 ===")
    print(f"  {'토큰':8s} {'층':>4s} {'no-CoT 기저':>12s} {'CoT 기저':>10s}")
    for group in ("nocot", "cot"):
        for layer in LAYERS:
            m = moments[(group, layer)]
            own = pca.retained_fraction(bases["nocot"], m)
            other = pca.retained_fraction(bases["cot_fixed"], m)
            print(f"  {group:8s} {layer:4d} {100 * own:11.2f}% {100 * other:9.2f}%   "
                  f"(토큰 {m.count})")

    print("\n=== 같은 프레임, 여섯 설정 ===")
    total = len(rows["nocot_full"])
    for j in list(range(0, total, max(total // 5, 1)))[:5]:
        head = rows["nocot_full"][j]
        print(f"\n[{head['goal']}] {head['episode_id'][:20]} 프레임 {head['frame']}")
        for name, *_ in SETTINGS:
            row = rows[name][j]
            print(f"  {name:12s} ({row['tokens']:2d}) {row['text'][:230]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
