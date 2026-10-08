"""The language half of stored frames, for the language-ablation evaluation.

The question this serves is whether a trained PR2L policy uses the tokens that come after the
image at all -- the question and the generated answer. Every condition so far lands within
1.4 %p of every other (CoT 47.4, no-CoT 46.6, no-CoT without a minimum length 46.0) although
the generated text differs from a 48-token explanation to a 12-token sentence, which is what
one would see if the policy read only the sixteen visual tokens. Those sit before the question
and, the model being causal, are bit-for-bit identical across conditions.

The test keeps the live visual tokens and swaps everything after them for the same span taken
from a *different* frame of the training set. The donor is drawn from demonstrations hunting
the same object, so the question it carries names the right goal and the only thing destroyed
is what the language tokens say about this particular image. Swapping rather than blanking keeps
the token count and the value distribution the policy was trained on.

The pool is built once, here, so that every evaluation shard reads one file instead of a few
hundred trajectory archives. Selection is seeded and recorded, so it can be rebuilt exactly.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset  # noqa: E402
from vlm_features import POOLED_VISUAL_TOKENS  # noqa: E402

HABITAT_ROOT = Path("/data/topovlm/habitat")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--trajectories", type=int, default=30, help="per goal object")
    parser.add_argument("--frames", type=int, default=50, help="per trajectory")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = HABITAT_ROOT / "embeddings" / f"pr2l_habitat_web_hd_{args.condition}"
    records = dataset.read_manifest(root)
    by_goal = defaultdict(list)
    for record in records:
        by_goal[record["object_category"]].append(record)

    rng = np.random.default_rng(args.seed)
    arrays, chosen = {}, {}
    for goal in sorted(by_goal):
        group = by_goal[goal]
        picks = rng.choice(len(group), size=min(args.trajectories, len(group)), replace=False)
        spans, lengths, names = [], [], []
        for index in sorted(picks):
            record = group[index]
            tokens, offsets = dataset.read_tokens(root / "train" / f"{record['episode_id']}.npz")
            steps = len(offsets) - 1
            frames = rng.choice(steps, size=min(args.frames, steps), replace=False)
            for frame in sorted(frames):
                span = tokens[offsets[frame] + POOLED_VISUAL_TOKENS:offsets[frame + 1]]
                spans.append(np.asarray(span))
                lengths.append(len(span))
            names.append(record["episode_id"])
        arrays[f"{goal}_tokens"] = np.concatenate(spans, axis=0)
        arrays[f"{goal}_offsets"] = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        chosen[goal] = names
        print(f"[pool] {goal:11s} 궤적 {len(names)}개, 프레임 {len(lengths)}개, "
              f"언어 토큰 평균 {np.mean(lengths):.1f} (최소 {min(lengths)}, 최대 {max(lengths)})",
              flush=True)

    out = root / "language_pool.npz"
    np.savez(out, **arrays)
    (root / "language_pool.json").write_text(json.dumps(
        {"seed": args.seed, "trajectories": args.trajectories, "frames": args.frames,
         "chosen": chosen}, indent=1) + "\n")
    print(f"[pool] {out} 저장 ({out.stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
