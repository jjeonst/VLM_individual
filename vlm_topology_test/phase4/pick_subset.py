"""Name a subset of an evaluation that already ran, so a cheaper run can be compared against it.

The epoch-40 numbers were measured on 500 episodes. Re-measuring intermediate checkpoints on all
500 costs tens of GPU-hours for the CoT condition, and the question those checkpoints answer --
where the success curve crosses the paper's number -- does not need that precision.

What it does need is the *same* episodes. `stratified_subset` cannot supply them: it draws from
one generator at a size that depends on the count, so its 200 are not 200 of its 500. Asking it
for a smaller sample would produce a curve whose last point was measured on different episodes
from the rest, and the difference between two such points is not a convergence effect.

So the subset is drawn from the finished run's own rows, keeping each target object's share, and
written out by name. The intermediate checkpoints are then given that list, and the epoch-40
value on the same episodes is recomputed from the stored per-episode results at no cost.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate import CHECKPOINT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-run", default="stageB_cot",
                        help="the finished run whose episodes define the pool")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path,
                        default=CHECKPOINT_ROOT / "subset200.json")
    args = parser.parse_args()

    rows = json.loads((CHECKPOINT_ROOT / f"{args.from_run}_eval_val.json").read_text())
    print(f"[subset] {args.from_run}: {len(rows)} 에피소드에서 {args.count}개를 고른다")

    by_object = defaultdict(list)
    for row in rows:
        by_object[row["object_category"]].append(row)

    rng = np.random.default_rng(args.seed)
    chosen: list[dict] = []
    for name, group in sorted(by_object.items()):
        take = max(1, round(args.count * len(group) / len(rows)))
        picked = rng.choice(len(group), size=min(take, len(group)), replace=False)
        chosen.extend(group[i] for i in sorted(picked))

    ids = sorted(row["episode_id"] for row in chosen)
    args.out.write_text(json.dumps(ids, indent=0) + "\n")

    print(f"\n[subset] {len(ids)} 에피소드 -> {args.out}")
    print(f"{'물체':14s} {'전체':>6s} {'부분':>6s} {'비율':>8s}  성공률 전체 / 부분")
    for name in sorted(by_object):
        whole = by_object[name]
        part = [r for r in chosen if r["object_category"] == name]
        print(f"  {name:12s} {len(whole):6d} {len(part):6d} "
              f"{100 * len(part) / len(whole):7.1f}%  "
              f"{100 * np.mean([r['success'] for r in whole]):5.1f}% / "
              f"{100 * np.mean([r['success'] for r in part]):5.1f}%")

    # The whole point of the subset is that the finished run's value on it stands in for the
    # value on all 500, so the two are printed together rather than left to be trusted.
    print(f"\n[subset] {args.from_run} 성공률: 전체 500 "
          f"{100 * np.mean([r['success'] for r in rows]):.1f}%  |  "
          f"부분 {len(chosen)} {100 * np.mean([r['success'] for r in chosen]):.1f}%")

    for other in ("stageB_image", "stageB_zero"):
        path = CHECKPOINT_ROOT / f"{other}_eval_val.json"
        if not path.exists():
            continue
        part = [r for r in json.loads(path.read_text()) if r["episode_id"] in set(ids)]
        whole = json.loads(path.read_text())
        print(f"[subset] {other:14s} 성공률: 전체 {len(whole)} "
              f"{100 * np.mean([r['success'] for r in whole]):.1f}%  |  "
              f"부분 {len(part)} {100 * np.mean([r['success'] for r in part]):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
