"""Join a sharded evaluation back together and report it as one run.

Splitting the evaluation by scene is exact rather than approximate -- the 500 episodes are chosen
before the split, no episode spans two scenes, and each shard writes one row per episode it ran
-- so the join is a concatenation. What needs checking is that the shards between them cover the
selection once: a missing shard leaves a gap that a success rate will happily average over, and
a repeated one silently double-weights whichever scenes it held.

Both are checked here against the same subset the shards were drawn from, and a mismatch is an
error rather than a warning.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluate import CHECKPOINT_ROOT, load_episodes, limit_scenes, stratified_subset, summarise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    shards = sorted(CHECKPOINT_ROOT.glob(f"{args.run_name}_eval_shard*.json"))
    if not shards:
        print(f"[merge] {args.run_name} 의 샤드 결과가 없다")
        return 1

    rows: list[dict] = []
    for path in shards:
        part = json.loads(path.read_text())
        print(f"[merge] {path.name}: {len(part)} 에피소드")
        rows.extend(part)

    ids = [row["episode_id"] for row in rows]
    duplicates = [name for name, count in Counter(ids).items() if count > 1]
    expected = stratified_subset(limit_scenes(load_episodes(args.split), None), args.episodes)
    wanted = {episode["episode_id"] for episode in expected}
    missing = wanted - set(ids)

    print(f"\n[merge] 샤드 {len(shards)}개 | 에피소드 {len(rows)} | 고유 {len(set(ids))} | "
          f"기대 {len(wanted)}")
    if duplicates or missing or len(rows) != len(wanted):
        print(f"[merge] 실패 — 중복 {len(duplicates)}개, 누락 {len(missing)}개")
        for name in (duplicates[:5] + sorted(missing)[:5]):
            print(f"    {name}")
        return 1

    summarise(rows)
    out = args.out or CHECKPOINT_ROOT / f"{args.run_name}_eval_val.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[merge] {out} 저장 ({len(shards)}개 샤드를 합침)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
