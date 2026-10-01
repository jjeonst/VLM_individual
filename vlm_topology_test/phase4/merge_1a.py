"""Join experiment 1-A's shards and judge it against our own loop.

The criterion is the aggregate success rate over the whole validation split, not a per-episode
match -- Appendix V.9 showed the two loops feed the policy identical inputs and still part ways
when a four-way argmax flips on a near tie, so pairing episodes is not available. What has to
agree is the rate, and the rows are printed side by side only to show where the two runs put
their successes, never to require that they be the same episodes.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate import CHECKPOINT_ROOT, DISTANCE_BINS


def summarise(name: str, rows: list[dict]) -> dict:
    rate = 100 * np.mean([r["success"] for r in rows])
    print(f"\n=== {name} · {len(rows)} 에피소드 ===")
    print(f"  성공률 {rate:.1f}%  SPL {np.mean([r['spl'] for r in rows]):.3f}")

    by_object = defaultdict(list)
    for row in rows:
        by_object[row["object_category"]].append(row["success"])
    for key in sorted(by_object):
        group = by_object[key]
        print(f"    {key:12s} {len(group):4d}  {100 * np.mean(group):5.1f}%")

    print("  출발 거리별")
    for low, high in DISTANCE_BINS:
        group = [r["success"] for r in rows if low <= r["start_geodesic"] < high]
        if not group:
            continue
        label = f"{low}-{high} m" if np.isfinite(high) else f"{low}+ m"
        print(f"    {label:10s} {len(group):4d}  {100 * np.mean(group):5.1f}%")
    return {"rate": rate, "n": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, default=CHECKPOINT_ROOT,
                        help="directory holding validate_1a_shard0*.json")
    parser.add_argument("--compare", type=Path,
                        default=CHECKPOINT_ROOT / "stageB_image_full_eval_val.json")
    parser.add_argument("--out", type=Path, default=CHECKPOINT_ROOT / "validate_1a.json")
    parser.add_argument("--tolerance", type=float, default=3.0)
    args = parser.parse_args()

    paths = sorted(args.shards.glob("validate_1a_shard0*.json"))
    if not paths:
        print("[1-A] 샤드 결과가 없다")
        return 1

    rows: list[dict] = []
    for path in paths:
        part = json.loads(path.read_text())
        print(f"[1-A] {path.name}: {len(part)} 에피소드")
        rows.extend(part)

    ids = [r["episode_id"] for r in rows]
    duplicates = len(ids) - len(set(ids))
    print(f"\n[1-A] 샤드 {len(paths)}개 | 에피소드 {len(rows)} | 고유 {len(set(ids))}")
    if duplicates:
        print(f"[1-A] 실패 — 중복 {duplicates}건. 샤드가 겹쳤다")
        return 1

    theirs = summarise("habitat-lab 환경", rows)

    if not args.compare.exists():
        print(f"\n[1-A] 비교 대상 {args.compare} 이 없다")
        return 1
    mine = json.loads(args.compare.read_text())
    ours = summarise("우리 루프", mine)

    if ours["n"] != theirs["n"]:
        print(f"\n[1-A] 주의 — 에피소드 수가 다르다 ({ours['n']} vs {theirs['n']}). "
              "같은 집합이 아니면 두 비율을 견줄 수 없다")

    gap = theirs["rate"] - ours["rate"]
    # Two independent runs over the same episodes, so the standard error of the difference is
    # the two combined. It is printed because "within 3 points" means little without knowing
    # how tightly either number is pinned.
    se = float(np.sqrt(sum(p * (100 - p) / n for p, n in
                           ((theirs["rate"], theirs["n"]), (ours["rate"], ours["n"])))))
    print(f"\n=== 판정 ===")
    print(f"  habitat-lab   {theirs['rate']:5.1f}%")
    print(f"  우리 루프      {ours['rate']:5.1f}%")
    print(f"  차이          {gap:+5.1f}%p   결합 표준오차 ±{se:.1f}%p  ({abs(gap) / se:.1f}σ)")
    ok = abs(gap) <= args.tolerance
    print(f"  판정: {'통과' if ok else '실패'} (허용 ±{args.tolerance:.0f}%p)")
    if ok:
        print("\n  → 평가 루프가 관대해서 높다는 설명은 닫힌다. 표준 구현이 같은 수치를 낸다.")
    else:
        print("\n  → 이 차이가 우리 평가 루프의 편향이다. 물체별·거리별 표를 대조해 어디서")
        print("    벌어지는지 좁힐 것.")

    args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[1-A] {args.out} 저장")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
