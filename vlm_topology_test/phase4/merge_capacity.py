"""Judge Appendix X: does the recurrent size explain the gap to the paper?

The three criteria were fixed before the runs started (Appendix X.4) and are applied here in the
order they have to be read:

  3. |L - L'| as large as |L - S|  -> the seed alone moves the number as much as the size does,
     so nothing can be concluded from either. This is checked first because it decides whether
     the other two mean anything.
  1. L, M and S within 3 points    -> capacity does not set the success rate in this range, and
     no size the paper might have used produces 11.6 %. The hypothesis is rejected.
  2. S falls to 12-16 %            -> capacity is a live explanation, and the drop says how much
     of the gap it covers.

XS, with no recurrence at all, is not part of that test. It measures something else: how much of
the success rate the memory is responsible for, which is a floor rather than a comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate import CHECKPOINT_ROOT

PAPER = 11.6

# The reference L is the original run at the paper-derived size. It was evaluated on the same
# 500-episode subset, so the new runs are read against that number rather than the 2,000-episode
# one (24.2 %), which would compare different episode sets.
RUNS = [("L  (2048x2, 시드 0)", "stageB_image", 78_848_548),
        ("L' (2048x2, 시드 1)", "stageB_imgL2", 78_848_548),
        ("M  (1024x2)", "stageB_imgM", 36_360_740),
        ("S  ( 512x1)", "stageB_imgS", 22_321_428),
        ("XS (순환 없음)", "stageB_imgXS", 19_042_836)]


def rate_of(name: str) -> tuple[float, int] | None:
    """Success rate from whichever file that run wrote."""
    for candidate in (f"{name}_eval.json", f"{name}_eval_val.json"):
        path = CHECKPOINT_ROOT / candidate
        if path.exists():
            rows = json.loads(path.read_text())
            return 100 * float(np.mean([r["success"] for r in rows])), len(rows)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=float, default=3.0)
    args = parser.parse_args()

    found = {}
    print(f"{'실행':22s} {'파라미터':>12s} {'에피소드':>8s} {'성공률':>8s} {'논문 대비':>10s}")
    for label, name, params in RUNS:
        got = rate_of(name)
        if got is None:
            print(f"{label:22s} {params:12,d} {'미완료':>8s}")
            continue
        rate, count = got
        found[label.split()[0]] = rate
        print(f"{label:22s} {params:12,d} {count:8d} {rate:7.1f}% {rate / PAPER:9.2f}배")
    print(f"{'논문':22s} {'미명시':>12s} {2000:8d} {PAPER:7.1f}%")

    need = {"L", "L'", "S"}
    if not need <= set(found):
        print(f"\n[X] 아직 판정할 수 없다 — {sorted(need - set(found))} 이 없다")
        return 1

    seed_gap = abs(found["L"] - found["L'"])
    size_gap = abs(found["L"] - found["S"])
    print(f"\n=== 판정 ===")
    print(f"  시드 차이 |L - L'|   {seed_gap:5.1f}%p")
    print(f"  크기 차이 |L - S|    {size_gap:5.1f}%p")

    # Criterion 3 first: without it the other two cannot be read.
    if seed_gap >= size_gap:
        print("\n  ③ 검정력 부족 — 초기화만 바꿔도 크기를 바꾼 만큼 움직인다.")
        print("     시드를 더 돌리기 전에는 ①·②를 읽을 수 없다.")
        return 1

    sizes = [found[k] for k in ("L", "M", "S") if k in found]
    spread = max(sizes) - min(sizes)
    if spread <= args.tolerance:
        print(f"\n  ① L·M·S 가 {spread:.1f}%p 안에 모였다 (허용 {args.tolerance:.0f}%p).")
        print("     이 범위에서 순환 용량은 성공률을 좌우하지 않는다. 용량 가설 기각.")
        print(f"     논문이 어떤 크기를 썼든 {PAPER}%를 만들지 못한다.")
    elif 12.0 <= found["S"] <= 16.0:
        print(f"\n  ② S 가 {found['S']:.1f}%로 논문 부근까지 떨어졌다.")
        print(f"     용량이 살아 있는 설명이며, 격차 {found['L'] - PAPER:.1f}%p 중 "
              f"{found['L'] - found['S']:.1f}%p 를 덮는다.")
    else:
        print(f"\n  판정 보류 — L·M·S 가 {spread:.1f}%p 로 흩어졌으나 S({found['S']:.1f}%)가 "
              f"논문 부근(12~16%)도 아니다.")
        print("     용량이 성공률을 움직이기는 하나 격차를 설명하지는 못한다.")

    if "XS" in found:
        print(f"\n  순환의 몫: L {found['L']:.1f}% - XS {found['XS']:.1f}% = "
              f"{found['L'] - found['XS']:.1f}%p")
        print("     크기를 줄이는 것과 순환을 없애는 것은 다른 일이며, 이 값이 후자의 크기다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
