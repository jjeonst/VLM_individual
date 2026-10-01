"""Judge Appendix C: does VC-1's training recipe account for the gap to the paper?

Two settings separate this reproduction from the recipe PR2L says it inherited -- a gradient-norm
cap of 0.2 and an AdamW epsilon of 1e-5, neither of which was here. C3 turns both on and lands at
13.6 %, which the paper's 11.6 % cannot be told apart from. What C3 alone cannot say is which of
the two did it, and whether either did anything a different random seed would not have.

So the runs are read in a fixed order, set before any of them started:

  3. |base - base'| against |base - C3|. If changing only the seed moves the number as much as
     changing the recipe does, nothing below can be read. Checked first for that reason.
  1. C3 near 11.6 %                     -> the recipe accounts for the gap.
  2. C1 alone falls / C2 alone falls     -> says which half.
  4. none of them move                   -> the recipe is not the cause.

The expectation written down before running was C3 >> C1: Adam divides the gradient by its own
running magnitude, so with eps at 1e-8 a clipped gradient produces nearly the same step as an
unclipped one. Only when eps is large enough to dominate the denominator does the step become
proportional to the gradient, and only then does a cap bite. C1 falling as far as C3 would mean
that reading is wrong.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from evaluate import CHECKPOINT_ROOT

PAPER, PAPER_N = 11.6, 2000

RUNS = [("기준   (없음 · 1e-8, 시드 0)", "stageB_image", "base"),
        ("기준′  (없음 · 1e-8, 시드 1)", "stageB_imgBase2", "base2"),
        ("C1     (0.2  · 1e-8)", "stageB_imgC1", "c1"),
        ("C2     (없음 · 1e-5)", "stageB_imgC2", "c2"),
        ("C3     (0.2  · 1e-5)", "stageB_imgC3", "c3")]


def rate(name: str) -> tuple[float, int] | None:
    for candidate in (f"{name}_eval.json", f"{name}_eval_val.json"):
        path = CHECKPOINT_ROOT / candidate
        if path.exists():
            rows = json.loads(path.read_text())
            return 100 * float(np.mean([r["success"] for r in rows])), len(rows)
    return None


def se(p: float, n: int) -> float:
    return 100 * math.sqrt((p / 100) * (1 - p / 100) / n)


def gap(a: tuple[float, int], b: tuple[float, int], label: str) -> float:
    d = a[0] - b[0]
    s = math.sqrt(se(*a) ** 2 + se(*b) ** 2)
    print(f"  {label:30s} {d:+6.1f}%p  ±{s:.2f}p  {abs(d) / s:4.1f}σ")
    return abs(d)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=float, default=3.0)
    args = parser.parse_args()

    found: dict[str, tuple[float, int]] = {}
    print(f"{'실행':30s} {'성공률':>8s} {'에피소드':>8s} {'논문 대비':>9s}")
    for label, name, key in RUNS:
        got = rate(name)
        if got is None:
            print(f"  {label:28s} {'미완료':>8s}")
            continue
        found[key] = got
        print(f"  {label:28s} {got[0]:7.1f}% {got[1]:8d} {got[0] / PAPER:8.2f}배")
    print(f"  {'논문':28s} {PAPER:7.1f}% {PAPER_N:8d}")

    need = {"base", "base2", "c3"}
    if not need <= set(found):
        print(f"\n[C] 아직 판정할 수 없다 — {sorted(need - set(found))} 이 없다")
        return 1

    print("\n=== 비교 ===")
    seed_gap = gap(found["base"], found["base2"], "시드  |기준 - 기준′|")
    recipe_gap = gap(found["base"], found["c3"], "레시피 |기준 - C3|")
    gap(found["c3"], (PAPER, PAPER_N), "C3 - 논문")
    for key, label in (("c1", "클리핑만 |기준 - C1|"), ("c2", "eps만   |기준 - C2|")):
        if key in found:
            gap(found["base"], found[key], label)

    print("\n=== 판정 ===")
    # Criterion 3 first: it decides whether the rest can be read at all.
    if seed_gap >= recipe_gap:
        print("  ③ 검정력 부족 — 시드만 바꿔도 레시피를 바꾼 만큼 움직인다.")
        print("     시드를 더 돌리기 전에는 나머지를 읽을 수 없다.")
        return 1
    print(f"  ③ 통과 — 레시피 효과가 시드 효과의 {recipe_gap / max(seed_gap, 1e-9):.1f}배")

    near = abs(found["c3"][0] - PAPER) <= args.tolerance
    if near:
        covered = found["base"][0] - found["c3"][0]
        total = found["base"][0] - PAPER
        print(f"  ① C3가 논문 부근이다 — 격차 {total:.1f}%p 중 {covered:.1f}%p "
              f"({100 * covered / total:.0f}%)를 레시피가 덮는다.")
    else:
        print(f"  ④ C3({found['c3'][0]:.1f}%)가 논문 부근이 아니다. 레시피는 격차를 "
              "설명하지 못한다.")

    if {"c1", "c2"} <= set(found):
        d1 = found["base"][0] - found["c1"][0]
        d2 = found["base"][0] - found["c2"][0]
        print(f"\n  ② 분해 — 클리핑만 {d1:+.1f}%p | eps만 {d2:+.1f}%p | "
              f"둘 다 {found['base'][0] - found['c3'][0]:+.1f}%p")
        # A sum well below the joint effect means the two are not independent -- which is what
        # the mechanism predicts, eps being what lets the cap take hold.
        joint = found["base"][0] - found["c3"][0]
        print(f"     단독 효과의 합 {d1 + d2:+.1f}%p 대 함께 켰을 때 {joint:+.1f}%p")
        if d1 + d2 < joint - 2:
            print("     -> 함께 켰을 때가 더 크다. 두 설정이 상호작용한다는 뜻이며,")
            print("        eps가 커야 클리핑이 실제로 물린다는 읽기와 맞는다.")
        elif abs(d1) > abs(d2):
            print("     -> 클리핑이 주된 몫이다.")
        else:
            print("     -> eps가 주된 몫이다.")
    else:
        print("\n  ② C1·C2가 아직 없어 분해할 수 없다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
