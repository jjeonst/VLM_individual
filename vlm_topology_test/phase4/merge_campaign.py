"""Read out the 2026-09-10 campaign: CoT and no-CoT under VC-1's recipe, and the Z.2 control.

Three numbers were outstanding, each with a prediction written down before it arrived:

  * CoT + recipe     -- a uniform *ratio* (x0.540, from the image encoder) predicted 32.5 %
  * no-CoT + recipe  -- once CoT came in at 47.4 % the additive reading (a constant ~12 %p taken
                        off) became the live one; it predicts 27.8 - (60.2 - 47.4) ... except no
                        no-CoT baseline without the recipe exists, so the test here is against
                        the paper's 27.8 % and against CoT's own gap
  * Z.2 control      -- rounding the live representation to float16 should leave 25.2 % alone

Shard files are checked for identity before anything is summed: the no-CoT and CoT runs share
episode lists and field layout, so a file written under the wrong run name would look plausible.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data/topovlm/checkpoints/pr2l_phase4")


def se(p: float, n: int) -> float:
    return 100 * math.sqrt((p / 100) * (1 - p / 100) / n)


def gap(a: float, na: int, b: float, nb: int, label: str) -> None:
    d = a - b
    s = math.sqrt(se(a, na) ** 2 + se(b, nb) ** 2)
    print(f"  {label:34s} {d:+6.1f}%p  ±{s:.2f}p  {abs(d) / s:4.1f}σ")


def shards(run: str) -> list[list[dict]]:
    return [json.loads((ROOT / f"{run}_eval_shard{s:02d}.json").read_text()) for s in range(4)]


def rate(rows: list[dict]) -> float:
    return 100 * float(np.mean([r["success"] for r in rows]))


def first(row: dict, names: tuple[str, ...]):
    for name in names:
        if name in row:
            return row[name]
    return None


def breakdown(rows: list[dict], label: str) -> None:
    by_goal, by_dist = defaultdict(list), defaultdict(list)
    for r in rows:
        by_goal[first(r, ("goal", "object", "target", "object_category"))].append(r["success"])
        dist = first(r, ("start_distance", "geodesic_distance", "start_geodesic",
                         "distance_to_goal_start"))
        if dist is not None:
            b = "0-2" if dist < 2 else "2-4" if dist < 4 else "4-6" if dist < 6 else "6+"
            by_dist[b].append(r["success"])
    print(f"  [{label}] 물체별  " + "  ".join(
        f"{k} {100 * np.mean(v):.0f}%({len(v)})" for k, v in sorted(by_goal.items(), key=lambda kv: str(kv[0]))))
    if by_dist:
        print(f"  [{label}] 거리별  " + "  ".join(
            f"{b}m {100 * np.mean(by_dist[b]):.0f}%({len(by_dist[b])})"
            for b in ("0-2", "2-4", "4-6", "6+") if b in by_dist))


def main() -> int:
    print("=== 샤드 파일이 서로 다른 실행의 것인가 ===")
    for s in range(4):
        a = ROOT / f"stageB_nocotC3_eval_shard{s:02d}.json"
        b = ROOT / f"stageB_cotFixC3_eval_shard{s:02d}.json"
        ha, hb = hashlib.md5(a.read_bytes()).hexdigest(), hashlib.md5(b.read_bytes()).hexdigest()
        verdict = "동일 파일 — 중단" if ha == hb else "다름"
        print(f"  샤드 {s}: nocot {a.stat().st_size:6d}B  CoT {b.stat().st_size:6d}B  {verdict}")
        if ha == hb:
            return 1

    nocot_shards, cot_shards = shards("stageB_nocotC3"), shards("stageB_cotFixC3")
    nocot = [r for part in nocot_shards for r in part]
    cot = [r for part in cot_shards for r in part]
    z2 = json.loads((ROOT / "stageB_image_matchprec_eval.json").read_text())
    (ROOT / "stageB_nocotC3_eval_val.json").write_text(json.dumps(nocot))

    print("\n키:", sorted(nocot[0].keys()))
    print("\n=== 샤드별 ===")
    for label, parts in (("nocot", nocot_shards), ("CoT", cot_shards)):
        print(f"  {label:6s} " + "  ".join(f"{len(p)}개 {rate(p):4.1f}%" for p in parts))

    print("\n=== 전체 ===")
    for label, rows in (("nocot + 레시피", nocot), ("CoT + 레시피", cot),
                        ("Z.2 대조 (이미지, 반올림)", z2)):
        spl = np.mean([r.get("spl", 0) for r in rows])
        stop = 100 * np.mean([first(r, ("stopped", "called_stop")) or 0 for r in rows])
        print(f"  {label:26s} {len(rows)} 에피소드  {rate(rows):5.1f}%  (±{se(rate(rows), len(rows)):.1f}p)"
              f"  SPL {spl:.3f}  정지 {stop:.1f}%")

    pn, pc, pz = rate(nocot), rate(cot), rate(z2)
    print("\n=== 비교 ===")
    gap(pn, 500, 27.8, 2000, "nocot+레시피 − 논문 27.8")
    gap(pc, 500, 41.9, 2000, "CoT+레시피 − 논문 41.9")
    gap(pc, 500, pn, 500, "CoT − nocot (둘 다 레시피)")
    print(f"  {'논문 CoT − nocot':34s} {41.9 - 27.8:+6.1f}%p")
    gap(pz, 500, 25.2, 500, "Z.2: 반올림 켬 − 기준 25.2")

    print("\n=== 세부 ===")
    breakdown(nocot, "nocot")
    breakdown(cot, "CoT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
