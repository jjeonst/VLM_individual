"""Success rate against training epoch, for each recurrent size (follow-up to Appendix X-R).

X-R read S's 13.0 % as a ceiling set by capacity, and the whole conclusion rests on that reading.
There is a second reading that fits the same number: S is not capped, only slower, and at forty
epochs it simply has not arrived yet. Those two say opposite things about the paper -- the first
that its policy was smaller, the second that its policy was trained less -- and the difference
between them is visible in the shape of the curve rather than its endpoint.

L settled by epoch 30 (14.0, 20.2, 25.2, 25.2). If S has settled too, capacity sets the ceiling.
If S is still climbing at 40, its endpoint is an artefact of the epoch budget and the criterion-2
verdict has to be reopened.

The paper offers nothing to compare a curve against: it reports no training accuracy, no loss,
and its two Habitat figures are bar charts of final success rates. What can be asked is only
whether *our* curves have flattened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate import CHECKPOINT_ROOT

EPOCHS = (10, 20, 30, 40)
# L is the original run at the paper-derived size; its curve was measured earlier (V.7).
KNOWN_L = {10: 14.0, 20: 20.2, 30: 25.2, 40: 25.2}
RUNS = [("L  2048x2", None), ("L' 2048x2", "stageB_imgL2"), ("M  1024x2", "stageB_imgM"),
        ("S   512x1", "stageB_imgS"), ("XS   없음", "stageB_imgXS")]
PAPER = 11.6


def rate(name: str, epoch: int) -> float | None:
    stem = name if epoch == 40 else f"{name}_epoch{epoch:03d}"
    for candidate in (f"{stem}_eval.json", f"{stem}_eval_val.json"):
        path = CHECKPOINT_ROOT / candidate
        if path.exists():
            rows = json.loads(path.read_text())
            return 100 * float(np.mean([r["success"] for r in rows]))
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500,
                        help="각 점을 잰 에피소드 수. 포화 기준은 이 값이 정하는 오차에서 나온다")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="30→40 변화가 결합 표준오차의 이 배수 안이면 포화로 본다")
    args = parser.parse_args()

    print(f"{'실행':12s}" + "".join(f"{f'ep{e}':>9s}" for e in EPOCHS) + f"{'30→40':>9s}")
    curves: dict[str, dict[int, float]] = {}
    for label, name in RUNS:
        series = KNOWN_L if name is None else {e: rate(name, e) for e in EPOCHS}
        curves[label] = series
        cells = "".join(f"{series[e]:8.1f}%" if series.get(e) is not None else f"{'—':>9s}"
                        for e in EPOCHS)
        tail = (f"{series[40] - series[30]:+8.1f}p"
                if series.get(40) is not None and series.get(30) is not None else f"{'—':>9s}")
        print(f"  {label:10s}{cells}{tail}")
    print(f"  {'논문':10s}" + f"{'':27s}" + f"{PAPER:8.1f}%")

    print("\n=== 포화 판정 ===")
    # The verdict turns on S alone: it is the run whose endpoint carries the criterion-2 reading.
    # Absence of its curve is not evidence that it settled, so the default is "cannot judge"
    # rather than "passed" -- an earlier version defaulted the other way and announced the
    # conclusion off L's curve while S had not been measured at all.
    settled_S: bool | None = None
    for label, series in curves.items():
        if series.get(40) is None or series.get(30) is None:
            print(f"  {label:10s} 미완료")
            continue
        delta = series[40] - series[30]
        # A fixed percentage-point threshold cannot be read against measurements this noisy. At
        # 500 episodes each point carries about +/-1.9 %p and their difference about +/-2.6 %p,
        # so the first version's flat 1.5 %p bar sat below the precision of the thing it was
        # judging -- it would have called a real plateau "still climbing" as often as not. The
        # bar is now the combined standard error itself.
        se = np.sqrt(sum(p * (100 - p) / args.episodes for p in (series[30], series[40])))
        settled = abs(delta) <= args.sigma * se
        print(f"  {label:10s} 30→40 {delta:+5.1f}%p  ±{se:.1f}p  {abs(delta) / se:.1f}σ  "
              f"{'포화와 부합' if settled else '아직 오르는 중'}")
        if label.startswith("S "):
            settled_S = settled

    print()
    if settled_S is None:
        print("  → S의 30·40 epoch 값이 아직 없어 판정할 수 없다. 곡선의 모양이 아니라")
        print("    끝값만 있는 상태이며, 그것으로는 천장인지 통과점인지 가릴 수 없다.")
        return 1
    if settled_S:
        print("  → 작은 정책의 30→40 변화가 측정 오차 안이다. '아직 오르는 중이라 끝값이 낮다'는")
        print("    반대 설명을 지지하는 증거가 없으므로 부록 X-R의 ② 판정은 선다.")
        print("    다만 이것은 포화와 '부합'하는 것이지 포화를 적극적으로 입증한 것은 아니다 —")
        print(f"    {args.episodes} 에피소드로는 오차 안의 변화를 볼 수 없다. 엄밀히 하려면")
        print("    ep30·ep40을 2,000 에피소드로 다시 재야 한다(오차가 절반으로 준다).")
    else:
        print("  → S가 40 epoch에서도 아직 오르고 있다. 13.0 %는 천장이 아니라 통과점일 수 있고,")
        print("    그렇다면 '논문이 작은 망을 썼다'와 '논문이 덜 학습했다'가 구분되지 않는다.")
        print("    X-R의 ② 판정을 보류하고 S를 더 오래 학습시켜 다시 봐야 한다.")

    print("\n  주의: 논문에는 학습 곡선도 학습 정확도도 없다(Habitat 관련 그림 둘은 모두 최종")
    print("  성공률 막대그래프). 여기서 비교되는 것은 우리 실행들끼리이며, 논문 곡선과의 대조가")
    print("  아니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
