"""Experiments 5-A and 5-B of Appendix W -- are the two conditions actually comparable?

The claim under test is not about navigation at all. It is that when we report "CoT 60.8 % vs
image encoder 25.5 %", the two numbers differ *only* in the representation fed to the policy.
Two ways that could quietly be false:

5-A  The conditions were trained on different trajectories. Step counts were checked after the
     duplicate-manifest incident, but matching totals are not matching sets -- one condition
     could have dropped trajectory X and picked up trajectory Y and still balance.

5-B  The policies differ by more than the input projection. Appendix C.2 item 5 makes holding
     the architecture fixed a stated control, so any second difference breaks the comparison.
     The widths differ by construction (2048 vs 2176), so the parameter counts must differ by
     exactly (2176 - 2048) * 1024 and by nothing else.

Neither needs a GPU or the embeddings themselves -- 5-A reads manifests, 5-B builds the two
policies on the meta device and counts. Both are falsifiers: they can only fail loudly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dataset import read_manifest
from policy import NavigationPolicy

EMBEDDING_ROOT = Path("/data/topovlm/habitat/embeddings")
COT = "pr2l_habitat_web_hd_cot"
IMAGE = "pr2l_habitat_web_hd_image_encoder"


def experiment_5a(root: Path, cot: str, image: str) -> bool:
    """The two conditions must name the same trajectories, not merely the same number of them."""
    print("=" * 72)
    print("5-A  두 조건의 학습 집합이 문자 그대로 같은가")
    print("=" * 72)

    rows = {name: read_manifest(root / name) for name in (cot, image)}
    ids = {name: {r["episode_id"] for r in part} for name, part in rows.items()}
    steps = {name: sum(r["steps"] for r in part) for name, part in rows.items()}

    for name in (cot, image):
        print(f"  {name:40s} 궤적 {len(rows[name]):6d} | 고유 {len(ids[name]):6d} | "
              f"스텝 {steps[name]:,}")
        if len(rows[name]) != len(ids[name]):
            print(f"    중복이 남아 있다 ({len(rows[name]) - len(ids[name])}건)")

    only_cot = ids[cot] - ids[image]
    only_image = ids[image] - ids[cot]
    print(f"\n  CoT에만 {len(only_cot)}건 | 이미지에만 {len(only_image)}건")
    for name in sorted(only_cot)[:5]:
        print(f"    CoT만:    {name}")
    for name in sorted(only_image)[:5]:
        print(f"    이미지만: {name}")

    # A trajectory present in both but at a different length means one of the two encodings ran
    # against a different replay, which would be a worse fault than a missing trajectory.
    length = {name: {r["episode_id"]: r["steps"] for r in part} for name, part in rows.items()}
    shared = ids[cot] & ids[image]
    mismatched = [e for e in shared if length[cot][e] != length[image][e]]
    print(f"  공통 궤적 {len(shared)}건 중 길이가 다른 것 {len(mismatched)}건")
    for name in sorted(mismatched)[:5]:
        print(f"    {name}: CoT {length[cot][name]} vs 이미지 {length[image][name]}")

    passed = not only_cot and not only_image and not mismatched
    print(f"\n  판정: {'통과' if passed else '실패'}")
    return passed


def experiment_5b(cot_width: int, image_width: int) -> bool:
    """Every parameter outside the input projection must be shared between the conditions."""
    print()
    print("=" * 72)
    print("5-B  정책 파라미터가 입력 투영만 빼고 같은가")
    print("=" * 72)

    # The policies are only counted, never run, so no weights need to be allocated.
    with torch.device("meta"):
        built = {width: NavigationPolicy(token_dim=width) for width in (cot_width, image_width)}

    shapes = {width: {name: tuple(p.shape) for name, p in model.named_parameters()}
              for width, model in built.items()}
    totals = {width: sum(p.numel() for p in model.parameters())
              for width, model in built.items()}

    for width in (cot_width, image_width):
        print(f"  폭 {width:5d} | 파라미터 {totals[width]:,}")

    names = {width: set(shape) for width, shape in shapes.items()}
    only_cot = names[cot_width] - names[image_width]
    only_image = names[image_width] - names[cot_width]
    differing = [n for n in names[cot_width] & names[image_width]
                 if shapes[cot_width][n] != shapes[image_width][n]]

    print(f"\n  CoT에만 있는 파라미터 {len(only_cot)}개 | 이미지에만 {len(only_image)}개")
    print(f"  이름이 같은데 모양이 다른 것 {len(differing)}개")
    for name in sorted(differing):
        print(f"    {name}: {shapes[cot_width][name]} vs {shapes[image_width][name]}")

    expected = (image_width - cot_width) * 1024
    actual = totals[image_width] - totals[cot_width]
    print(f"\n  기대 차이 ({image_width} - {cot_width}) x 1024 = {expected:,}")
    print(f"  실제 차이                          = {actual:,}")

    # One differing parameter is expected -- the projection weight. A bias difference or a second
    # differing tensor would mean the width leaked somewhere it should not have.
    passed = (not only_cot and not only_image and len(differing) == 1 and actual == expected)
    print(f"\n  판정: {'통과' if passed else '실패'}")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=EMBEDDING_ROOT)
    parser.add_argument("--cot", default=COT)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--cot-width", type=int, default=2048)
    parser.add_argument("--image-width", type=int, default=2176)
    args = parser.parse_args()

    a = experiment_5a(args.root, args.cot, args.image)
    b = experiment_5b(args.cot_width, args.image_width)

    print()
    print(f"5-A {'통과' if a else '실패'} | 5-B {'통과' if b else '실패'}")
    return 0 if (a and b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
