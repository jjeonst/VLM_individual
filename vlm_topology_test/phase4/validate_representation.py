"""Experiment 3-A of Appendix W -- does evaluation see the representation training saw?

The policy was fitted on tokens read from disk and is judged on tokens made on the fly. Nothing
forces those two to agree. They are produced by different call sites, stored at different
precisions, and reached through different batching, and a policy handed a representation it never
learned would look like a navigation result while actually being an encoding bug.

This encodes frames of an already-encoded training trajectory through the path `evaluate.py`
takes, and compares against what `encode.py` wrote. One run therefore checks the whole chain at
once -- the prompt, the decoding seed, the layer slice, the pooling, the PCA basis, and the
stored dtype.

**Two comparisons, not one.** Every file on disk predates the fix in `vlm_features._assemble`
(Appendix Z.1), so a fresh encoding under today's code differs from the stored one for a reason
that is already known, and a single comparison could not separate "the chain leaks" from "we
fixed the visual slice". So the chain is checked under PR2L_LEGACY_BOS, where agreement is
expected exactly, and the fix is then measured on its own by encoding the same frames both ways.

The expected residual in the first comparison is float16 rounding, since `encode.py` stores
half precision and `evaluate.py` keeps float32 (Appendix Z.2). Half precision carries about
three decimal digits, so the relative error should sit near 1e-3 and the check is written in
relative terms rather than absolute -- the token values are not O(1).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import pca
import vlm_features
from dataset import read_manifest
from encode import HABITAT_ROOT, MANIFEST, frames_of, output_root
from vlm_features import LAYERS, encode_batch, load_vlm


def stored_tokens(record: dict, frame: int) -> np.ndarray:
    """The rows `encode.py` wrote for one frame, found through the trajectory's offsets."""
    payload = np.load(HABITAT_ROOT / record["tokens_path"])
    offsets = payload["offsets"]
    return payload["tokens"][offsets[frame]:offsets[frame + 1]]


def encode_like_evaluate(vlm, basis, frames: np.ndarray, goal: str, episode_id: str,
                         indices: list[int], batch_size: int | None = None
                         ) -> tuple[list[np.ndarray], list[int]]:
    """Reproduce `evaluate.py`'s per-step encoding, including how it names the decoding seed.

    Evaluation batches across episodes and passes one name per frame; the seed is drawn from the
    name and the step number, so passing them per frame here gives the same seed the training
    encoder used for the same frame.

    `batch_size` splits the frames into separate calls. Per-frame seeding is supposed to make the
    result independent of how the work is divided, so varying this must not change anything --
    which is a claim worth testing rather than assuming.

    Returns the reduced tokens and, per frame, how many leading rows are the deterministic block:
    the pooled visual tokens and the question. Those come straight out of the prefill and no
    sampling touches them, so they can be compared even when the generated text diverges.
    """
    step = batch_size or len(indices)
    reduced, deterministic = [], []
    for start in range(0, len(indices), step):
        chunk = indices[start:start + step]
        features = encode_batch(vlm, frames[start:start + step], goal,
                                [episode_id] * len(chunk), chunk, with_cot=True)
        for item in features:
            reduced.append(np.concatenate([basis.apply(item.tokens[:, layer])
                                           for layer in range(len(LAYERS))], axis=1))
            deterministic.append(item.visual_count + item.prompt_count)
    return reduced, deterministic


def compare(name: str, left: np.ndarray, right: np.ndarray) -> dict:
    """Relative disagreement between two encodings of one frame."""
    if left.shape != right.shape:
        return {"frame": name, "shape": f"{left.shape} vs {right.shape}", "relative": float("inf")}
    a = left.astype(np.float64)
    b = right.astype(np.float64)
    scale = max(np.abs(a).max(), np.abs(b).max(), 1e-12)
    absolute = np.abs(a - b).max()
    return {"frame": name, "shape": str(left.shape), "absolute": float(absolute),
            "relative": float(absolute / scale),
            "cosine": float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="cot")
    parser.add_argument("--trajectories", type=int, default=3)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=5e-3,
                        help="relative disagreement allowed under legacy; float16 storage sets it")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = output_root(args.condition)
    encoded = {row["episode_id"]: row for row in read_manifest(root)}
    records = [json.loads(line) for line in MANIFEST.open()]
    # A trajectory needs both its manifest row (for the token file) and its render (for the
    # frames), so pick from the intersection.
    usable = [r for r in records if r["episode_id"] in encoded][:args.trajectories]
    if not usable:
        print("[3-A] 인코딩된 궤적과 렌더가 모두 있는 것이 없다")
        return 1

    basis = pca.load(root / "pca.npz")
    vlm = load_vlm()

    prefix_rows, whole_rows, fix_rows, repeat_rows, split_rows = [], [], [], [], []
    for record in usable:
        row = encoded[record["episode_id"]]
        frames_all = frames_of(record)
        indices = sorted({int(i) for i in
                          np.linspace(0, row["steps"] - 1, args.frames).round()})
        batch = np.asarray(frames_all[indices])
        stored = [stored_tokens(row, i) for i in indices]

        # The switch is read by `_assemble` at call time, so flipping the module attribute is
        # enough; the model stays loaded across both passes.
        vlm_features.LEGACY_BOS = True
        legacy, keep = encode_like_evaluate(vlm, basis, batch, record["object_category"],
                                            record["episode_id"], indices)
        again, _ = encode_like_evaluate(vlm, basis, batch, record["object_category"],
                                        record["episode_id"], indices)
        alone, _ = encode_like_evaluate(vlm, basis, batch, record["object_category"],
                                        record["episode_id"], indices, batch_size=1)
        vlm_features.LEGACY_BOS = False
        fixed, _ = encode_like_evaluate(vlm, basis, batch, record["object_category"],
                                        record["episode_id"], indices)

        for slot, frame in enumerate(indices):
            tag = f"{record['episode_id'][-12:]}#{frame}"
            head = keep[slot]
            prefix_rows.append(compare(tag, stored[slot][:head], legacy[slot][:head]))
            whole_rows.append(compare(tag, stored[slot], legacy[slot]))
            fix_rows.append(compare(tag, legacy[slot][:head], fixed[slot][:head]))
            repeat_rows.append(compare(tag, legacy[slot], again[slot]))
            split_rows.append(compare(tag, legacy[slot], alone[slot]))

    def report(title: str, rows: list[dict], tolerance: float | None) -> bool:
        print("=" * 78)
        print(title)
        print("=" * 78)
        for item in rows:
            print(f"  {item['frame']:22s} {item['shape']:>22s} "
                  f"상대 {item['relative']:.2e}  코사인 {item.get('cosine', float('nan')):.6f}")
        worst = max(item["relative"] for item in rows)
        agreed = sum(np.isfinite(item["relative"]) for item in rows)
        print(f"\n  모양 일치 {agreed}/{len(rows)} | 최대 상대 불일치 {worst:.2e}")
        if tolerance is None:
            print()
            return True
        ok = worst <= tolerance
        print(f"  판정: {'통과' if ok else '실패'} (허용 {tolerance:.0e})\n")
        return ok

    # The deterministic block first. It is the one that carries the claim: the pooled visual
    # tokens and the question come out of the prefill, so if they agree then the image transform,
    # the layer slice, the pooling, the PCA basis and the stored dtype all agree, and whatever
    # else disagrees can only be the sampled text.
    passed = report("3-A(가)  결정적 블록 — 풀링 시각 16 + 질문 18 (표집 무관)",
                    prefix_rows, args.tolerance)
    report("3-A(나)  생성 텍스트를 포함한 전체 열", whole_rows, args.tolerance)
    report("3-A(다)  같은 배치로 두 번 인코딩 (재현성)", repeat_rows, args.tolerance)
    report("3-A(라)  배치 4개 vs 한 장씩 (프레임별 시드가 배치와 무관한가)",
           split_rows, args.tolerance)
    report("Z.1 정량화  수정 전 vs 수정 후, 결정적 블록만", fix_rows, None)
    print(f"  BOS 수정이 표현을 바꾸는 크기: 코사인 최소 "
          f"{min(i.get('cosine', 1.0) for i in fix_rows):.6f}")

    if args.out:
        args.out.write_text(json.dumps({"prefix_vs_stored": prefix_rows,
                                        "whole_vs_stored": whole_rows,
                                        "repeat": repeat_rows, "split": split_rows,
                                        "fixed_vs_legacy": fix_rows}, indent=2) + "\n")
        print(f"\n[3-A] {args.out} 저장")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
