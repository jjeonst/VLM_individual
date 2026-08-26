"""Turn the rendered frames into the representations the policy is trained on.

Three modes, meant to be run in this order.

``check`` proves the extraction is taking vectors from where it believes it is. It encodes one
frame twice under two different target objects. Because the language model is causal and the
image sits before the question, the visual positions cannot know what was asked, so their
vectors must come out bit-for-bit identical while the text positions must differ. If that fails,
the sequence is not laid out as assumed and nothing after this point is trustworthy.

``fit`` finds the 1024 directions that the reduction keeps. Following Appendix C.2, one frame is
drawn from each trajectory, and the directions are computed from the tokens of those frames and
then frozen. It also reports what a per-layer basis would have retained, which costs nothing
extra and settles whether the single shared basis the paper describes is serving both layers.

``encode`` runs every frame and writes the result. Each trajectory becomes one file holding its
tokens end to end plus the offsets that say where each frame's tokens start, because the number
of tokens per frame varies with how long the model's answer ran.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import pca
from vlm_features import (LAYERS, MAX_NEW_TOKENS, MIN_NEW_TOKENS, TEMPERATURE,
                          encode_vision_batch,
                          encode_batch, frame_seed, load_vlm, load_vision_backbone)

HABITAT_ROOT = Path("/data/topovlm/habitat")
MANIFEST = HABITAT_ROOT / "episodes" / "pr2l_habitat_web_hd" / "train" / "manifest.jsonl"
STORE_DTYPE = np.float16          # chosen with the user; the paper does not state a precision


def output_root(condition: str) -> Path:
    return HABITAT_ROOT / "embeddings" / f"pr2l_habitat_web_hd_{condition}"


def read_manifest() -> list[dict]:
    return [json.loads(line) for line in MANIFEST.open()]


def frames_of(record: dict) -> np.ndarray:
    return np.load(HABITAT_ROOT / record["rgb_path"], mmap_mode="r")


def sample_frame_index(record: dict) -> int:
    """The one frame per trajectory the paper draws to estimate the reduction.

    Drawn from a digest of the episode id so that the same trajectory always contributes the
    same frame, however the work is split across jobs.
    """
    return int(frame_seed(record["episode_id"], -1) % max(record["steps"], 1))


def encode_trajectory(vlm, record: dict, batch_size: int, with_cot: bool,
                      indices: list[int] | None = None, vision_only: bool = False):
    """Yield this trajectory's frames' representations, a batch at a time.

    `vision_only` takes the image-encoder route, which never touches the language model.
    """
    frames = frames_of(record)
    wanted = list(range(len(frames))) if indices is None else indices
    for start in range(0, len(wanted), batch_size):
        chunk = wanted[start:start + batch_size]
        batch = np.asarray(frames[chunk])
        if vision_only:
            yield encode_vision_batch(vlm, batch)
        else:
            yield encode_batch(vlm, batch, record["object_category"],
                               record["episode_id"], chunk, with_cot=with_cot)


# --------------------------------------------------------------------------- check


def run_check(vlm, records: list[dict]) -> int:
    """Encode one frame under two goals and confirm only the text positions move."""
    record = records[0]
    frames = np.asarray(frames_of(record)[:1])
    other = "toilet" if record["object_category"] != "toilet" else "chair"

    first = encode_batch(vlm, frames, record["object_category"], record["episode_id"], [0])[0]
    second = encode_batch(vlm, frames, other, record["episode_id"], [0])[0]

    visual = first.visual_count
    same_visual = np.array_equal(first.tokens[:visual], second.tokens[:visual])
    text_a = first.tokens[visual:visual + first.prompt_count]
    text_b = second.tokens[visual:visual + second.prompt_count]
    text_differs = text_a.shape != text_b.shape or not np.array_equal(text_a, text_b)

    print(f"[check] 목표 '{record['object_category']}' vs '{other}'")
    print(f"  시각 토큰 {visual}개가 동일한가: {'OK' if same_visual else 'MISMATCH'}")
    print(f"  질문 토큰이 달라지는가:        {'OK' if text_differs else 'MISMATCH'}")
    print(f"  토큰 수: 시각 {visual} + 질문 {first.prompt_count} + 생성 "
          f"{first.generated_count} = {first.tokens.shape[0]}")
    print(f"  생성된 답 A: {first.generated_text[:160]!r}")
    print(f"  생성된 답 B: {second.generated_text[:160]!r}")
    if not (same_visual and text_differs):
        print("[check] FAILED — 표현을 꺼내는 위치가 가정과 다르다.")
        return 1
    return 0


# ----------------------------------------------------------------------------- fit


def run_fit(vlm, records: list[dict], with_cot: bool, condition: str, limit: int | None) -> int:
    """Estimate the reduction from one frame per trajectory, then freeze it."""
    if limit is not None:
        records = records[:limit]
    pooled = pca.Moments()
    per_layer = [pca.Moments() for _ in LAYERS]

    started = time.time()
    for position, record in enumerate(records, start=1):
        index = sample_frame_index(record)
        features = next(encode_trajectory(vlm, record, 1, with_cot, indices=[index]))[0]
        for layer in range(len(LAYERS)):
            vectors = features.tokens[:, layer]
            pooled.update(vectors)
            per_layer[layer].update(vectors)
        if position % 200 == 0 or position == len(records):
            rate = position / (time.time() - started)
            print(f"[fit] {position}/{len(records)} 궤적, {pooled.count} 토큰, "
                  f"{rate:.2f} 궤적/s", flush=True)

    shared = pca.fit(pooled)
    separate = [pca.fit(m) for m in per_layer]

    print(f"\n=== 기저 적합 결과 (표본 {len(records)} 궤적 / {pooled.count} 토큰) ===")
    print(f"  공유 기저 전체 설명분산: {100 * shared.explained:.2f}%")
    for layer, (name, moments, own) in enumerate(zip(LAYERS, per_layer, separate)):
        norm = float(np.sqrt(np.trace(moments.covariance) / pca.DIM))
        print(f"  층 {name}: 평균 성분 크기 {norm:.4f} | "
              f"공유 기저로 보존 {100 * pca.retained_fraction(shared, moments):5.2f}% | "
              f"층별 기저로 보존 {100 * own.explained:5.2f}%")

    root = output_root(condition)
    pca.save(root / "pca.npz", shared, {
        "condition": condition, "layers": list(LAYERS), "keep": pca.KEEP,
        "trajectories": len(records), "tokens": pooled.count,
        "explained": shared.explained,
        "per_layer_retained_shared": [pca.retained_fraction(shared, m) for m in per_layer],
        "per_layer_retained_own": [b.explained for b in separate],
        "temperature": TEMPERATURE, "min_new_tokens": MIN_NEW_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS, "with_cot": with_cot,
    })
    for layer, basis in zip(LAYERS, separate):
        pca.save(root / f"pca_layer{layer}.npz", basis, {"layer": layer, "reference_only": True})
    print(f"\n[fit] {root / 'pca.npz'} 저장 (층별 기저는 비교용으로만 함께 저장)")
    return 0


# -------------------------------------------------------------------------- encode


def manifest_row(record: dict, target: Path, tokens: int | None = None,
                 steps: int | None = None, width: int | None = None,
                 answers: list[str] | None = None) -> dict:
    """One manifest line. Missing counts are read back from the stored file."""
    if tokens is None or steps is None or width is None:
        payload = np.load(target)
        offsets = payload["offsets"]
        steps = len(offsets) - 1
        tokens = int(offsets[-1])
        width = int(payload["tokens"].shape[1])
    return {"episode_id": record["episode_id"], "object_category": record["object_category"],
            "scene_key": record["scene_key"], "steps": int(steps), "tokens": int(tokens),
            "width": int(width), "tokens_path": str(target.relative_to(HABITAT_ROOT)),
            "sample_answers": answers or []}


def run_encode(vlm, records: list[dict], with_cot: bool, condition: str,
               batch_size: int, shard: int | None, shards: int) -> int:
    """Encode every frame and write one file per trajectory."""
    root = output_root(condition)
    # The image-encoder condition stores the backbone's own 2176-wide patches: the reduction and
    # the two-layer stacking are both PR2L-specific (Appendix C.2, PR2L items 2 and 3).
    vision_only = condition == "image_encoder"
    basis = None if vision_only else pca.load(root / "pca.npz")
    if shard is not None:
        records = [r for i, r in enumerate(records) if i % shards == shard]

    token_dir = root / "train"
    token_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    written, total_frames, total_tokens, reused = [], 0, 0, 0
    started = time.time()
    for position, record in enumerate(records, start=1):
        target = token_dir / f"{record['episode_id']}.npz"
        if target.exists():
            # An earlier run already encoded this one. Its entry still has to go into this
            # run's manifest: training reads the manifest, not the directory, so a trajectory
            # with a file but no entry is invisible. Without this, a job that is resumed after
            # a timeout silently trains on only the part encoded since the restart. Reading the
            # offsets alone is cheap -- npz members load on demand, so the tokens stay on disk.
            written.append(manifest_row(record, target))
            reused += 1
            continue
        pieces, counts, answers = [], [], []
        for batch in encode_trajectory(vlm, record, batch_size, with_cot,
                                       vision_only=vision_only):
            for features in batch:
                if vision_only:
                    reduced = features.tokens
                else:
                    reduced = np.concatenate(
                        [basis.apply(features.tokens[:, layer])
                         for layer in range(len(LAYERS))], axis=1)
                pieces.append(reduced.astype(STORE_DTYPE))
                counts.append(reduced.shape[0])
                if len(answers) < 3:
                    answers.append(features.generated_text)
        tokens = np.concatenate(pieces, axis=0)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        np.savez(target, tokens=tokens, offsets=offsets)

        total_frames += len(counts)
        total_tokens += int(tokens.shape[0])
        written.append(manifest_row(record, target, tokens=int(tokens.shape[0]),
                                    steps=len(counts), width=int(tokens.shape[1]),
                                    answers=answers))
        if position % 5 == 0 or position == len(records):
            elapsed = time.time() - started
            print(f"[encode] {position}/{len(records)} 궤적, {total_frames} 프레임, "
                  f"{total_frames / max(elapsed, 1e-9):.2f} 프레임/s", flush=True)

    name = "all" if shard is None else f"shard{shard:03d}"
    with (manifest_dir / f"{name}.jsonl").open("w") as handle:
        for row in written:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    elapsed = time.time() - started
    rate = total_frames / max(elapsed, 1e-9)
    print(f"\n[encode] 궤적 {len(written)} (이번 실행 {len(written) - reused}, "
          f"기존 재사용 {reused}), 프레임 {total_frames}, 토큰 {total_tokens}")
    print(f"[encode] {rate:.2f} 프레임/s | 프레임당 평균 토큰 "
          f"{total_tokens / max(total_frames, 1):.1f}")
    print(f"[encode] 전체 1,219,318 프레임 환산: GPU 1개로 "
          f"{1219318 / max(rate, 1e-9) / 3600:.1f} 시간")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["check", "fit", "encode"])
    parser.add_argument("--condition", default="cot",
                        choices=["cot", "nocot", "image_encoder"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N trajectories (pilots)")
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--shards", type=int, default=80)
    args = parser.parse_args()

    records = read_manifest()
    with_cot = args.condition == "cot"
    # The image-encoder condition needs the vision half only, which fits on a much
    # smaller card and leaves the 3090s to the language model's work.
    vlm = load_vision_backbone() if args.condition == "image_encoder" else load_vlm()

    if args.mode == "check":
        return run_check(vlm, records)
    if args.mode == "fit":
        return run_fit(vlm, records, with_cot, args.condition, args.limit)
    if args.limit is not None:
        records = records[:args.limit]
    return run_encode(vlm, records, with_cot, args.condition, args.batch_size,
                      args.shard, args.shards)


if __name__ == "__main__":
    raise SystemExit(main())
