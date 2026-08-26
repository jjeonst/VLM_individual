"""Encode the expert rollouts three ways, for the representation figure (Appendix H.2).

This is the figure's second stage. `rollout.py` produced thirty shortest-path-follower
trajectories and recorded, at every step, the frame the agent saw and how far it still had to
walk. This module turns those frames into the three representations the figure puts side by
side, and it is deliberately separate from `encode.py`: that one encodes 1.2 million frames of
human demonstrations into what the policy trains on, and everything about it -- sharding, the
learned PCA basis, keeping only three sample answers per trajectory -- is shaped by that scale.
Fifteen hundred frames need none of it, and one thing here is the opposite of what `encode.py`
wants.

**Every frame's answer text is kept.** `encode.py` stores three per trajectory, because the text
is a debugging aid there and the representation is the product. Here the text *is* a product:
the figure's boxes group points by the room the VLM named, so a frame whose answer was dropped
is a point that cannot be placed in a box.

**What is stored is the token-wise mean.** Appendix H.1 fixes this -- "we then perform principal
component analysis (PCA) on the tokenwise average of all embeddings for each observation" -- and
H.2 opens by saying the Habitat analysis is the same one. Storing the mean rather than every
token costs nothing in fidelity for this figure: the reduction to 1024 dimensions is a linear
map, so applying it to the mean and averaging after applying it give the same answer, and the
choice of whether to apply it at all can be left to the plotting stage. It also turns 2 GB into
25 MB.

**Three passes, because the prompt is what is being studied.**

    cot     "Would a {goal} be found here? Why or why not?"   the main text's condition
    room    "What room is this?"                              Appendix D's, and the box labels
    image   no prompt at all                                  the vision backbone alone

The room pass does double duty. It supplies the upper row of the figure that compares against
Figure 8 directly, and its generated text supplies the room labels for *both* figures' boxes --
including the boxes drawn over the image-encoder row, which has no text of its own. That is what
makes the lower row a comparison rather than a separate picture: the same frames, carrying the
same labels, laid out by a representation that did not go through the language model.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from vlm_features import (encode_batch, encode_vision_batch, load_vision_backbone, load_vlm)

HABITAT_ROOT = Path("/data/topovlm/habitat")
ROLLOUT_ROOT = HABITAT_ROOT / "figure_rollouts"
OUT_ROOT = HABITAT_ROOT / "figure_embeddings"

# Appendix D.3: "we ... choose 'What room is this?' as our task-relevant prompt". It takes no
# goal, which is the point -- the same question is asked of every frame in every column, so a
# difference between columns is a difference in what the rooms look like, not in what was asked.
PROMPT_ROOM = "What room is this?"

STORE_DTYPE = np.float16          # as in `encode.py`; the paper does not state a precision


def rollouts() -> list[dict]:
    rows = [json.loads(line) for line in (ROLLOUT_ROOT / "manifest.jsonl").open()]
    names = [row["name"] for row in rows]
    if len(set(names)) != len(names):
        raise RuntimeError(f"롤아웃 매니페스트에 중복 이름이 있다: {len(names)}행 "
                           f"고유 {len(set(names))}")
    return sorted(rows, key=lambda row: row["name"])


def encode_one(vlm, record: dict, condition: str, batch_size: int) -> dict:
    """One trajectory's frames, as a mean vector per frame plus the text that produced it."""
    payload = np.load(ROLLOUT_ROOT / f"{record['name']}.npz", mmap_mode="r")
    frames = payload["frames"]
    means, answers = [], []

    for start in range(0, len(frames), batch_size):
        chunk = list(range(start, min(start + batch_size, len(frames))))
        batch = np.asarray(frames[chunk])
        if condition == "image":
            features = encode_vision_batch(vlm, batch)
        else:
            features = encode_batch(
                vlm, batch, record["object_category"], record["name"], chunk,
                template=PROMPT_ROOM if condition == "room" else None)
        for item in features:
            # [N, layers, 4096] for the prompted conditions, [16, 2176] for the backbone.
            means.append(item.tokens.mean(axis=0))
            answers.append(getattr(item, "generated_text", ""))

    return {"mean": np.stack(means).astype(STORE_DTYPE), "answers": answers,
            "distances": np.asarray(payload["distances"], dtype=np.float32),
            "poses": np.asarray(payload["poses"], dtype=np.float32)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=["cot", "room", "image"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    records = rollouts()
    frames_total = sum(row["steps"] for row in records)
    print(f"[fig-encode] 조건 {args.condition} | 궤적 {len(records)} | 프레임 {frames_total:,}",
          flush=True)

    vlm = load_vision_backbone() if args.condition == "image" else load_vlm()
    print(f"[fig-encode] 모델 적재 완료 ({torch.cuda.get_device_name(0)})", flush=True)

    out = OUT_ROOT / args.condition
    out.mkdir(parents=True, exist_ok=True)

    # Clear whatever a previous rollout set left here. These files are named after the
    # trajectories they encode, so a regenerated set writes new names beside the old ones rather
    # than over them, and the directory quietly becomes a mixture of two rollouts. That is
    # exactly how a superseded shard manifest ended up upweighting 261 training trajectories
    # earlier today; the same accumulation here would put frames from retired trajectories into
    # the figure.
    stale = sorted(out.glob("*.npz")) + sorted(out.glob("*.answers.json")) \
        + sorted(out.glob("manifest.jsonl"))
    for path in stale:
        path.unlink()
    if stale:
        print(f"[fig-encode] 이전 산출물 {len(stale)}개 삭제 (전량 재생성한다)", flush=True)

    written, done, started = [], 0, time.time()

    for position, record in enumerate(records, start=1):
        result = encode_one(vlm, record, args.condition, args.batch_size)
        if len(result["mean"]) != record["steps"]:
            print(f"[fig-encode] 실패 — {record['name']}: 프레임 {record['steps']}인데 "
                  f"표현 {len(result['mean'])}개")
            return 1
        np.savez(out / f"{record['name']}.npz", mean=result["mean"],
                 distances=result["distances"], poses=result["poses"])
        (out / f"{record['name']}.answers.json").write_text(
            json.dumps(result["answers"], ensure_ascii=False, indent=1))
        written.append({"name": record["name"],
                        "object_category": record["object_category"],
                        "scene_id": record["scene_id"], "steps": record["steps"],
                        "width": [int(v) for v in result["mean"].shape[1:]]})
        done += record["steps"]
        elapsed = time.time() - started
        print(f"[fig-encode] {position}/{len(records)} {record['name']}: "
              f"{record['steps']}프레임, 누적 {done}/{frames_total}, "
              f"{done / max(elapsed, 1e-9):.2f} 프레임/s", flush=True)

    (out / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in written))

    files = {path.stem for path in out.glob("*.npz")}
    names = {row["name"] for row in written}
    if files != names:
        print(f"[fig-encode] 실패 — 매니페스트 {len(names)}개와 파일 {len(files)}개가 어긋난다")
        return 1

    widths = {tuple(row["width"]) for row in written}
    print(f"\n[fig-encode] 궤적 {len(written)} / 프레임 {done:,} | 표현 폭 {widths} | "
          f"{time.time() - started:.0f}초")
    if args.condition != "image":
        empty = 0
        for row in written:
            answers = json.loads((out / f"{row['name']}.answers.json").read_text())
            empty += sum(1 for text in answers if not text.strip())
        print(f"[fig-encode] 빈 답변 {empty}/{done} — 논문이 말한 '라벨 없음' 군집의 하한이다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
