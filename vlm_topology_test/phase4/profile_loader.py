"""Where does an epoch's time actually go?

Training spends 2,867 seconds an epoch waiting for data against 480 doing arithmetic, and the
usual suspects are all idle while it waits: two of twelve cores, 130-220 MB/s of a disk that
benchmarks at 650, a request queue under 2.5 with twenty-four workers. Raising the worker count
threefold bought 1.24x; raising the prefetch depth bought nothing and cost 52 GB of shared
memory. Each of those was a guess, and each was wrong, so this stops guessing and measures.

Two questions, one part each.

**Part A -- what does producing one batch cost, stage by stage.** A single process, no
DataLoader, timing the four things `GroupDataset.__getitem__` does: reading the npz through
Python's zipfile, padding each frame out to the trajectory's longest token sequence, collating a
group into one tensor, and handing that tensor to shared memory. If one stage dominates, it is
visible here and the fix is local.

**Part B -- what does the queue between workers and the training loop cost.** The real loader
with the real settings, timing how long the training loop waits for each batch. The shape of
that distribution is the test: work spread evenly across workers gives waits clustered near a
small value, while the in-order delivery this file suspects gives a bimodal split -- most batches
already queued and free, a few costing seconds while everything behind one slow worker stalls.
Part A cannot distinguish those two; only the distribution can.

The run reads the same disk as any training job on the node, so numbers taken beside one are
pessimistic in absolute terms. Both questions are about proportions, which survive that.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import dataset as D
from train import GroupDataset, HABITAT_ROOT, group_by_length


def stage_timings(records: list[dict], embedding_root: Path, habitat_root: Path | None,
                  groups: list[list[dict]]) -> None:
    """Part A: the cost of each step between a file on disk and a tensor in shared memory."""
    load = pad = collate = share = 0.0
    frames = tokens_seen = 0
    raw_bytes = tensor_bytes = 0

    for group in groups:
        built = []
        for record in group:
            token_path, action_path, pose_path = D.trajectory_paths(
                record, embedding_root, habitat_root)

            started = time.perf_counter()
            payload = np.load(token_path)
            token_array, offsets = payload["tokens"], payload["offsets"]
            token_array = np.ascontiguousarray(token_array)   # force the zipfile copy to happen
            actions = np.load(action_path)
            pose = np.load(pose_path)
            load += time.perf_counter() - started
            raw_bytes += token_array.nbytes

            started = time.perf_counter()
            trajectory = D.load_trajectory(record, embedding_root, habitat_root)
            pad += time.perf_counter() - started
            frames += len(trajectory.actions)
            tokens_seen += int(np.diff(offsets).sum())
            built.append(trajectory)

        started = time.perf_counter()
        batch = D.collate(built)
        collate += time.perf_counter() - started
        tensor_bytes += sum(t.element_size() * t.nelement() for t in batch.values())

        started = time.perf_counter()
        for tensor in batch.values():
            tensor.share_memory_()
        share += time.perf_counter() - started

    # `pad` re-reads the files, so the reading it repeats is subtracted out to leave the padding.
    pad = max(pad - load, 0.0)
    total = load + pad + collate + share
    print(f"\n[A] 묶음 {len(groups)}개 / 프레임 {frames:,} / 토큰 {tokens_seen:,}")
    for name, value in (("npz 읽기 (zipfile)", load), ("프레임 채워넣기", pad),
                        ("collate (float32 승격)", collate), ("공유 메모리 전달", share)):
        print(f"    {name:24s} {value:7.2f}s  {100 * value / max(total, 1e-9):5.1f}%")
    print(f"    {'합계':24s} {total:7.2f}s  → 묶음당 {total / len(groups):.3f}s")
    print(f"    원시 토큰 {raw_bytes / 2**30:.2f} GB → 배치 텐서 {tensor_bytes / 2**30:.2f} GB "
          f"({tensor_bytes / max(raw_bytes, 1):.2f}배)")


def queue_timings(groups: list[list[dict]], embedding_root: Path, habitat_root: Path | None,
                  workers: int, prefetch: int, batches: int) -> None:
    """Part B: how long the training loop waits for each batch, and how those waits are spread."""
    loader = DataLoader(GroupDataset(groups, embedding_root, habitat_root),
                        batch_size=None, shuffle=True, num_workers=workers,
                        pin_memory=torch.cuda.is_available(),
                        persistent_workers=workers > 0,
                        prefetch_factor=prefetch if workers > 0 else None)

    waits, started = [], time.perf_counter()
    iterator = iter(loader)
    for _ in range(batches):
        mark = time.perf_counter()
        try:
            next(iterator)
        except StopIteration:
            break
        waits.append(time.perf_counter() - mark)
    elapsed = time.perf_counter() - started

    values = np.array(waits)
    print(f"\n[B] 워커 {workers} prefetch {prefetch} | 배치 {len(values)}개 {elapsed:.1f}s")
    print(f"    배치당 대기  평균 {values.mean():.3f}s  중앙 {np.median(values):.3f}s  "
          f"최대 {values.max():.3f}s")
    for q in (10, 50, 90, 99):
        print(f"      {q:2d}분위 {np.percentile(values, q):.3f}s")
    # The signature of in-order delivery: a few batches carrying most of the wall clock while
    # the rest arrive free, because they were queued behind the slow one and finished long ago.
    top = np.sort(values)[-max(1, len(values) // 10):]
    print(f"    상위 10% 배치가 전체 대기의 {100 * top.sum() / values.sum():.0f}% 를 차지 "
          f"(고르게 퍼져 있으면 10%)")
    print(f"    대기 0.05s 미만 배치 {100 * (values < 0.05).mean():.0f}%")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="cot")
    parser.add_argument("--stage-dir", type=Path, default=Path("/scratch/jonghoon/embeddings_cot"))
    parser.add_argument("--groups", type=int, default=24, help="groups timed in part A")
    parser.add_argument("--batches", type=int, default=60, help="batches drawn in part B")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--prefetch", type=int, default=2)
    # `train.py`'s own default. An earlier run of this file used 16,384 -- the transitions per
    # optimiser step, not per forward pass -- and produced 78 groups of a hundred trajectories
    # instead of the 3,561 groups of about two that training actually builds. The stage costs
    # measured on those would have described a batch a hundred times too large.
    parser.add_argument("--frame-budget", type=int, default=384)
    args = parser.parse_args()

    embedding_root = HABITAT_ROOT / "embeddings" / f"pr2l_habitat_web_hd_{args.condition}"
    records = D.read_manifest(embedding_root)
    habitat_root = None
    if args.stage_dir is not None and (args.stage_dir / "embeddings").exists():
        # `stage_locally` puts the token files one level down and leaves actions and pose at the
        # top, so the two roots are not the same directory.
        embedding_root, habitat_root = args.stage_dir / "embeddings", args.stage_dir
        print(f"[profile] 로컬 사본 사용: {args.stage_dir}")

    groups = group_by_length(records, args.frame_budget)
    print(f"[profile] 궤적 {len(records):,} | 묶음 {len(groups):,} "
          f"(묶음당 평균 {len(records) / len(groups):.1f}궤적)")

    rng = np.random.default_rng(0)
    picked = [groups[i] for i in rng.choice(len(groups), size=min(args.groups, len(groups)),
                                            replace=False)]
    stage_timings(records, embedding_root, habitat_root, picked)
    queue_timings(groups, embedding_root, habitat_root, args.workers, args.prefetch, args.batches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
