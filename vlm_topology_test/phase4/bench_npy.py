"""Is the zip container costing anything, or is the disk simply slow?

Profiling put 58.6% of the cost of building a batch inside `np.load` on the `.npz` holding a
trajectory's tokens. That number cannot say how much of it is the container and how much is
waiting for bytes: `zipfile` parses a directory and copies each member through Python, but the
same call also does the actual reading. Converting 365 GB to a different format on the strength
of an unsplit measurement is how the last three "obvious" fixes went wrong, so this splits it.

Three ways of getting the same array, timed on the same trajectories:

    npz        what training does now -- np.load through zipfile
    npy        the same bytes as a bare array file
    npy+mmap   mapped and copied, so the kernel moves the pages and Python never loops

**The page cache has to be taken out of it.** A file just written is in RAM, and reading it back
would time memory rather than disk; likewise the second format to be read would inherit whatever
the first left cached. Every file is therefore flushed and dropped with `POSIX_FADV_DONTNEED`
before each read, so all three start cold. Reads with the cache left warm are reported too,
since a fair part of training's reads do hit it.

Run beside training if need be: it competes for the same disk, which slows every condition and
leaves the comparison between them intact.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np

import dataset as D
from train import HABITAT_ROOT, group_by_length


def drop_cache(path: Path) -> None:
    """Evict this file from the page cache so the next read has to reach the disk."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def write_npy(source: Path, target: Path) -> tuple[Path, Path]:
    """The same two arrays the npz holds, as two plain files, flushed to disk."""
    payload = np.load(source)
    tokens, offsets = payload["tokens"], payload["offsets"]
    token_file, offset_file = target.with_suffix(".tokens.npy"), target.with_suffix(".off.npy")
    for array, path in ((tokens, token_file), (offsets, offset_file)):
        np.save(path, array)
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return token_file, offset_file


def timed(fn, *paths, cold: bool) -> tuple[float, int]:
    if cold:
        for path in paths:
            drop_cache(path)
    started = time.perf_counter()
    array = fn()
    total = int(np.asarray(array).nbytes)
    return time.perf_counter() - started, total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", type=Path, default=Path("/scratch/jonghoon/embeddings_cot"))
    parser.add_argument("--work-dir", type=Path, default=Path("/scratch/jonghoon/npy_bench"))
    parser.add_argument("--trajectories", type=int, default=120)
    parser.add_argument("--condition", default="cot")
    args = parser.parse_args()

    embedding_root = args.stage_dir / "embeddings"
    records = D.read_manifest(HABITAT_ROOT / "embeddings" / f"pr2l_habitat_web_hd_{args.condition}")
    rng = np.random.default_rng(0)
    picked = [records[i] for i in rng.choice(len(records), size=args.trajectories, replace=False)]

    args.work_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bench] 궤적 {len(picked)}개 | 작업 디렉터리 {args.work_dir}", flush=True)

    converted, convert_seconds, bytes_written = [], 0.0, 0
    for record in picked:
        source = embedding_root / "train" / f"{record['episode_id']}.npz"
        started = time.perf_counter()
        token_file, offset_file = write_npy(source, args.work_dir / record["episode_id"])
        convert_seconds += time.perf_counter() - started
        bytes_written += token_file.stat().st_size + offset_file.stat().st_size
        converted.append((source, token_file, offset_file))
    print(f"[bench] 변환 {bytes_written / 2**30:.2f} GB, {convert_seconds:.1f}초 "
          f"({bytes_written / 2**20 / convert_seconds:.0f} MB/s) — "
          f"365 GB 환산 {365 * 2**30 / max(bytes_written, 1) * convert_seconds / 60:.0f}분",
          flush=True)

    for cold in (True, False):
        results: dict[str, tuple[float, int]] = {}
        for name in ("npz", "npy", "npy+mmap"):
            seconds, total = 0.0, 0
            for source, token_file, offset_file in converted:
                if name == "npz":
                    took, size = timed(lambda: np.load(source)["tokens"], source, cold=cold)
                elif name == "npy":
                    took, size = timed(lambda: np.load(token_file), token_file, cold=cold)
                else:
                    took, size = timed(
                        lambda: np.array(np.load(token_file, mmap_mode="r")),
                        token_file, cold=cold)
                seconds += took
                total += size
            results[name] = (seconds, total)

        base = results["npz"][0]
        print(f"\n[bench] {'캐시 비움 (콜드)' if cold else '캐시 유지 (웜)'}")
        for name, (seconds, total) in results.items():
            print(f"    {name:10s} {seconds:6.2f}s  {total / 2**20 / seconds:7.0f} MB/s  "
                  f"npz 대비 {base / seconds:5.2f}배")

    shutil.rmtree(args.work_dir, ignore_errors=True)
    print(f"\n[bench] {args.work_dir} 정리 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
