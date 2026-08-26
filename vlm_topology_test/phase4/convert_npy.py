"""Rewrite the staged token files as plain arrays, because the zip container is expensive.

Each trajectory's tokens live in an uncompressed `.npz`, which means `np.load` goes through
Python's `zipfile`: a directory parse and a chunked copy per member. Measured on this node with
the page cache dropped between reads, that path sustains 42 MB/s against 120 MB/s for the same
bytes in a bare `.npy` -- 2.83x, and 4.63x when the pages are already resident. Memory-mapping
the file instead was no better than the zip (42 MB/s cold): its page faults are as small as
zipfile's chunks, so it is not the container's structure that costs, it is the size of the reads.

Both files are written per trajectory, tokens and offsets, beside the original. **The `.npz` is
left in place.** It is the only copy of the staged data, `/scratch` has terabytes free, and
keeping it means a mistake here costs nothing but the time to notice.

Verification is not optional and not a sample: every converted file is read back and compared
against its source, shape, dtype and contents. Silently corrupting one trajectory in 7,824 would
be invisible in a loss curve and expensive to find later.
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def targets(source: Path) -> tuple[Path, Path]:
    """The two files one `.npz` becomes.

    Built by trimming the extension off the name rather than with `with_suffix`, because
    `payload_id` permits dots in an episode id and `with_suffix` would eat everything after the
    last one.
    """
    base = source.name[: -len(source.suffix)] if source.suffix else source.name
    return source.with_name(base + ".tokens.npy"), source.with_name(base + ".off.npy")


def convert(source: Path, verify: bool) -> tuple[int, bool]:
    """Write one trajectory's two arrays, then read them back and compare."""
    token_file, offset_file = targets(source)
    if token_file.exists() and offset_file.exists():
        return 0, True                      # already done by an earlier, interrupted run

    payload = np.load(source)
    tokens, offsets = payload["tokens"], payload["offsets"]

    # Write beside the target and rename, so an interrupted run never leaves a short file that
    # a later run would mistake for finished.
    for array, path in ((tokens, token_file), (offsets, offset_file)):
        scratch = path.with_name(path.name + ".partial.npy")
        np.save(scratch, array)          # the name already ends in .npy, so nothing is appended
        os.replace(scratch, path)

    written = token_file.stat().st_size + offset_file.stat().st_size
    if not verify:
        return written, True
    back_tokens = np.load(token_file)
    back_offsets = np.load(offset_file)
    ok = (back_tokens.shape == tokens.shape and back_tokens.dtype == tokens.dtype
          and np.array_equal(back_tokens, tokens) and np.array_equal(back_offsets, offsets))
    return written, ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("/scratch/jonghoon/embeddings_cot/embeddings/train"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    sources = sorted(args.root.glob("*.npz"))
    if not sources:
        print(f"[convert] {args.root} 에 npz 가 없다")
        return 1
    total_source = sum(path.stat().st_size for path in sources)
    print(f"[convert] {len(sources):,}개 / {total_source / 2**30:.1f} GB → .npy "
          f"(워커 {args.workers}, 검증 {'끔' if args.no_verify else '켬'})", flush=True)

    written, failures, done, started = 0, [], 0, time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for source, (size, ok) in zip(sources, pool.map(
                lambda p: convert(p, not args.no_verify), sources)):
            written += size
            done += 1
            if not ok:
                failures.append(source.name)
            if done % 500 == 0 or done == len(sources):
                elapsed = time.time() - started
                rate = written / 2**20 / max(elapsed, 1e-9)
                remaining = (len(sources) - done) / max(done / max(elapsed, 1e-9), 1e-9)
                print(f"[convert] {done:,}/{len(sources):,} | {written / 2**30:.1f} GB | "
                      f"{rate:.0f} MB/s | 남은 시간 {remaining / 60:.0f}분", flush=True)

    elapsed = time.time() - started
    print(f"\n[convert] 완료 {done:,}개 / {written / 2**30:.1f} GB / {elapsed / 60:.1f}분")
    if failures:
        print(f"[convert] 실패 — 읽어들인 내용이 원본과 다르다: {len(failures)}개")
        for name in failures[:10]:
            print(f"    {name}")
        return 1

    made = len(list(args.root.glob("*.tokens.npy")))
    if made != len(sources):
        print(f"[convert] 실패 — npz {len(sources):,}개인데 tokens.npy 는 {made:,}개다")
        return 1
    print(f"[convert] 검증 통과: npz {len(sources):,} = tokens.npy {made:,}, 내용 일치")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
