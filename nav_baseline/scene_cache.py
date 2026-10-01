"""Serve scene meshes from the compute node's local disk instead of the shared filesystem.

The 3D scenes live on a network filesystem that every compute node reads over the network.
Measured on this cluster, reading from it runs at about 0.12 GB/s while the node's own disk
reaches roughly 2.5 GB/s, so any file the simulator opens repeatedly is worth copying to the
node first.

This module mirrors the files a scene needs into a directory on the local disk and returns
paths into that mirror. The mirror keeps the original directory layout because the scene
dataset configuration refers to scenes by relative path. Copies are made once per node and
reused by later jobs, and a partially written mirror is never handed out: each scene is
copied into a temporary directory and renamed into place only when complete.

Whether this actually saves time depends on where the loading cost sits. If the simulator
spends its time parsing and decompressing the mesh rather than reading bytes, a local copy
changes little. ``benchmark`` measures both paths on the current node so the question can be
answered with numbers rather than assumed.
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

SOURCE_ROOT = Path("/data/topovlm/habitat/scene_datasets/hm3d_v0.2")
CONFIG_NAME = "hm3d_annotated_basis.scene_dataset_config.json"

# Files a scene needs. The semantic mesh is large and only required when a semantic sensor
# is used, which these experiments do not do, so it is copied only when asked for.
# The .scn file describes the semantic scene. It is small, and copying it keeps the
# simulator from logging a load failure for every scene even though nothing needs it.
CORE_SUFFIXES = (".basis.glb", ".basis.navmesh", ".basis.scn")
SEMANTIC_SUFFIXES = (".semantic.glb", ".semantic.txt")


def local_root() -> Path:
    """Directory on the node's own disk that holds the mirror."""
    base = os.environ.get("SCENE_CACHE_ROOT") or f"/scratch/{os.environ.get('USER', 'shared')}"
    return Path(base) / "hm3d_v0.2"


def cache_available() -> bool:
    """Whether a local disk is present and writable on this node."""
    try:
        local_root().mkdir(parents=True, exist_ok=True)
        probe = local_root() / ".writable"
        probe.write_text("ok")
        probe.unlink()
        return True
    except Exception:
        return False


def ensure_config(root: Path) -> Path:
    """Copy the scene dataset configuration into the mirror if it is not already there."""
    target = root / CONFIG_NAME
    if not target.exists():
        root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_ROOT / CONFIG_NAME, target)
    return target


def ensure_scene(scene_key: str, split: str = "train", *, with_semantic: bool = False) -> Path:
    """Mirror one scene onto the local disk and return the directory holding it.

    ``scene_key`` is a directory name such as ``00006-HkseAnWCgqk``. The scene's own files
    are named after the part following the first hyphen.
    """
    source = SOURCE_ROOT / split / scene_key
    root = local_root()
    target = root / split / scene_key
    marker = target / ".complete"
    if marker.exists():
        return target

    suffixes = CORE_SUFFIXES + (SEMANTIC_SUFFIXES if with_semantic else ())
    name = scene_key.split("-", 1)[-1]
    staging = target.with_name(target.name + f".partial.{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    for suffix in suffixes:
        candidate = source / f"{name}{suffix}"
        if candidate.exists():
            shutil.copy2(candidate, staging / candidate.name)
    (staging / ".complete").write_text("ok")

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging.rename(target)
    except OSError:
        # Another process finished the same scene first; keep theirs and drop ours.
        shutil.rmtree(staging, ignore_errors=True)
    return target


def scene_paths(scene_key: str, split: str = "train", *, use_cache: bool = True,
                with_semantic: bool = False) -> tuple[str, str]:
    """Return (scene mesh path, scene dataset configuration path) for the simulator.

    Falls back to the shared filesystem when no local disk is available, so callers do not
    need to branch on the environment.
    """
    name = scene_key.split("-", 1)[-1]
    if use_cache and cache_available():
        root = local_root()
        config = ensure_config(root)
        directory = ensure_scene(scene_key, split, with_semantic=with_semantic)
        mesh = directory / f"{name}.basis.glb"
        if mesh.exists():
            return str(mesh), str(config)
    return (str(SOURCE_ROOT / split / scene_key / f"{name}.basis.glb"),
            str(SOURCE_ROOT / CONFIG_NAME))


def benchmark(scene_key: str, split: str = "train") -> dict:
    """Time one scene load from the shared filesystem and from the local mirror."""
    from nav_baseline.env import make_sim

    remote_mesh, remote_config = scene_paths(scene_key, split, use_cache=False)
    started = time.time()
    sim = make_sim(remote_mesh, scene_dataset_config=remote_config)
    remote_seconds = time.time() - started
    sim.close()

    started = time.time()
    local_mesh, local_config = scene_paths(scene_key, split, use_cache=True)
    copy_seconds = time.time() - started

    started = time.time()
    sim = make_sim(local_mesh, scene_dataset_config=local_config)
    local_seconds = time.time() - started
    sim.close()

    return {"scene": scene_key,
            "shared_filesystem_seconds": round(remote_seconds, 2),
            "copy_seconds": round(copy_seconds, 2),
            "local_disk_seconds": round(local_seconds, 2),
            "speedup": round(remote_seconds / max(local_seconds, 1e-6), 2)}


def main():
    import argparse
    import json
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from nav_baseline.env import scene_keys_with_episodes

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=3)
    parser.add_argument("--prefetch", action="store_true",
                        help="only copy the scenes onto the local disk, without timing")
    args = parser.parse_args()

    keys = scene_keys_with_episodes()[: args.scenes]
    print(f"local cache root: {local_root()} (available: {cache_available()})", flush=True)

    if args.prefetch:
        started = time.time()
        for key in keys:
            ensure_scene(key)
            print(f"  cached {key}", flush=True)
        print(f"prefetched {len(keys)} scenes in {time.time() - started:.1f}s")
        return

    results = [benchmark(key) for key in keys]
    for row in results:
        print(json.dumps(row), flush=True)
    shared = sum(r["shared_filesystem_seconds"] for r in results) / len(results)
    local = sum(r["local_disk_seconds"] for r in results) / len(results)
    copy = sum(r["copy_seconds"] for r in results) / len(results)
    print(f"\naverage over {len(results)} scenes:")
    print(f"  shared filesystem {shared:.1f}s   local disk {local:.1f}s   "
          f"copy {copy:.1f}s   speedup {shared / max(local, 1e-6):.2f}x")
    print(f"  first use costs {copy + local:.1f}s, every later use costs {local:.1f}s")


if __name__ == "__main__":
    main()
