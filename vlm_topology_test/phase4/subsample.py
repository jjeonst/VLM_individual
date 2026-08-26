"""Fetch the Habitat-Web human demonstrations and take the paper's subsample.

The paper trains on human demonstrations rather than on paths produced by a planner. That
choice is the reason the demonstrations contain search behaviour at all: a planner that
already knows where the goal is walks straight to it, so nothing in its actions can teach a
policy what to do when the goal is not yet visible. The demonstrations used here were
collected from people through Habitat-Web and are distributed by the PIRLNav authors as
``objectnav_hm3d_hd``.

Each demonstration is stored as a **list of actions** together with the pose the episode
started from. No images are stored. Habitat's dynamics are deterministic, so replaying the
actions from that starting pose reproduces exactly what the person saw; that replay is the
next step of this study and not this module's concern.

The full set is far larger than this study can encode with a vision-language model, and the
paper faced the same limit. Its rule is quoted here in full because this module implements
it literally:

    "we used a subset of the dataset, built by dividing the dataset by both target object
    and scene, then sampling every tenth demo. This would ensure that our training data
    still contained examples from every training scene + target object combination that
    existed."
    -- Appendix C.2, item 1

Two things follow from that sentence. First, the grouping is by **pair** of scene and target
object, not by either alone. Second, "every tenth" is a fixed stride, not a random draw, so
the selection carries no seed and repeats identically.

The reason for grouping before striding is stated in the paper's own next sentence: a plain
one-in-ten draw over the whole set would leave some scene/object combinations with no
examples at all, and the policy would then be asked at evaluation time to find an object in
a building where it never saw that object during training. Grouping first makes that
impossible. This module therefore treats "no combination is missing" as a hard check rather
than as a statistic to report.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

HF_REPO = "axel81/pirlnav"
HF_SUBDIR = "objectnav_hm3d_hd/train/content"

SOURCE_DIR = Path("/data/topovlm/habitat/sources/pirlnav_hf/objectnav_hm3d_hd/train/content")
SELECTION_DIR = Path("/data/topovlm/habitat/episode_selections/pr2l_habitat_web_hd")
SELECTION_FILE = SELECTION_DIR / "train_every_tenth.jsonl"

STRIDE = 10

# What the paper and its cited sources report for the full dataset, used to confirm that the
# downloaded files are the ones the paper used rather than one of the other releases.
PAPER_TRAJECTORIES = 77_000
PAPER_STEPS = 12_000_000
PAPER_MEAN_STEPS = 159
PAPER_SCENES = 80
PAPER_OBJECTS = {"chair", "bed", "plant", "toilet", "tv_monitor", "sofa"}

# What the paper reports after subsampling.
PAPER_SUBSET_TRAJECTORIES = 7_550
PAPER_SUBSET_STEPS = 1_100_000


def download(force: bool = False) -> list[Path]:
    """Fetch the per-scene demonstration files, skipping any already present."""
    from huggingface_hub import list_repo_files, hf_hub_download

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    remote = [f for f in list_repo_files(HF_REPO, repo_type="dataset")
              if f.startswith(HF_SUBDIR) and f.endswith(".json.gz")]
    print(f"[download] {len(remote)} scene files in {HF_REPO}/{HF_SUBDIR}", flush=True)

    paths = []
    fetched = 0
    for name in sorted(remote):
        target = SOURCE_DIR / Path(name).name
        if target.exists() and not force:
            paths.append(target)
            continue
        local = hf_hub_download(HF_REPO, name, repo_type="dataset")
        target.write_bytes(Path(local).read_bytes())
        paths.append(target)
        fetched += 1
    print(f"[download] {fetched} newly fetched, {len(paths) - fetched} already present",
          flush=True)
    return paths


def scene_key(scene_id: str) -> str:
    """Reduce a scene path to the identifier shared by the demonstrations and the meshes."""
    # e.g. "hm3d/train/00744-1S7LAXRdDqK/1S7LAXRdDqK.basis.glb" -> "00744-1S7LAXRdDqK"
    parts = scene_id.split("/")
    return parts[-2] if len(parts) >= 2 else scene_id


def read_shard(path: Path) -> list[dict]:
    """Read one scene's demonstrations, keeping only what later steps need."""
    payload = json.loads(gzip.open(path, "rt").read())
    records = []
    for episode in payload["episodes"]:
        replay = episode.get("reference_replay") or []
        records.append({
            "episode_id": episode["episode_id"],
            "scene_id": episode["scene_id"],
            "scene_key": scene_key(episode["scene_id"]),
            "object_category": episode["object_category"],
            "shard_path": str(path),
            "replay_length": len(replay),
        })
    return records


def load_inventory(paths: list[Path]) -> list[dict]:
    """Read every scene file, in a fixed order so the selection is reproducible."""
    inventory = []
    for index, path in enumerate(sorted(paths), start=1):
        inventory.extend(read_shard(path))
        if index % 20 == 0 or index == len(paths):
            print(f"[inventory] {index}/{len(paths)} scenes read, "
                  f"{len(inventory)} demonstrations so far", flush=True)
    return inventory


def select_every_tenth(inventory: list[dict]) -> list[dict]:
    """Group by (scene, target object) and keep every tenth demonstration in each group.

    Order within a group is the order the demonstrations appear in their scene file, which is
    fixed on disk, so no seed is involved and repeated runs agree.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in inventory:
        groups[(record["scene_key"], record["object_category"])].append(record)

    selected = []
    for key in sorted(groups):
        selected.extend(groups[key][::STRIDE])
    return selected


def summarise(records: list[dict]) -> dict:
    """Counts used by both checks below."""
    lengths = [r["replay_length"] for r in records]
    return {
        "trajectories": len(records),
        "steps": sum(lengths),
        "mean_steps": sum(lengths) / max(len(lengths), 1),
        "scenes": {r["scene_key"] for r in records},
        "objects": {r["object_category"] for r in records},
        "combinations": {(r["scene_key"], r["object_category"]) for r in records},
        "empty": sum(1 for n in lengths if n == 0),
    }


def within(value: float, target: float, tolerance: float) -> bool:
    return abs(value - target) <= target * tolerance


def check_full_dataset(stats: dict) -> list[str]:
    """Confirm the downloaded files are the release the paper used."""
    problems = []
    print("\n=== 1차 검증: 내려받은 원본이 논문이 쓴 데이터인가 ===")
    rows = [
        ("총 궤적", stats["trajectories"], PAPER_TRAJECTORIES,
         within(stats["trajectories"], PAPER_TRAJECTORIES, 0.20)),
        ("총 스텝", stats["steps"], PAPER_STEPS,
         within(stats["steps"], PAPER_STEPS, 0.20)),
        ("평균 길이", round(stats["mean_steps"], 1), PAPER_MEAN_STEPS,
         within(stats["mean_steps"], PAPER_MEAN_STEPS, 0.20)),
        ("장면 수", len(stats["scenes"]), PAPER_SCENES,
         len(stats["scenes"]) == PAPER_SCENES),
    ]
    for label, observed, expected, ok in rows:
        print(f"  {'OK ' if ok else 'MISMATCH'}  {label}: {observed:,} (논문 {expected:,})"
              if isinstance(observed, int)
              else f"  {'OK ' if ok else 'MISMATCH'}  {label}: {observed} (논문 {expected})")
        if not ok:
            problems.append(label)

    missing = PAPER_OBJECTS - stats["objects"]
    extra = stats["objects"] - PAPER_OBJECTS
    ok = not missing and not extra
    print(f"  {'OK ' if ok else 'MISMATCH'}  목표 물체: {len(stats['objects'])}종"
          f"{'' if ok else f' (누락 {sorted(missing)}, 초과 {sorted(extra)})'}")
    if not ok:
        problems.append("목표 물체")
    return problems


def check_subset(stats: dict, full: dict) -> list[str]:
    """Confirm the subsample matches what the paper reports and covers every combination."""
    problems = []
    print("\n=== 2차 검증: 부표본이 논문과 같은가 ===")
    rows = [
        ("선정 궤적", stats["trajectories"], PAPER_SUBSET_TRAJECTORIES,
         within(stats["trajectories"], PAPER_SUBSET_TRAJECTORIES, 0.20)),
        ("선정 스텝", stats["steps"], PAPER_SUBSET_STEPS,
         within(stats["steps"], PAPER_SUBSET_STEPS, 0.20)),
        ("장면 수", len(stats["scenes"]), PAPER_SCENES,
         len(stats["scenes"]) == PAPER_SCENES),
        ("목표 물체", len(stats["objects"]), len(PAPER_OBJECTS),
         stats["objects"] == PAPER_OBJECTS),
    ]
    for label, observed, expected, ok in rows:
        print(f"  {'OK ' if ok else 'MISMATCH'}  {label}: {observed:,} (논문 {expected:,})")
        if not ok:
            problems.append(label)

    lost = full["combinations"] - stats["combinations"]
    print(f"  {'OK ' if not lost else 'MISMATCH'}  (장면 x 물체) 조합: "
          f"{len(stats['combinations'])}/{len(full['combinations'])} 유지, 누락 {len(lost)}")
    if lost:
        print(f"      누락 예시: {sorted(lost)[:5]}")
        problems.append("조합 누락")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="run every check but do not write the selection file")
    args = parser.parse_args()

    if args.skip_download:
        paths = sorted(SOURCE_DIR.glob("*.json.gz"))
        if not paths:
            print(f"[error] no shards under {SOURCE_DIR}", file=sys.stderr)
            return 1
    else:
        paths = download(force=args.force_download)

    inventory = load_inventory(paths)
    full = summarise(inventory)
    problems = check_full_dataset(full)

    selected = select_every_tenth(inventory)
    subset = summarise(selected)
    problems += check_subset(subset, full)

    print("\n=== 선정 결과 요약 ===")
    print(f"  궤적 {subset['trajectories']:,} / 스텝 {subset['steps']:,} "
          f"/ 평균 {subset['mean_steps']:.1f} 스텝")
    print(f"  길이 0인 궤적: 원본 {full['empty']}개, 선정분 {subset['empty']}개")
    per_object = defaultdict(int)
    for record in selected:
        per_object[record["object_category"]] += 1
    print("  목표별: " + ", ".join(f"{k} {v}" for k, v in sorted(per_object.items())))
    print(f"  예상 RGB 용량: {subset['steps'] * 900 / 1024 / 1024:.0f} GB (프레임당 900 KB)")

    if problems:
        print(f"\n[STOP] 논문과 어긋나는 항목: {problems}", file=sys.stderr)
        print("       선정 파일을 쓰지 않고 멈춥니다.", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\n[dry-run] 검증만 하고 파일은 쓰지 않았습니다.")
        return 0

    SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    with SELECTION_FILE.open("w") as handle:
        for record in selected:
            handle.write(json.dumps(record) + "\n")
    print(f"\n[write] {SELECTION_FILE} ({len(selected):,} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
