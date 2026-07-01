"""Real-dataset diagnostics for predictive branching hypotheses.

The runner intentionally uses only dataset-native trajectories/states:
- R2R/VLN-CE GT 3D path locations for navigation.
- Minari/D4RL Franka Kitchen achieved-goal states for robotics.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("TOPOVLM_DATASET_ROOT", "/data/topovlm"))
R2R_DATASET_ROOT = (
    DATA_ROOT
    / "r2r_vlnce_v1_3_preprocessed"
    / "extracted"
    / "R2R_VLNCE_v1-3_preprocessed"
)
KITCHEN_DATASET_ROOT = DATA_ROOT / "d4rl_kitchen_compositional_v2" / "hf_snapshot" / "kitchen"
RESULT_ROOT = REPO_ROOT / "analysis" / "results" / "predictive_branching_datasets"
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "analysis" / "predictive_branching_datasets"


@dataclass(frozen=True)
class R2RRecord:
    episode_id: str
    trajectory_id: int
    scene_id: str
    instruction_text: str
    positions: np.ndarray
    start_cell: tuple[int, int]
    goal_cell: tuple[int, int]


@dataclass(frozen=True)
class KitchenEpisode:
    dataset_name: str
    episode_name: str
    objects: tuple[str, ...]
    progress: np.ndarray
    order: tuple[str, ...]


def _safe_json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _copy_outputs_to_artifact_lane(result_root: Path, artifact_root: Path) -> None:
    artifact_root.mkdir(parents=True, exist_ok=True)
    for source in sorted(result_root.rglob("*")):
        if not source.is_file():
            continue
        target = artifact_root / source.relative_to(result_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _load_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _scene_key(scene_id: str) -> str:
    return Path(scene_id).stem


def _cell_from_position(position: np.ndarray, cell_size: float) -> tuple[int, int]:
    return (int(math.floor(float(position[0]) / cell_size)), int(math.floor(float(position[2]) / cell_size)))


def _direction_bin(delta: np.ndarray, bins: int = 8) -> int | None:
    norm = float(np.linalg.norm(delta[[0, 2]]))
    if norm < 1e-6:
        return None
    angle = math.atan2(float(delta[2]), float(delta[0]))
    wrapped = (angle + 2.0 * math.pi) % (2.0 * math.pi)
    return int(math.floor(wrapped / (2.0 * math.pi / bins))) % bins


def _load_r2r_records(split: str, cell_size: float = 0.5) -> list[R2RRecord]:
    split_dir = R2R_DATASET_ROOT / split
    episodes_payload = _load_json_gz(split_dir / f"{split}.json.gz")
    gt_payload = _load_json_gz(split_dir / f"{split}_gt.json.gz")
    records: list[R2RRecord] = []
    for episode in episodes_payload["episodes"]:
        episode_id = str(episode["episode_id"])
        gt = gt_payload.get(episode_id)
        if gt is None:
            continue
        positions = np.asarray(gt.get("locations", []), dtype=np.float64)
        if positions.ndim != 2 or positions.shape[0] < 2 or positions.shape[1] < 3:
            continue
        records.append(
            R2RRecord(
                episode_id=episode_id,
                trajectory_id=int(episode.get("trajectory_id", -1)),
                scene_id=str(episode["scene_id"]),
                instruction_text=str(episode.get("instruction", {}).get("instruction_text", "")),
                positions=positions[:, :3],
                start_cell=_cell_from_position(positions[0], cell_size),
                goal_cell=_cell_from_position(positions[-1], cell_size),
            )
        )
    return records


def _r2r_branch_cells(records: list[R2RRecord], cell_size: float = 0.5) -> dict[tuple[int, int], set[int]]:
    outgoing: dict[tuple[int, int], set[int]] = defaultdict(set)
    for record in records:
        for start, end in zip(record.positions[:-1], record.positions[1:]):
            direction = _direction_bin(end - start)
            if direction is None:
                continue
            outgoing[_cell_from_position(start, cell_size)].add(direction)
    return {cell: dirs for cell, dirs in outgoing.items() if len(dirs) >= 2}


def _r2r_split_summary(records: list[R2RRecord]) -> dict[str, Any]:
    by_scene: dict[str, list[R2RRecord]] = defaultdict(list)
    for record in records:
        by_scene[record.scene_id].append(record)
    scene_summaries = []
    for scene_id, scene_records in by_scene.items():
        scene_summaries.append(
            {
                "scene_id": scene_id,
                "scene_key": _scene_key(scene_id),
                "episodes": len(scene_records),
                "unique_start_cells": len({r.start_cell for r in scene_records}),
                "unique_goal_cells": len({r.goal_cell for r in scene_records}),
                "branch_cells": len(_r2r_branch_cells(scene_records)),
            }
        )
    scene_summaries.sort(
        key=lambda row: (row["branch_cells"], row["unique_goal_cells"], row["episodes"]),
        reverse=True,
    )
    return {
        "episodes": len(records),
        "scenes": len(by_scene),
        "top_scenes": scene_summaries[:12],
    }


def _plot_r2r_scene(scene_id: str, records: list[R2RRecord], output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    branch_cells = _r2r_branch_cells(records)
    goal_counts_original = Counter(record.goal_cell for record in records)
    goal_counts_for_label = Counter(goal_counts_original)
    goal_order = [goal for goal, _ in goal_counts_original.most_common()]
    goal_rank = {goal: idx for idx, goal in enumerate(goal_order)}
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(max(1, len(goal_order)))]

    fig, ax = plt.subplots(figsize=(14.0, 9.0), constrained_layout=True)
    for record in records:
        rank = goal_rank[record.goal_cell]
        color = colors[rank]
        x = record.positions[:, 0]
        z = record.positions[:, 2]
        label = None
        if goal_counts_for_label[record.goal_cell] > 1 and rank < 10:
            label = f"goal-cell {rank} (n={goal_counts_original[record.goal_cell]})"
            goal_counts_for_label[record.goal_cell] = -goal_counts_for_label[record.goal_cell]
        ax.plot(x, z, color=color, alpha=0.44, linewidth=0.72, label=label)
        ax.scatter(x[0], z[0], s=12, marker="o", color=color, edgecolor="black", linewidth=0.25, alpha=0.72)
        ax.scatter(x[-1], z[-1], s=20, marker="x", color=color, linewidth=0.8, alpha=0.82)

    if branch_cells:
        branch_xy = np.asarray([(cell[0] * 0.5 + 0.25, cell[1] * 0.5 + 0.25) for cell in branch_cells], dtype=float)
        ax.scatter(
            branch_xy[:, 0],
            branch_xy[:, 1],
            s=13,
            marker="s",
            color="black",
            alpha=0.44,
            label="branch candidate cell",
        )

    ax.scatter([], [], marker="o", s=20, color="white", edgecolor="black", label="start")
    ax.scatter([], [], marker="x", s=24, color="black", label="goal")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Matterport x coordinate (m)")
    ax.set_ylabel("Matterport z coordinate (m)")
    ax.set_title(
        "R2R/VLN-CE real GT trajectories by scene\n"
        f"{_scene_key(scene_id)}: {len(records)} episodes, "
        f"{len({r.start_cell for r in records})} start cells, "
        f"{len({r.goal_cell for r in records})} goal cells, "
        f"{len(branch_cells)} branch candidate cells"
    )
    ax.grid(True, linewidth=0.3, alpha=0.28)
    handles, labels = ax.get_legend_handles_labels()
    dedup: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label and label not in dedup:
            dedup[label] = handle
    ax.legend(
        dedup.values(),
        dedup.keys(),
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=7,
        frameon=True,
        borderaxespad=0.0,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {
        "file": str(output_path),
        "scene_id": scene_id,
        "scene_key": _scene_key(scene_id),
        "episodes": len(records),
        "unique_start_cells": len({r.start_cell for r in records}),
        "unique_goal_cells": len({r.goal_cell for r in records}),
        "branch_cells": len(branch_cells),
    }


def _analyze_r2r(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    split_records = {split: _load_r2r_records(split) for split in ("train", "val_seen", "val_unseen")}
    summaries = {split: _r2r_split_summary(records) for split, records in split_records.items()}

    train_by_scene: dict[str, list[R2RRecord]] = defaultdict(list)
    for record in split_records["train"]:
        train_by_scene[record.scene_id].append(record)
    top_scene_ids = [row["scene_id"] for row in summaries["train"]["top_scenes"][:5]]
    plot_files = []
    for rank, scene_id in enumerate(top_scene_ids):
        plot_files.append(
            _plot_r2r_scene(
                scene_id,
                train_by_scene[scene_id],
                output_root / f"navigation_r2r_scene_{rank:02d}_{_scene_key(scene_id)}.png",
            )
        )

    payload = {
        "dataset_path": str(R2R_DATASET_ROOT),
        "source_dataset": "VLN-CE R2R_VLNCE_v1-3_preprocessed",
        "trajectory_source": "*_gt.json.gz locations",
        "branch_definition": "A 0.5 m x-z cell visited by trajectories with at least two outgoing 8-bin direction choices.",
        "splits": summaries,
        "plots": plot_files,
    }
    _safe_json_dump(output_root / "navigation_r2r_summary.json", payload)
    return payload


def _episode_names(file_handle: h5py.File) -> list[str]:
    return sorted([key for key in file_handle.keys() if key.startswith("episode_")], key=lambda item: int(item.split("_")[-1]))


def _progress_for_episode(group: h5py.Group) -> tuple[tuple[str, ...], np.ndarray]:
    achieved_group = group["observations"]["achieved_goal"]
    desired_group = group["observations"]["desired_goal"]
    objects = tuple(sorted(achieved_group.keys()))
    progress_columns = []
    for object_name in objects:
        achieved = np.asarray(achieved_group[object_name], dtype=np.float64)
        desired = np.asarray(desired_group[object_name], dtype=np.float64)
        target = desired[-1]
        distances = np.linalg.norm(achieved - target, axis=1)
        denom = max(float(distances[0]), 1e-9)
        progress = np.clip(1.0 - distances / denom, 0.0, 1.0)
        progress_columns.append(progress)
    return objects, np.stack(progress_columns, axis=1)


def _completion_order(objects: tuple[str, ...], progress: np.ndarray, threshold: float = 0.60) -> tuple[str, ...]:
    events = []
    for idx, object_name in enumerate(objects):
        reached = np.flatnonzero(progress[:, idx] >= threshold)
        if reached.size:
            events.append((int(reached[0]), object_name))
    events.sort(key=lambda row: (row[0], row[1]))
    return tuple(object_name for _, object_name in events)


def _load_kitchen_episodes(dataset_name: str) -> list[KitchenEpisode]:
    hdf5_path = KITCHEN_DATASET_ROOT / dataset_name / "data" / "main_data.hdf5"
    episodes: list[KitchenEpisode] = []
    with h5py.File(hdf5_path, "r") as handle:
        for episode_name in _episode_names(handle):
            group = handle[episode_name]
            objects, progress = _progress_for_episode(group)
            episodes.append(
                KitchenEpisode(
                    dataset_name=dataset_name,
                    episode_name=episode_name,
                    objects=objects,
                    progress=progress,
                    order=_completion_order(objects, progress),
                )
            )
    return episodes


def _pca_2d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    projected = centered @ components.T
    variance = np.var(projected, axis=0)
    total = float(np.var(centered, axis=0).sum())
    explained = variance / total if total > 0 else np.zeros_like(variance)
    return projected, components, explained


def _plot_kitchen_order_counts(episodes_by_dataset: dict[str, list[KitchenEpisode]], output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    datasets = list(episodes_by_dataset.keys())
    counters = {name: Counter(ep.order for ep in episodes) for name, episodes in episodes_by_dataset.items()}
    merged = Counter()
    for counter in counters.values():
        merged.update(counter)
    top_orders = []
    for order in [*counters.get("complete-v2", Counter()).keys(), *[order for order, _ in merged.most_common(12)]]:
        if order not in top_orders:
            top_orders.append(order)
    top_orders = top_orders[:12]

    fig, ax = plt.subplots(figsize=(13.5, 7.8), constrained_layout=True)
    x = np.arange(len(top_orders))
    width = 0.24
    colors = {"complete-v2": "#4c78a8", "partial-v2": "#f58518", "mixed-v2": "#54a24b"}
    for i, dataset_name in enumerate(datasets):
        values = [counters[dataset_name].get(order, 0) for order in top_orders]
        ax.bar(x + (i - 1) * width, values, width=width, color=colors.get(dataset_name), label=dataset_name)
    labels = ["\n-> ".join(order) if order else "no threshold event" for order in top_orders]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=7)
    ax.set_ylabel("episode count")
    ax.set_title("D4RL/Minari Kitchen heuristic subtask-order diversity\nthreshold: first time object progress >= 0.60")
    ax.legend(frameon=True)
    ax.grid(axis="y", linewidth=0.3, alpha=0.35)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {"file": str(output_path), "top_orders": [[*order] for order in top_orders]}


def _plot_kitchen_progress_embedding(episodes_by_dataset: dict[str, list[KitchenEpisode]], output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_points = []
    path_payloads = []
    rng = np.random.default_rng(7)
    for dataset_name, episodes in episodes_by_dataset.items():
        selected = episodes if len(episodes) <= 80 else list(rng.choice(episodes, size=80, replace=False))
        for episode in selected:
            stride = max(1, episode.progress.shape[0] // 80)
            sampled = episode.progress[::stride]
            all_points.append(sampled)
            path_payloads.append((dataset_name, sampled))
    point_matrix = np.concatenate(all_points, axis=0)
    projected, components, explained = _pca_2d(point_matrix)

    cursor = 0
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), constrained_layout=True, sharex=True, sharey=True)
    dataset_order = ["complete-v2", "partial-v2", "mixed-v2"]
    colors = {"complete-v2": "#4c78a8", "partial-v2": "#f58518", "mixed-v2": "#54a24b"}
    for dataset_name, sampled in path_payloads:
        n = sampled.shape[0]
        coords = projected[cursor : cursor + n]
        cursor += n
        ax = axes[dataset_order.index(dataset_name)]
        ax.plot(coords[:, 0], coords[:, 1], color=colors[dataset_name], linewidth=0.55, alpha=0.25)
        ax.scatter(coords[0, 0], coords[0, 1], s=10, marker="o", color=colors[dataset_name], edgecolor="black", linewidth=0.15, alpha=0.50)
        ax.scatter(coords[-1, 0], coords[-1, 1], s=14, marker="x", color=colors[dataset_name], linewidth=0.55, alpha=0.58)
    for ax, dataset_name in zip(axes, dataset_order):
        ax.set_title(f"{dataset_name}: sampled real episodes")
        ax.set_xlabel("PC1 from achieved-goal progress")
        ax.grid(True, linewidth=0.3, alpha=0.30)
    axes[0].set_ylabel("PC2 from achieved-goal progress")
    axes[-1].legend(
        handles=[
            Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=5, label="start"),
            Line2D([0], [0], marker="x", color="black", linestyle="None", markersize=5, label="end"),
        ],
        loc="upper right",
        frameon=True,
    )
    fig.suptitle(
        "Kitchen state-progress trajectories in a shared 2D embedding\n"
        f"PCA explained variance: PC1={explained[0]:.2f}, PC2={explained[1]:.2f}; marker legend shows starts and ends"
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {
        "file": str(output_path),
        "pca_explained_variance": [float(explained[0]), float(explained[1])],
        "components": components.tolist(),
    }


def _analyze_kitchen(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    datasets = ["complete-v2", "partial-v2", "mixed-v2"]
    episodes_by_dataset = {dataset: _load_kitchen_episodes(dataset) for dataset in datasets}
    summaries = {}
    for dataset_name, episodes in episodes_by_dataset.items():
        order_counts = Counter(ep.order for ep in episodes)
        objects = sorted({obj for episode in episodes for obj in episode.objects})
        summaries[dataset_name] = {
            "hdf5_path": str(KITCHEN_DATASET_ROOT / dataset_name / "data" / "main_data.hdf5"),
            "episodes": len(episodes),
            "objects": objects,
            "unique_heuristic_orders": len(order_counts),
            "top_heuristic_orders": [
                {"order": [*order], "count": count} for order, count in order_counts.most_common(12)
            ],
        }

    plots = [
        _plot_kitchen_order_counts(episodes_by_dataset, output_root / "robotics_kitchen_subtask_order_counts.png"),
        _plot_kitchen_progress_embedding(episodes_by_dataset, output_root / "robotics_kitchen_progress_embedding.png"),
    ]
    payload = {
        "dataset_path": str(KITCHEN_DATASET_ROOT),
        "source_dataset": "farama-minari/D4RL kitchen partial-v2, complete-v2, mixed-v2",
        "state_source": "HDF5 observations/achieved_goal and observations/desired_goal",
        "branch_definition": "Different object-completion orders from real achieved-goal trajectories under a common task object set.",
        "completion_order_rule": "first timestep where normalized object progress reaches 0.60",
        "datasets": summaries,
        "plots": plots,
    }
    _safe_json_dump(output_root / "robotics_kitchen_summary.json", payload)
    return payload


def run_predictive_branching_dataset_analysis(cfg: Any | None = None) -> dict[str, Any]:
    del cfg
    if not R2R_DATASET_ROOT.exists():
        raise FileNotFoundError(f"Missing R2R/VLN-CE dataset root: {R2R_DATASET_ROOT}")
    if not KITCHEN_DATASET_ROOT.exists():
        raise FileNotFoundError(f"Missing D4RL Kitchen dataset root: {KITCHEN_DATASET_ROOT}")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    navigation = _analyze_r2r(RESULT_ROOT / "navigation_r2r_vlnce")
    robotics = _analyze_kitchen(RESULT_ROOT / "robotics_d4rl_kitchen")
    manifest = {
        "status": "passed",
        "result_root": str(RESULT_ROOT),
        "artifact_root": str(ARTIFACT_ROOT),
        "datasets": {
            "navigation": {
                "name": "r2r_vlnce_v1_3_preprocessed",
                "path": str(R2R_DATASET_ROOT),
                "episodes_train": navigation["splits"]["train"]["episodes"],
                "scenes_train": navigation["splits"]["train"]["scenes"],
                "top_scene_branch_cells": navigation["splits"]["train"]["top_scenes"][0]["branch_cells"],
            },
            "robotics": {
                "name": "d4rl_kitchen_compositional_v2",
                "path": str(KITCHEN_DATASET_ROOT),
                "datasets": {
                    name: {
                        "episodes": summary["episodes"],
                        "unique_heuristic_orders": summary["unique_heuristic_orders"],
                    }
                    for name, summary in robotics["datasets"].items()
                },
            },
        },
        "outputs": {
            "navigation_summary": str(RESULT_ROOT / "navigation_r2r_vlnce" / "navigation_r2r_summary.json"),
            "robotics_summary": str(RESULT_ROOT / "robotics_d4rl_kitchen" / "robotics_kitchen_summary.json"),
            "png_files": [
                *[plot["file"] for plot in navigation["plots"]],
                *[plot["file"] for plot in robotics["plots"]],
            ],
        },
    }
    _safe_json_dump(RESULT_ROOT / "manifest.json", manifest)
    _copy_outputs_to_artifact_lane(RESULT_ROOT, ARTIFACT_ROOT)
    return manifest
