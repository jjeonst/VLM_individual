"""Notebook helpers for HM3D trajectory and branch-structure visualizations."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from configs.builder import build_config_from_exp
from topovlm_data.habitat_manifest import resolve_data_path


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION_NAMES = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
}
DEFAULT_DATA_ROOT = Path("data/habitat")
DEFAULT_EPISODE_MANIFEST = Path("episodes/pr2l_hm3d_objectnav/train/manifest.jsonl")
DEFAULT_GRAPH_MANIFEST = Path("graphs/pr2l_hm3d_bc/train/manifest.jsonl")
DEFAULT_EXP = "habitat/pr2l_hm3d_bc"
DEFAULT_RESULT_DIR = Path("analysis/results/hm3d_trajectory_notebook_views")


def load_episode_manifest(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    manifest_path: str | Path = DEFAULT_EPISODE_MANIFEST,
) -> list[dict[str, object]]:
    path = resolve_data_path(Path(data_root), manifest_path)
    return _load_jsonl(path)


def load_graph_manifest(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    manifest_path: str | Path = DEFAULT_GRAPH_MANIFEST,
) -> list[dict[str, object]]:
    path = resolve_data_path(Path(data_root), manifest_path)
    return _load_jsonl(path)


def select_scene_trajectory_records(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    *,
    max_trajectories: int | None = 6,
    min_steps: int = 80,
    min_turns: int = 5,
    require_graph_cache: bool = True,
) -> list[dict[str, object]]:
    root = Path(data_root)
    records = load_episode_manifest(root)
    graph_ids = _graph_episode_ids(root) if require_graph_cache else None
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if graph_ids is not None and str(record["episode_id"]) not in graph_ids:
            continue
        actions = load_actions(record, root)
        turns = int(np.isin(actions, [2, 3]).sum())
        if len(actions) < min_steps or turns < min_turns:
            continue
        enriched = dict(record)
        enriched["analysis_steps"] = int(len(actions))
        enriched["analysis_turns"] = turns
        enriched["analysis_first_turn"] = first_turn_label(actions)
        key = (str(record["scene_id"]), str(record.get("object_category", record["goal_text"])))
        groups[key].append(enriched)
    if not groups:
        raise ValueError("No HM3D scene/object group satisfies the trajectory selection criteria.")
    ranked_groups = sorted(
        groups.values(),
        key=lambda group: (
            -len(group),
            -float(np.mean([record["analysis_turns"] for record in group])),
            str(group[0]["scene_id"]),
            str(group[0].get("object_category", group[0]["goal_text"])),
        ),
    )
    for group in ranked_groups:
        if max_trajectories is None:
            return _round_robin_first_turns(group, len(group))
        if len(group) >= max_trajectories:
            return _round_robin_first_turns(group, max_trajectories)
    best_group = ranked_groups[0]
    return _round_robin_first_turns(best_group, min(max_trajectories, len(best_group)))


def trajectory_selection_summary(records: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = []
    for index, record in enumerate(records, start=1):
        summary.append(
            {
                "trajectory": f"T{index:02d}",
                "episode_id": record["episode_id"],
                "scene": scene_name(record),
                "object": record.get("object_category", record["goal_text"]),
                "steps": record.get("analysis_steps"),
                "turns": record.get("analysis_turns"),
                "first_turn": record.get("analysis_first_turn"),
            }
        )
    return summary


def load_actions(record: dict[str, object], data_root: str | Path = DEFAULT_DATA_ROOT) -> np.ndarray:
    path = resolve_data_path(Path(data_root), str(record["actions_path"]))
    return np.load(path).astype(np.int64)


def first_turn_label(actions: np.ndarray) -> str:
    for action in actions:
        if int(action) == 2:
            return "first_left"
        if int(action) == 3:
            return "first_right"
        if int(action) == 0:
            return "stop_before_turn"
    return "no_turn"


def scene_name(record: dict[str, object]) -> str:
    scene_path = Path(str(record["scene_id"]))
    parent = scene_path.parent.name
    return parent if parent else scene_path.stem


def trajectory_labels(records: list[dict[str, object]]) -> dict[str, str]:
    return {
        str(record["episode_id"]): f"T{index:02d} {record.get('analysis_first_turn', '')}".strip()
        for index, record in enumerate(records, start=1)
    }


def trajectory_colors(records: list[dict[str, object]]) -> dict[str, tuple[float, float, float, float]]:
    if len(records) <= 20:
        cmap = plt.get_cmap("tab20")
        return {
            str(record["episode_id"]): cmap((index - 1) % 20)
            for index, record in enumerate(records, start=1)
        }
    cmap = plt.get_cmap("hsv")
    return {
        str(record["episode_id"]): cmap((index - 1) / max(len(records), 1))
        for index, record in enumerate(records, start=1)
    }


def replay_habitat_topdown(
    records: list[dict[str, object]],
    data_root: str | Path = DEFAULT_DATA_ROOT,
    *,
    exp: str = DEFAULT_EXP,
    max_steps: int | None = None,
) -> dict[str, object]:
    from habitat.utils.visualizations import maps
    from topovlm_data.habitat_objectnav import objectnav_source_trajectory_id
    from topovlm_data.hm3d_objectnav_render import _filter_env_episodes_to_selection, _open_habitat_env

    root = Path(data_root)
    cfg = build_config_from_exp(exp)
    cfg.data.data_root = str(root)
    habitat_config = Path(cfg.data.habitat_config)
    if not habitat_config.is_absolute():
        cfg.data.habitat_config = str(REPO_ROOT / habitat_config)
    selection_ids = {str(record["source_trajectory_id"]) for record in records}
    record_by_source_id = {str(record["source_trajectory_id"]): record for record in records}
    env = _open_habitat_env(cfg)
    try:
        _filter_env_episodes_to_selection(env, selection_ids)
        trajectories_by_episode_id = {}
        for _ in range(len(env.episodes)):
            env.reset()
            episode = env.current_episode
            source_id = objectnav_source_trajectory_id(episode)
            if source_id not in selection_ids:
                continue
            record = record_by_source_id[source_id]
            actions = load_actions(record, root)
            limit = len(actions) if max_steps is None else min(len(actions), int(max_steps))
            positions = [_agent_position(env)]
            replayed_actions = []
            for action in actions[:limit]:
                action_id = int(action)
                replayed_actions.append(action_id)
                if action_id == 0:
                    break
                env.step(action_id)
                positions.append(_agent_position(env))
            trajectories_by_episode_id[str(record["episode_id"])] = {
                "record": record,
                "positions": np.asarray(positions, dtype=np.float32),
                "actions": np.asarray(replayed_actions, dtype=np.int64),
            }
        missing = {str(record["episode_id"]) for record in records}.difference(
            trajectories_by_episode_id
        )
        if missing:
            raise ValueError(f"Habitat replay missed selected episodes: {sorted(missing)[:3]}")
        trajectory_heights = np.concatenate(
            [
                trajectory["positions"][:, 1]
                for trajectory in trajectories_by_episode_id.values()
            ]
        )
        height_bin_meters = 0.5
        height_bins = np.round(trajectory_heights / height_bin_meters)
        map_heights = np.asarray(
            [
                float(np.median(trajectory_heights[height_bins == height_bin]))
                for height_bin in np.unique(height_bins)
            ],
            dtype=np.float32,
        )
        topdown_layers = [
            maps.get_topdown_map(
                env.sim.pathfinder,
                float(height),
                map_resolution=1024,
                draw_border=True,
            )
            for height in map_heights
        ]
        topdown_map = np.maximum.reduce(topdown_layers)
        for trajectory in trajectories_by_episode_id.values():
            trajectory["grid_positions"] = np.asarray(
                [
                    maps.to_grid(
                        float(position[2]),
                        float(position[0]),
                        topdown_map.shape[:2],
                        sim=env.sim,
                    )
                    for position in trajectory["positions"]
                ],
                dtype=np.int64,
            )
            height, width = topdown_map.shape[:2]
            grid = trajectory["grid_positions"]
            if (
                np.any(grid[:, 0] < 0)
                or np.any(grid[:, 0] >= height)
                or np.any(grid[:, 1] < 0)
                or np.any(grid[:, 1] >= width)
            ):
                raise ValueError(
                    "Habitat replay produced trajectory coordinates outside the top-down map."
                )
        return {
            "scene": scene_name(records[0]),
            "object": records[0].get("object_category", records[0]["goal_text"]),
            "topdown_map": topdown_map,
            "topdown_map_heights": map_heights,
            "records": records,
            "trajectories": [
                trajectories_by_episode_id[str(record["episode_id"])] for record in records
            ],
        }
    finally:
        env.close()


def plot_habitat_topdown(replay: dict[str, object]) -> tuple[plt.Figure, plt.Axes]:
    records = list(replay["records"])
    labels = trajectory_labels(records)
    colors = trajectory_colors(records)
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.imshow(replay["topdown_map"], cmap="gray", origin="upper")
    ax.set_title(f"Habitat traversed-floor map with {len(records)} expert trajectories")
    ax.set_axis_off()
    for trajectory in replay["trajectories"]:
        record = trajectory["record"]
        episode_id = str(record["episode_id"])
        color = colors[episode_id]
        label = labels[episode_id]
        grid = trajectory["grid_positions"]
        ax.plot(
            grid[:, 1],
            grid[:, 0],
            color=color,
            linewidth=1.8,
            alpha=0.9,
            label=label,
        )
        ax.scatter(
            grid[0, 1],
            grid[0, 0],
            color=color,
            s=42,
            marker="o",
            edgecolor="black",
        )
        ax.scatter(grid[-1, 1], grid[-1, 0], color=color, s=58, marker="x")
    legend_columns = min(4, max(1, int(np.ceil(len(records) / 18))))
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=7,
        ncol=legend_columns,
        title="Trajectory",
    )
    fig.suptitle(f"Same Habitat environment: {replay['scene']} / object={replay['object']}")
    return fig, ax


def load_observation_feature_rows(
    records: list[dict[str, object]],
    data_root: str | Path = DEFAULT_DATA_ROOT,
    *,
    max_frames_per_trajectory: int = 64,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    features = []
    rows = []
    root = Path(data_root)
    for trajectory_order, record in enumerate(records, start=1):
        rgb = np.load(resolve_data_path(root, str(record["rgb_path"])), mmap_mode="r")
        actions = load_actions(record, root)
        for step in _sample_frame_indices(len(rgb), max_frames_per_trajectory):
            features.append(_rgb_descriptor(np.asarray(rgb[step])))
            rows.append(_trajectory_row(record, trajectory_order, int(step), int(actions[min(step, len(actions) - 1)])))
    return np.stack(features, axis=0).astype(np.float32), rows


def load_vlm_node_feature_rows(
    records: list[dict[str, object]],
    data_root: str | Path = DEFAULT_DATA_ROOT,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    root = Path(data_root)
    graph_by_episode = {str(record["episode_id"]): record for record in load_graph_manifest(root)}
    features = []
    rows = []
    for trajectory_order, record in enumerate(records, start=1):
        graph_record = graph_by_episode[str(record["episode_id"])]
        payload = np.load(resolve_data_path(root, str(graph_record["graph_path"])))
        nodes = np.asarray(payload["nodes"], dtype=np.float32)
        if nodes.ndim == 3:
            nodes = nodes.mean(axis=1)
        frame_ranges = np.asarray(payload["frame_ranges"])
        node_actions = np.asarray(payload["node_actions"])
        for node_index, node_feature in enumerate(nodes):
            frame_start = int(frame_ranges[node_index, 0]) if frame_ranges.ndim == 2 else node_index
            action = int(node_actions[node_index]) if node_index < len(node_actions) else -1
            features.append(node_feature)
            row = _trajectory_row(record, trajectory_order, frame_start, action)
            row["node_index"] = int(node_index)
            rows.append(row)
    return np.stack(features, axis=0).astype(np.float32), rows


def pca_2d(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:2].T
    variance = singular_values**2
    explained = variance[:2] / variance.sum() if variance.sum() > 0 else np.zeros(2)
    return coords.astype(np.float32), explained.astype(np.float32)


def save_figure(
    fig: plt.Figure,
    filename: str,
    result_dir: str | Path = DEFAULT_RESULT_DIR,
) -> Path:
    output_dir = Path(result_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    return path


def plot_embedding_trajectories(
    coords: np.ndarray,
    rows: list[dict[str, object]],
    records: list[dict[str, object]],
    *,
    title: str,
) -> tuple[plt.Figure, plt.Axes]:
    labels = trajectory_labels(records)
    colors = trajectory_colors(records)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    row_indices_by_episode: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        row_indices_by_episode[str(row["episode_id"])].append(index)
    for record in records:
        episode_id = str(record["episode_id"])
        indices = sorted(row_indices_by_episode[episode_id], key=lambda index: int(rows[index]["step"]))
        if not indices:
            continue
        xy = coords[indices]
        color = colors[episode_id]
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.7, alpha=0.8, label=labels[episode_id])
        ax.scatter(xy[:, 0], xy[:, 1], color=color, s=18, alpha=0.75)
        ax.scatter(xy[0, 0], xy[0, 1], color=color, s=48, marker="o", edgecolor="black")
        ax.scatter(xy[-1, 0], xy[-1, 1], color=color, s=64, marker="x")
    ax.axhline(0.0, color="0.85", linewidth=0.8)
    ax.axvline(0.0, color="0.85", linewidth=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    return fig, ax


def action_name(action_id: int) -> str:
    return ACTION_NAMES.get(int(action_id), f"ACTION_{action_id}")


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _graph_episode_ids(root: Path) -> set[str]:
    return {str(record["episode_id"]) for record in load_graph_manifest(root)}


def _round_robin_first_turns(
    records: list[dict[str, object]], max_trajectories: int
) -> list[dict[str, object]]:
    by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_label[str(record["analysis_first_turn"])].append(record)
    for label_records in by_label.values():
        label_records.sort(
            key=lambda record: (
                -int(record["analysis_turns"]),
                -int(record["analysis_steps"]),
                str(record["episode_id"]),
            )
        )
    label_order = ["first_left", "first_right", "stop_before_turn", "no_turn"]
    chosen = []
    while len(chosen) < max_trajectories:
        added = False
        for label in label_order:
            if by_label[label]:
                chosen.append(by_label[label].pop(0))
                added = True
                if len(chosen) == max_trajectories:
                    break
        if not added:
            break
    chosen.sort(key=lambda record: str(record["episode_id"]))
    return chosen


def _sample_frame_indices(length: int, max_count: int) -> np.ndarray:
    if length <= max_count:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, max_count).round().astype(np.int64))


def _rgb_descriptor(frame: np.ndarray) -> np.ndarray:
    image = Image.fromarray(frame.astype("uint8")).resize((16, 16), Image.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    histograms = []
    for channel in range(3):
        hist, _ = np.histogram(pixels[..., channel], bins=8, range=(0.0, 1.0), density=True)
        histograms.append(hist.astype(np.float32))
    return np.concatenate([pixels.reshape(-1), *histograms], axis=0)


def _trajectory_row(
    record: dict[str, object], trajectory_order: int, step: int, action: int
) -> dict[str, object]:
    return {
        "trajectory": f"T{trajectory_order:02d}",
        "episode_id": str(record["episode_id"]),
        "scene": scene_name(record),
        "object": record.get("object_category", record["goal_text"]),
        "step": int(step),
        "action": int(action),
        "action_name": action_name(action),
        "first_turn": record.get("analysis_first_turn"),
    }


def _agent_position(env: object) -> np.ndarray:
    return np.asarray(env.sim.get_agent_state().position, dtype=np.float32).copy()
