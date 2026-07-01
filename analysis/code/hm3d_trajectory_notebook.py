"""Notebook helpers for HM3D trajectory and branch-structure visualizations."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
import colorsys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
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
MAP_HEIGHT_BIN_METERS = 0.5


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
    if max_trajectories is None:
        best_group = ranked_groups[0]
        return _round_robin_first_turns(best_group, len(best_group))

    for group in ranked_groups:
        if len(group) >= max_trajectories:
            return _round_robin_first_turns(group, max_trajectories)
    best_group = ranked_groups[0]
    return _round_robin_first_turns(best_group, min(max_trajectories, len(best_group)))


def replay_selected_objectnav_scene_topdowns(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    *,
    exp: str = DEFAULT_EXP,
    on_scene_replay: Callable[[int, dict[str, object]], None] | None = None,
) -> list[dict[str, object]]:
    from habitat.utils.visualizations import maps
    from topovlm_data.habitat_objectnav import load_objectnav_selection_records
    from topovlm_data.hm3d_objectnav_render import (
        _open_habitat_env,
        _select_goal_position,
    )

    root = Path(data_root)
    cfg = build_config_from_exp(exp)
    cfg.data.data_root = str(root)
    habitat_config = Path(cfg.data.habitat_config)
    if not habitat_config.is_absolute():
        cfg.data.habitat_config = str(REPO_ROOT / habitat_config)
    selections = load_objectnav_selection_records(cfg.data)
    selection_by_key = {_selection_record_key(selection): selection for selection in selections}
    if len(selection_by_key) != len(selections):
        raise ValueError("Duplicate ObjectNav selection keys in selection manifest")
    payload_by_key = _episode_payloads_by_selection_key(root)
    missing_payload_keys = set(selection_by_key).difference(payload_by_key)
    if missing_payload_keys:
        raise ValueError(
            f"Selected ObjectNav keys missing materialized expert actions: "
            f"{sorted(missing_payload_keys)[:3]}"
        )

    selections_by_scene: dict[str, dict[tuple[str, str], object]] = defaultdict(dict)
    for key, selection in selection_by_key.items():
        selections_by_scene[scene_name({"scene_id": selection.scene_id})][key] = selection
    scene_replays = []
    seen_keys = set()
    env = _open_habitat_env(cfg)
    try:
        _filter_env_episodes_to_selection_keys(env, selection_by_key)
        episode_by_key = {_episode_selection_key(episode): episode for episode in env.episodes}
        missing_episode_keys = set(selection_by_key).difference(episode_by_key)
        if missing_episode_keys:
            raise ValueError(f"Filtered Habitat env missed keys: {sorted(missing_episode_keys)[:3]}")

        for scene in sorted(selections_by_scene):
            scene_selection_by_key = selections_by_scene[scene]
            replay = {
                "scene": scene,
                "topdown_map": None,
                "topdown_map_heights": [],
                "_topdown_map_height_bins": set(),
                "records": [],
                "trajectories": [],
            }
            scene_seen_keys = set()
            ordered_scene_keys = sorted(
                scene_selection_by_key,
                key=lambda key: (
                    str(scene_selection_by_key[key].object_category),
                    key[0],
                ),
            )
            for target_key in ordered_scene_keys:
                episode_to_replay = episode_by_key[target_key]
                env.episodes = [episode_to_replay]
                env.reset()
                episode = env.current_episode
                key = _episode_selection_key(episode)
                if key != target_key:
                    raise ValueError(f"Habitat reset returned {key} instead of {target_key}")
                selection = scene_selection_by_key[key]
                payload_record = payload_by_key[key]
                actions = load_actions(payload_record, root)
                goal_position = _select_goal_position(env)
                positions = [_agent_position(env)]
                replayed_actions = []
                for step_index, action in enumerate(actions, start=1):
                    action = int(action)
                    replayed_actions.append(action)
                    if action == 0:
                        break
                    if step_index > cfg.eval.max_steps:
                        raise ValueError(
                            f"Materialized expert actions exceeded max_steps={cfg.eval.max_steps} "
                            f"for {selection.source_trajectory_id} {selection.object_category}"
                        )
                    env.step(action)
                    positions.append(_agent_position(env))
                else:
                    raise ValueError(
                        f"Materialized expert actions did not include STOP "
                        f"for {selection.source_trajectory_id} {selection.object_category}"
                    )
                record = _selection_to_plot_record(selection, replayed_actions)
                trajectory = {
                    "record": record,
                    "positions": np.asarray(positions, dtype=np.float32),
                    "goal_position": np.asarray(goal_position, dtype=np.float32),
                    "actions": np.asarray(replayed_actions, dtype=np.int64),
                }
                for height in _map_heights_from_trajectory_heights(trajectory["positions"][:, 1]):
                    height_bin = int(round(float(height) / MAP_HEIGHT_BIN_METERS))
                    if height_bin in replay["_topdown_map_height_bins"]:
                        continue
                    topdown_layer = maps.get_topdown_map(
                        env.sim.pathfinder,
                        float(height),
                        map_resolution=768,
                        draw_border=True,
                    )
                    replay["topdown_map"] = (
                        topdown_layer
                        if replay["topdown_map"] is None
                        else np.maximum(replay["topdown_map"], topdown_layer)
                    )
                    replay["topdown_map_heights"].append(float(height))
                    replay["_topdown_map_height_bins"].add(height_bin)
                if replay["topdown_map"] is None:
                    raise ValueError(f"Could not build top-down map for scene {scene}")
                trajectory["grid_positions"] = _positions_to_grid(
                    maps, env, replay["topdown_map"], trajectory["positions"]
                )
                trajectory["goal_grid_position"] = _positions_to_grid(
                    maps, env, replay["topdown_map"], trajectory["goal_position"][None, :]
                )[0]
                replay["records"].append(record)
                replay["trajectories"].append(trajectory)
                scene_seen_keys.add(key)
                seen_keys.add(key)
            missing_scene_keys = set(scene_selection_by_key).difference(scene_seen_keys)
            if missing_scene_keys:
                raise ValueError(
                    f"Habitat replay missed scene keys for {scene}: "
                    f"{sorted(missing_scene_keys)[:3]}"
                )
            ordered_trajectories = sorted(
                replay["trajectories"],
                key=lambda trajectory: trajectory_record_key(trajectory["record"]),
            )
            replay["records"] = [trajectory["record"] for trajectory in ordered_trajectories]
            replay["trajectories"] = ordered_trajectories
            replay["topdown_map_heights"] = np.asarray(
                sorted(set(float(height) for height in replay["topdown_map_heights"])),
                dtype=np.float32,
            )
            replay.pop("_topdown_map_height_bins")
            if on_scene_replay is not None:
                on_scene_replay(len(scene_replays), replay)
            scene_replays.append(replay)
    finally:
        env.close()

    missing = set(selection_by_key).difference(seen_keys)
    if missing:
        raise ValueError(f"Habitat replay missed selected ObjectNav keys: {sorted(missing)[:3]}")
    return scene_replays


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


def language_instruction(record: dict[str, object]) -> str:
    return str(record.get("goal_text", record.get("object_category", "unknown")))


def trajectory_group(record: dict[str, object]) -> str:
    return language_instruction(record)


def scene_name(record: dict[str, object]) -> str:
    scene_path = Path(str(record["scene_id"]))
    parent = scene_path.parent.name
    return parent if parent else scene_path.stem


def trajectory_record_key(record: dict[str, object]) -> str:
    source_id = str(record.get("source_trajectory_id", ""))
    if not source_id:
        source_id = f"{scene_name(record)}:{record['episode_id']}"
    object_category = str(record.get("object_category", record.get("goal_text", "unknown")))
    return f"{source_id}|{object_category}"


def trajectory_labels(records: list[dict[str, object]]) -> dict[str, str]:
    return {
        trajectory_record_key(record): f"T{index:02d} {record.get('analysis_first_turn', '')}".strip()
        for index, record in enumerate(records, start=1)
    }


def trajectory_colors(records: list[dict[str, object]]) -> dict[str, tuple[float, float, float, float]]:
    return group_shaded_trajectory_colors(records)


def group_shaded_trajectory_colors(
    records: list[dict[str, object]],
    *,
    alpha: float = 0.9,
) -> dict[str, tuple[float, float, float, float]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        groups[trajectory_group(record)].append(record)
    base_colors = group_base_colors(records)
    colors = {}
    for group, group_records in groups.items():
        group_records.sort(key=lambda record: (scene_name(record), trajectory_record_key(record)))
        total = max(1, len(group_records) - 1)
        for index, record in enumerate(group_records):
            frac = index / total if total else 0.5
            colors[trajectory_record_key(record)] = _adjust_color_lightness(
                base_colors[group],
                0.34 + 0.42 * frac,
                alpha,
            )
    return colors


def group_base_colors(records: list[dict[str, object]]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    groups = sorted({trajectory_group(record) for record in records})
    return {group: cmap(index % 10) for index, group in enumerate(groups)}


def group_legend_handles(records: list[dict[str, object]]) -> list[Line2D]:
    counts = Counter(trajectory_group(record) for record in records)
    base_colors = group_base_colors(records)
    return [
        Line2D(
            [0],
            [0],
            color=base_colors[group],
            linewidth=2.3,
            label=f"goal_text: {group} | n={counts[group]}",
        )
        for group in sorted(counts)
    ]


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


def marker_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
            label="start pose (circle)",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="black",
            linestyle="none",
            markersize=7,
            label="agent STOP / last pose (x)",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="black",
            markerfacecolor="white",
            linestyle="none",
            markersize=9,
            label="object goal position (star)",
        ),
    ]


def plot_habitat_topdown(replay: dict[str, object]) -> tuple[plt.Figure, plt.Axes]:
    records = list(replay["records"])
    colors = group_shaded_trajectory_colors(records, alpha=0.85)
    unique_goal_count = _unique_goal_position_count(replay)
    unique_stop_count = _unique_stop_position_count(replay)
    fig, ax = plt.subplots(figsize=(9.5, 9), constrained_layout=True)
    ax.imshow(replay["topdown_map"], cmap="gray", origin="upper")
    ax.set_title(
        f"{replay['scene']}: {len(records)} expert trajectories, "
        f"{unique_goal_count} object goals, {unique_stop_count} STOP poses"
    )
    ax.set_axis_off()
    for trajectory in replay["trajectories"]:
        record = trajectory["record"]
        color = colors[trajectory_record_key(record)]
        grid = trajectory["grid_positions"]
        (line,) = ax.plot(
            grid[:, 1],
            grid[:, 0],
            color=color,
            linewidth=1.15,
            alpha=0.9,
            zorder=3,
        )
        line.set_path_effects(
            [path_effects.Stroke(linewidth=2.0, foreground="white", alpha=0.35), path_effects.Normal()]
        )
        ax.scatter(
            grid[0, 1],
            grid[0, 0],
            color=color,
            s=15,
            marker="o",
            edgecolor="black",
            linewidth=0.35,
            alpha=0.72,
            zorder=4,
        )
        ax.scatter(
            grid[-1, 1],
            grid[-1, 0],
            color=color,
            s=24,
            marker="x",
            linewidths=0.7,
            alpha=0.82,
            zorder=4,
        )
    _annotate_goal_positions(ax, replay, records)
    group_legend = ax.legend(
        handles=group_legend_handles(records),
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        title="Language instruction",
    )
    ax.add_artist(group_legend)
    ax.legend(
        handles=marker_legend_handles(),
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        title="Markers",
    )
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


def plot_embedding_trajectories(
    coords: np.ndarray,
    rows: list[dict[str, object]],
    records: list[dict[str, object]],
    *,
    title: str,
) -> tuple[plt.Figure, plt.Axes]:
    labels = trajectory_labels(records)
    colors = group_shaded_trajectory_colors(records, alpha=0.78)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    row_indices_by_episode: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        row_indices_by_episode[str(row["trajectory_key"])].append(index)
    for record in records:
        trajectory_key = trajectory_record_key(record)
        indices = sorted(
            row_indices_by_episode[trajectory_key], key=lambda index: int(rows[index]["step"])
        )
        if not indices:
            continue
        xy = coords[indices]
        color = colors[trajectory_key]
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=color,
            linewidth=0.9,
            alpha=0.72,
            label=labels[trajectory_key],
        )
        ax.scatter(xy[:, 0], xy[:, 1], color=color, s=18, alpha=0.75)
        ax.scatter(xy[0, 0], xy[0, 1], color=color, s=48, marker="o", edgecolor="black")
        ax.scatter(xy[-1, 0], xy[-1, 1], color=color, s=64, marker="x")
    ax.axhline(0.0, color="0.85", linewidth=0.8)
    ax.axvline(0.0, color="0.85", linewidth=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    legend_columns = min(4, max(1, int(np.ceil(len(records) / 18))))
    trajectory_legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=7,
        ncol=legend_columns,
        title="Trajectory",
    )
    ax.add_artist(trajectory_legend)
    ax.legend(
        handles=marker_legend_handles(),
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        title="Markers",
    )
    return fig, ax


def action_name(action_id: int) -> str:
    return ACTION_NAMES.get(int(action_id), f"ACTION_{action_id}")


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _graph_episode_ids(root: Path) -> set[str]:
    return {str(record["episode_id"]) for record in load_graph_manifest(root)}


def _selection_record_key(selection: object) -> tuple[str, str]:
    return str(selection.source_trajectory_id), str(selection.object_category)


def _episode_selection_key(episode: object) -> tuple[str, str]:
    from topovlm_data.habitat_objectnav import objectnav_source_trajectory_id

    return objectnav_source_trajectory_id(episode), str(getattr(episode, "object_category"))


def _filter_env_episodes_to_selection_keys(
    env: object, selection_by_key: dict[tuple[str, str], object]
) -> None:
    selected_episodes = [
        episode for episode in env.episodes if _episode_selection_key(episode) in selection_by_key
    ]
    selected_episodes.sort(
        key=lambda episode: (
            str(getattr(episode, "scene_id")),
            str(getattr(episode, "object_category")),
            _episode_selection_key(episode)[0],
        )
    )
    selected_keys = [_episode_selection_key(episode) for episode in selected_episodes]
    counts = Counter(selected_keys)
    duplicate_keys = [key for key, count in counts.items() if count > 1]
    if duplicate_keys:
        raise ValueError(f"Duplicate Habitat episodes after ObjectNav key filtering: {duplicate_keys[:3]}")
    missing_keys = set(selection_by_key).difference(counts)
    if missing_keys:
        raise ValueError(f"Selection keys absent from Habitat env: {sorted(missing_keys)[:3]}")
    env.episodes = selected_episodes


def _episode_payloads_by_selection_key(root: Path) -> dict[tuple[str, str], dict[str, object]]:
    payload_by_key = {}
    duplicate_keys = []
    for record in load_episode_manifest(root):
        key = (
            str(record["source_trajectory_id"]),
            str(record.get("object_category", record["goal_text"])),
        )
        if key in payload_by_key:
            duplicate_keys.append(key)
        payload_by_key[key] = record
    if duplicate_keys:
        raise ValueError(f"Duplicate materialized expert action payload keys: {duplicate_keys[:3]}")
    return payload_by_key


def _selection_to_plot_record(selection: object, actions: list[int]) -> dict[str, object]:
    action_array = np.asarray(actions, dtype=np.int64)
    return {
        "episode_id": f"{selection.source_trajectory_id}|{selection.object_category}",
        "scene_id": selection.scene_id,
        "goal_text": selection.object_category,
        "object_category": selection.object_category,
        "source_trajectory_id": selection.source_trajectory_id,
        "analysis_actions": action_array,
        "analysis_steps": int(len(action_array)),
        "analysis_turns": int(np.isin(action_array, [2, 3]).sum()),
        "analysis_first_turn": first_turn_label(action_array),
        "analysis_group": str(selection.object_category),
        "analysis_language_instruction": str(selection.object_category),
    }


def _map_heights_from_trajectory_heights(trajectory_heights: np.ndarray) -> np.ndarray:
    height_bins = np.round(trajectory_heights / MAP_HEIGHT_BIN_METERS)
    return np.asarray(
        [
            float(np.median(trajectory_heights[height_bins == height_bin]))
            for height_bin in np.unique(height_bins)
        ],
        dtype=np.float32,
    )


def _positions_to_grid(
    maps_module: object, env: object, topdown_map: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    grid_positions = np.asarray(
        [
            maps_module.to_grid(
                float(position[2]),
                float(position[0]),
                topdown_map.shape[:2],
                sim=env.sim,
            )
            for position in positions
        ],
        dtype=np.int64,
    )
    height, width = topdown_map.shape[:2]
    if (
        np.any(grid_positions[:, 0] < 0)
        or np.any(grid_positions[:, 0] >= height)
        or np.any(grid_positions[:, 1] < 0)
        or np.any(grid_positions[:, 1] >= width)
    ):
        raise ValueError("Habitat replay produced coordinates outside the top-down map.")
    return grid_positions


def _adjust_color_lightness(
    rgba: tuple[float, float, float, float], lightness: float, alpha: float
) -> tuple[float, float, float, float]:
    hue, _, saturation = colorsys.rgb_to_hls(float(rgba[0]), float(rgba[1]), float(rgba[2]))
    red, green, blue = colorsys.hls_to_rgb(hue, float(lightness), saturation)
    return red, green, blue, float(alpha)


def _annotate_goal_positions(
    ax: plt.Axes, replay: dict[str, object], records: list[dict[str, object]]
) -> None:
    base_colors = group_base_colors(records)
    goals_by_group: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for trajectory in replay["trajectories"]:
        record = trajectory["record"]
        if "goal_grid_position" not in trajectory:
            raise ValueError("Top-down replay is missing actual Habitat goal_grid_position.")
        goal = np.asarray(trajectory["goal_grid_position"], dtype=np.int64)
        goals_by_group[trajectory_group(record)].append((int(goal[0]), int(goal[1])))
    for index, group in enumerate(sorted(goals_by_group)):
        color = base_colors[group]
        goal_counts = Counter(goals_by_group[group])
        goals = np.asarray(list(goal_counts), dtype=np.float32)
        rows = goals[:, 0]
        cols = goals[:, 1]
        counts = np.asarray([goal_counts[tuple(goal)] for goal in goals.astype(np.int64)])
        map_height, map_width = np.asarray(replay["topdown_map"]).shape[:2]
        ax.scatter(
            cols,
            rows,
            color=color,
            marker="*",
            s=55 + 12 * np.minimum(counts, 8),
            edgecolor="black",
            linewidth=0.55,
            alpha=0.88,
            zorder=5,
        )
        duplicate_labels = 0
        for row, col, count in zip(rows, cols, counts):
            if count <= 1 or duplicate_labels >= 12:
                continue
            ax.annotate(
                f"x{int(count)}",
                xy=(float(col), float(row)),
                xytext=(5, -5),
                textcoords="offset points",
                fontsize=6.5,
                color="black",
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "alpha": 0.72},
                zorder=6,
            )
            duplicate_labels += 1
        label_row, label_col = np.median(goals, axis=0)
        x_offset = -12 if label_col > 0.72 * map_width else 12
        y_offset = 18
        if label_row < 0.18 * map_height:
            y_offset = 28
        elif label_row > 0.82 * map_height:
            y_offset = -22
        ax.annotate(
            f"{group}\nn={len(goals_by_group[group])}, visible={len(goal_counts)}",
            xy=(float(label_col), float(label_row)),
            xytext=(x_offset, y_offset + 12 * (index % 2)),
            textcoords="offset points",
            fontsize=8,
            color="black",
            ha="right" if x_offset < 0 else "left",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88},
            arrowprops={"arrowstyle": "-", "color": color, "linewidth": 1.1},
        )


def _unique_goal_position_count(replay: dict[str, object]) -> int:
    goals = set()
    for trajectory in replay["trajectories"]:
        if "goal_grid_position" not in trajectory:
            raise ValueError("Top-down replay is missing actual Habitat goal_grid_position.")
        goal = np.asarray(trajectory["goal_grid_position"], dtype=np.int64)
        goals.add((int(goal[0]), int(goal[1])))
    return len(goals)


def _unique_stop_position_count(replay: dict[str, object]) -> int:
    stops = set()
    for trajectory in replay["trajectories"]:
        grid = np.asarray(trajectory["grid_positions"], dtype=np.int64)
        stop = grid[-1]
        stops.add((int(stop[0]), int(stop[1])))
    return len(stops)


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
        "trajectory_key": trajectory_record_key(record),
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
