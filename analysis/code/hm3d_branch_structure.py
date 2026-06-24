"""Analyze action-only branch structure in HM3D ObjectNav expert trajectories."""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from configs.schema import TopoVLMConfig
from topovlm_data.habitat_manifest import (
    HabitatEpisodeRecord,
    load_episode_records,
    resolve_data_path,
    resolve_materialization_data_root,
)
from topovlm_data.habitat_objectnav import load_objectnav_selection_records
from utils.checkpoint_io import resolve_source_commit


ANALYSIS_NAME = "hm3d_branch_structure"
ACTION_NAMES = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
}
TURN_ACTIONS = {2, 3}
TURN_ANGLE_DEGREES = 30
REVERSAL_TURN_RUN_MIN_ACTIONS = 180 // TURN_ANGLE_DEGREES


def run_hm3d_branch_structure_analysis(cfg: TopoVLMConfig) -> dict[str, object]:
    data_root, all_records, records, missing_selected_ids = _load_analysis_records(cfg)
    action_counts: Counter[int] = Counter()
    transition_counts: Counter[str] = Counter()
    first_turn_counts: Counter[str] = Counter()
    scene_object_first_turns: dict[tuple[str, str], set[str]] = defaultdict(set)
    scene_object_episode_counts: Counter[tuple[str, str]] = Counter()
    object_first_turn_counts: dict[str, Counter[str]] = defaultdict(Counter)
    object_episode_counts: Counter[str] = Counter()
    length_bin_counts: Counter[str] = Counter()
    length_bin_turn_run_starts: Counter[str] = Counter()
    length_bin_turn_steps: Counter[str] = Counter()
    length_bin_steps: Counter[str] = Counter()
    episode_lengths = []
    turn_step_count = 0
    turn_run_start_count = 0
    inflection_count = 0
    terminal_stop_count = 0
    immediate_stop_count = 0
    leading_initial_turns = []
    leading_stem_forward_lengths = []
    first_after_stem_counts: Counter[str] = Counter()
    forward_run_lengths = []
    turn_run_lengths = []
    reversal_like_turn_run_count = 0
    episodes_with_reversal_like_turn = 0
    action_mode_sequences = []

    for record in records:
        actions = _load_actions(data_root, record)
        runs = _action_runs(actions)
        action_mode_sequences.append([action for action, _ in runs])
        leading_turns, leading_forward, first_after_stem = _leading_policy_stem(actions)
        leading_initial_turns.append(leading_turns)
        leading_stem_forward_lengths.append(leading_forward)
        first_after_stem_counts[ACTION_NAMES.get(first_after_stem, "END")] += 1
        episode_has_reversal_like_turn = False
        for action, run_length in runs:
            if action == 1:
                forward_run_lengths.append(run_length)
            if action in TURN_ACTIONS:
                turn_run_lengths.append(run_length)
                if run_length >= REVERSAL_TURN_RUN_MIN_ACTIONS:
                    reversal_like_turn_run_count += 1
                    episode_has_reversal_like_turn = True
        if episode_has_reversal_like_turn:
            episodes_with_reversal_like_turn += 1
        episode_turn_steps = sum(1 for action in actions if action in TURN_ACTIONS)
        episode_turn_run_starts = _count_turn_run_starts(actions)
        episode_lengths.append(len(actions))
        action_counts.update(actions)
        turn_step_count += episode_turn_steps
        turn_run_start_count += episode_turn_run_starts
        inflection_count += _count_inflections(actions)
        if actions and actions[-1] == 0:
            terminal_stop_count += 1
        if actions == [0]:
            immediate_stop_count += 1
        for previous, current in zip(actions, actions[1:]):
            transition_counts[f"{ACTION_NAMES[previous]}->{ACTION_NAMES[current]}"] += 1
        first_turn = _first_turn_label(actions)
        first_turn_counts[first_turn] += 1
        group_key = (record.scene_id, record.object_category or record.goal_text)
        scene_object_first_turns[group_key].add(first_turn)
        scene_object_episode_counts[group_key] += 1
        object_key = record.object_category or record.goal_text
        object_first_turn_counts[object_key][first_turn] += 1
        object_episode_counts[object_key] += 1
        length_bin = _length_bin(len(actions))
        length_bin_counts[length_bin] += 1
        length_bin_turn_run_starts[length_bin] += episode_turn_run_starts
        length_bin_turn_steps[length_bin] += episode_turn_steps
        length_bin_steps[length_bin] += len(actions)

    total_steps = sum(episode_lengths)
    action_count_map = _named_action_counts(action_counts)
    result = {
        "status": "ok",
        "analysis_name": ANALYSIS_NAME,
        "scope": _scope_from_config(cfg),
        "config_name": cfg.config_name,
        "data": {
            "dataset_name": cfg.data.dataset_name,
            "trajectory_source": cfg.data.trajectory_source,
            "cache_format": cfg.data.cache_format,
            "episodes_manifest": cfg.data.episodes_manifest,
            "episode_selection_manifest": cfg.data.episode_selection_manifest,
            "total_manifest_records": len(all_records),
            "analyzed_records": len(records),
            "missing_selected_record_count": len(missing_selected_ids),
        },
        "operational_definition": {
            "branch_candidate": (
                "action-only decision-sensitive trajectory position where the expert leaves "
                "MOVE_FORWARD into TURN_LEFT or TURN_RIGHT, or where the local action changes"
            ),
            "not_claimed": (
                "spatial junction localization, counterfactual branch availability, and VLM/policy "
                "latent topology are not identified from this action-only manifest"
            ),
        },
        "metrics": {
            "episodes": len(records),
            "steps": total_steps,
            "episode_length": _length_summary(episode_lengths),
            "action_counts": action_count_map,
            "action_fractions": _fractions(action_count_map, total_steps),
            "action_entropy_bits": _entropy_bits(action_counts),
            "turn_structure": {
                "turn_steps": turn_step_count,
                "turn_step_fraction": _safe_fraction(turn_step_count, total_steps),
                "turn_run_starts": turn_run_start_count,
                "turn_run_starts_per_episode": _safe_fraction(turn_run_start_count, len(records)),
                "inflections": inflection_count,
                "inflections_per_episode": _safe_fraction(inflection_count, len(records)),
                "terminal_stop_episodes": terminal_stop_count,
                "immediate_stop_episodes": immediate_stop_count,
            },
            "first_turn": dict(sorted(first_turn_counts.items())),
            "top_action_transitions": _top_counts(transition_counts, limit=12),
            "scene_object_trajectory_diversity": _scene_object_diversity(
                scene_object_episode_counts, scene_object_first_turns
            ),
            "object_category_first_turn": _object_category_first_turn_summary(
                object_episode_counts, object_first_turn_counts
            ),
            "length_bin_turn_pressure": _length_bin_turn_pressure(
                length_bin_counts,
                length_bin_turn_run_starts,
                length_bin_turn_steps,
                length_bin_steps,
            ),
            "policy_topology": {
                "action_mode_prefix_tree": _action_mode_prefix_tree_summary(
                    action_mode_sequences
                ),
                "leading_initial_turns": _length_summary(leading_initial_turns),
                "leading_stem_forward": _length_summary(leading_stem_forward_lengths),
                "first_after_stem": dict(sorted(first_after_stem_counts.items())),
                "forward_run_length": _length_summary(forward_run_lengths),
                "turn_run_length": _length_summary(turn_run_lengths),
                "reversal_like_turn_runs": {
                    "turn_angle_degrees": TURN_ANGLE_DEGREES,
                    "min_turn_actions": REVERSAL_TURN_RUN_MIN_ACTIONS,
                    "runs": reversal_like_turn_run_count,
                    "episodes": episodes_with_reversal_like_turn,
                    "episode_fraction": _safe_fraction(
                        episodes_with_reversal_like_turn, len(records)
                    ),
                },
            },
        },
        "evidence_axes": _evidence_axes(
            turn_run_start_count=turn_run_start_count,
            total_steps=total_steps,
            episodes=len(records),
            first_turn_counts=first_turn_counts,
            scene_object_episode_counts=scene_object_episode_counts,
            scene_object_first_turns=scene_object_first_turns,
            object_episode_counts=object_episode_counts,
            object_first_turn_counts=object_first_turn_counts,
        ),
        "topology_role_evidence": _topology_role_evidence(
            episodes=len(records),
            turn_run_start_count=turn_run_start_count,
            total_steps=total_steps,
            leading_stem_forward_lengths=leading_stem_forward_lengths,
            first_after_stem_counts=first_after_stem_counts,
            forward_run_lengths=forward_run_lengths,
            terminal_stop_count=terminal_stop_count,
            immediate_stop_count=immediate_stop_count,
            episodes_with_reversal_like_turn=episodes_with_reversal_like_turn,
            action_mode_sequences=action_mode_sequences,
            graph_manifest_available=resolve_data_path(data_root, cfg.data.graph_manifest).exists(),
        ),
        "interpretation_limits": [
            "Action-only shortest-path data can rank branch-candidate pressure but cannot prove true topological branch points.",
            "Scene/object action diversity is trajectory-family evidence, not same-state counterfactual evidence.",
            "Latent vector and learned-policy topology require cached VLM representations or policy logits in a later analysis gate.",
        ],
        "next_gates": [
            "Add pose/navmesh or graph metadata before claiming spatial junction localization.",
            "Probe cached VLM tokens only after dataset-level branch pressure is summarized.",
            "Compare policy errors at turn-run starts versus forward-stable segments after a trained policy checkpoint is selected.",
        ],
    }
    figure_paths = _write_figures(result)
    if figure_paths:
        result["figures"] = [
            {"path": str(figure_paths[0]), "role": "branch_structure_summary"},
            {"path": str(figure_paths[1]), "role": "policy_topology_summary"},
        ]
    result_manifest = _write_result_manifest(cfg, result)
    if result_manifest is not None:
        result["result_manifest"] = str(result_manifest)
    return result


def _load_analysis_records(
    cfg: TopoVLMConfig,
) -> tuple[Path, list[HabitatEpisodeRecord], list[HabitatEpisodeRecord], list[str]]:
    data_root = resolve_materialization_data_root(cfg.data.data_root)
    episode_manifest = resolve_data_path(data_root, cfg.data.episodes_manifest)
    all_records = load_episode_records(episode_manifest)
    if cfg.data.episode_selection_manifest is None:
        records = all_records
        missing_selected_ids: list[str] = []
    else:
        records_by_source_id = {
            record.source_trajectory_id: record
            for record in all_records
            if record.source_trajectory_id is not None
        }
        selected_ids = [
            record.source_trajectory_id for record in load_objectnav_selection_records(cfg.data)
        ]
        missing_selected_ids = [
            source_id for source_id in selected_ids if source_id not in records_by_source_id
        ]
        records = [
            records_by_source_id[source_id]
            for source_id in selected_ids
            if source_id in records_by_source_id
        ]
    if cfg.data.max_episodes is not None:
        records = records[: cfg.data.max_episodes]
    return data_root, all_records, records, missing_selected_ids


def _load_actions(data_root: Path, record: HabitatEpisodeRecord) -> list[int]:
    actions_path = resolve_data_path(data_root, record.actions_path)
    actions = np.load(actions_path, allow_pickle=False)
    if actions.ndim != 1:
        raise ValueError(f"Expected 1D action array: {actions_path}")
    return [int(action) for action in actions.tolist()]


def _count_turn_run_starts(actions: list[int]) -> int:
    count = 0
    for index, action in enumerate(actions):
        if action in TURN_ACTIONS and (index == 0 or actions[index - 1] != action):
            count += 1
    return count


def _count_inflections(actions: list[int]) -> int:
    return sum(1 for previous, current in zip(actions, actions[1:]) if previous != current)


def _action_runs(actions: list[int]) -> list[tuple[int, int]]:
    runs = []
    for action in actions:
        if runs and runs[-1][0] == action:
            runs[-1] = (action, runs[-1][1] + 1)
        else:
            runs.append((action, 1))
    return runs


def _leading_policy_stem(actions: list[int]) -> tuple[int, int, int | None]:
    index = 0
    while index < len(actions) and actions[index] in TURN_ACTIONS:
        index += 1
    leading_turns = index
    while index < len(actions) and actions[index] == 1:
        index += 1
    leading_forward = index - leading_turns
    first_after_stem = actions[index] if index < len(actions) else None
    return leading_turns, leading_forward, first_after_stem


def _first_turn_label(actions: list[int]) -> str:
    for action in actions:
        if action in TURN_ACTIONS:
            return ACTION_NAMES[action]
    return "NO_TURN"


def _length_bin(length: int) -> str:
    if length <= 1:
        return "1"
    if length <= 32:
        return "2-32"
    if length <= 64:
        return "33-64"
    if length <= 128:
        return "65-128"
    return "129+"


def _named_action_counts(counts: Counter[int]) -> dict[str, int]:
    return {ACTION_NAMES[action]: int(counts.get(action, 0)) for action in sorted(ACTION_NAMES)}


def _fractions(counts: dict[str, int], total: int) -> dict[str, float]:
    return {name: _safe_fraction(count, total) for name, count in counts.items()}


def _safe_fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _entropy_bits(counts: Counter[int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def _length_summary(lengths: list[int]) -> dict[str, float | int]:
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "p90": 0.0}
    values = np.asarray(lengths, dtype=np.float64)
    return {
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def _action_mode_prefix_tree_summary(
    action_mode_sequences: list[list[int]],
) -> dict[str, object]:
    root = {"count": 0, "terminal": 0, "children": {}}
    for sequence in action_mode_sequences:
        node = root
        node["count"] += 1
        for action in sequence:
            children = node["children"]
            if action not in children:
                children[action] = {"count": 0, "terminal": 0, "children": {}}
            node = children[action]
            node["count"] += 1
        node["terminal"] += 1

    branch_nodes = []
    deterministic_nodes = []
    terminal_nodes = []
    _collect_prefix_tree_nodes(root, (), branch_nodes, deterministic_nodes, terminal_nodes)
    return {
        "episodes": len(action_mode_sequences),
        "branch_nodes": len(branch_nodes),
        "deterministic_continuation_nodes": len(deterministic_nodes),
        "terminal_nodes": len(terminal_nodes),
        "top_branch_points": _top_prefix_branch_points(branch_nodes, limit=12),
    }


def _collect_prefix_tree_nodes(
    node: dict[str, object],
    prefix: tuple[int, ...],
    branch_nodes: list[tuple[tuple[int, ...], dict[str, object]]],
    deterministic_nodes: list[tuple[tuple[int, ...], dict[str, object]]],
    terminal_nodes: list[tuple[tuple[int, ...], dict[str, object]]],
) -> None:
    children = node["children"]
    if len(children) >= 2:
        branch_nodes.append((prefix, node))
    if len(children) == 1 and not node["terminal"]:
        deterministic_nodes.append((prefix, node))
    if node["terminal"]:
        terminal_nodes.append((prefix, node))
    for action, child in children.items():
        _collect_prefix_tree_nodes(
            child, prefix + (int(action),), branch_nodes, deterministic_nodes, terminal_nodes
        )


def _top_prefix_branch_points(
    branch_nodes: list[tuple[tuple[int, ...], dict[str, object]]], *, limit: int
) -> list[dict[str, object]]:
    rows = []
    for prefix, node in sorted(branch_nodes, key=lambda item: (-int(item[1]["count"]), len(item[0])))[
        :limit
    ]:
        children = node["children"]
        rows.append(
            {
                "support": int(node["count"]),
                "depth": len(prefix),
                "prefix": [ACTION_NAMES[action] for action in prefix[-8:]],
                "children": {
                    ACTION_NAMES[action]: int(child["count"])
                    for action, child in sorted(
                        children.items(), key=lambda item: (-int(item[1]["count"]), item[0])
                    )
                },
            }
        )
    return rows


def _top_counts(counts: Counter[str], *, limit: int) -> list[dict[str, object]]:
    return [
        {"name": name, "count": int(count)}
        for name, count in counts.most_common(limit)
    ]


def _scene_object_diversity(
    episode_counts: Counter[tuple[str, str]], first_turns: dict[tuple[str, str], set[str]]
) -> dict[str, object]:
    groups_with_multiple_episodes = [key for key, count in episode_counts.items() if count >= 2]
    groups_with_first_turn_diversity = [
        key for key in groups_with_multiple_episodes if len(first_turns[key]) >= 2
    ]
    return {
        "scene_object_groups": len(episode_counts),
        "groups_with_multiple_episodes": len(groups_with_multiple_episodes),
        "groups_with_first_turn_diversity": len(groups_with_first_turn_diversity),
        "first_turn_diversity_fraction": _safe_fraction(
            len(groups_with_first_turn_diversity), len(groups_with_multiple_episodes)
        ),
    }


def _object_category_first_turn_summary(
    episode_counts: Counter[str], first_turn_counts: dict[str, Counter[str]]
) -> list[dict[str, object]]:
    rows = []
    for object_category, episode_count in episode_counts.items():
        counts = first_turn_counts[object_category]
        rows.append(
            {
                "object_category": object_category,
                "episodes": int(episode_count),
                "first_turn_counts": dict(sorted(counts.items())),
                "first_turn_entropy_bits": _entropy_bits(
                    Counter({index: count for index, count in enumerate(counts.values())})
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -float(row["first_turn_entropy_bits"]),
            -int(row["episodes"]),
            str(row["object_category"]),
        ),
    )[:12]


def _length_bin_turn_pressure(
    bin_counts: Counter[str],
    bin_turn_run_starts: Counter[str],
    bin_turn_steps: Counter[str],
    bin_steps: Counter[str],
) -> list[dict[str, object]]:
    rows = []
    for label in ["1", "2-32", "33-64", "65-128", "129+"]:
        episodes = bin_counts[label]
        if episodes == 0:
            continue
        rows.append(
            {
                "length_bin": label,
                "episodes": int(episodes),
                "turn_run_starts_per_episode": _safe_fraction(
                    bin_turn_run_starts[label], episodes
                ),
                "turn_step_fraction": _safe_fraction(bin_turn_steps[label], bin_steps[label]),
            }
        )
    return rows


def _evidence_axes(
    *,
    turn_run_start_count: int,
    total_steps: int,
    episodes: int,
    first_turn_counts: Counter[str],
    scene_object_episode_counts: Counter[tuple[str, str]],
    scene_object_first_turns: dict[tuple[str, str], set[str]],
    object_episode_counts: Counter[str],
    object_first_turn_counts: dict[str, Counter[str]],
) -> list[dict[str, object]]:
    first_turn_turn_episodes = (
        first_turn_counts.get("TURN_LEFT", 0) + first_turn_counts.get("TURN_RIGHT", 0)
    )
    multi_scene_object_groups = [
        key for key, count in scene_object_episode_counts.items() if count >= 2
    ]
    diverse_scene_object_groups = [
        key for key in multi_scene_object_groups if len(scene_object_first_turns[key]) >= 2
    ]
    diverse_object_categories = [
        category
        for category, count in object_episode_counts.items()
        if count >= 10 and len(object_first_turn_counts[category]) >= 2
    ]
    return [
        {
            "axis": "turn-run pressure",
            "evidence_status": "supports",
            "readout": {
                "turn_run_starts_per_episode": _safe_fraction(turn_run_start_count, episodes),
                "turn_run_starts_per_100_steps": 100.0
                * _safe_fraction(turn_run_start_count, total_steps),
            },
            "interpretation": "expert trajectories contain frequent transition points where local policy leaves forward-stable behavior",
        },
        {
            "axis": "first-turn split",
            "evidence_status": "supports",
            "readout": {
                "episodes_with_left_or_right_first_turn_fraction": _safe_fraction(
                    first_turn_turn_episodes, episodes
                ),
                "left_first_turn_episodes": int(first_turn_counts.get("TURN_LEFT", 0)),
                "right_first_turn_episodes": int(first_turn_counts.get("TURN_RIGHT", 0)),
            },
            "interpretation": "many episodes require an early left/right branch choice rather than only forward execution",
        },
        {
            "axis": "same scene-object trajectory diversity",
            "evidence_status": "mixed",
            "readout": {
                "diverse_groups": len(diverse_scene_object_groups),
                "multi_episode_groups": len(multi_scene_object_groups),
                "fraction": _safe_fraction(
                    len(diverse_scene_object_groups), len(multi_scene_object_groups)
                ),
            },
            "interpretation": "same scene/object families often contain different first-turn choices, but this is not same-state counterfactual evidence",
        },
        {
            "axis": "object-category branch diversity",
            "evidence_status": "mixed",
            "readout": {
                "diverse_object_categories_with_10plus_episodes": len(diverse_object_categories),
                "object_categories": len(object_episode_counts),
            },
            "interpretation": "object category conditions branch distributions, but category alone is not a topology label",
        },
        {
            "axis": "latent or policy representation separability",
            "evidence_status": "insufficient",
            "readout": {
                "graph_cache_available": False,
                "policy_logits_available": False,
            },
            "interpretation": "requires VLM/topology latent cache or trained policy logits; action-only expert manifest cannot answer this axis",
        },
    ]


def _topology_role_evidence(
    *,
    episodes: int,
    turn_run_start_count: int,
    total_steps: int,
    leading_stem_forward_lengths: list[int],
    first_after_stem_counts: Counter[str],
    forward_run_lengths: list[int],
    terminal_stop_count: int,
    immediate_stop_count: int,
    episodes_with_reversal_like_turn: int,
    action_mode_sequences: list[list[int]],
    graph_manifest_available: bool,
) -> list[dict[str, object]]:
    prefix_tree = _action_mode_prefix_tree_summary(action_mode_sequences)
    left_right_after_stem = first_after_stem_counts.get("TURN_LEFT", 0) + first_after_stem_counts.get(
        "TURN_RIGHT", 0
    )
    leading_stem = _length_summary(leading_stem_forward_lengths)
    forward_runs = _length_summary(forward_run_lengths)
    return [
        {
            "role": "junction",
            "evidence_status": "partial",
            "readout": {
                "prefix_tree_branch_nodes": prefix_tree["branch_nodes"],
                "turn_run_starts_per_100_steps": 100.0
                * _safe_fraction(turn_run_start_count, total_steps),
                "left_or_right_after_stem_fraction": _safe_fraction(
                    left_right_after_stem, episodes
                ),
            },
            "interpretation": "action-sequence branches are visible, but spatial junction localization needs pose or navmesh path evidence",
        },
        {
            "role": "edge",
            "evidence_status": "visible",
            "readout": {
                "forward_run_median": forward_runs["median"],
                "forward_run_p90": forward_runs["p90"],
                "forward_run_max": forward_runs["max"],
                "deterministic_prefix_nodes": prefix_tree["deterministic_continuation_nodes"],
            },
            "interpretation": "forward-stable deterministic action segments are directly visible in expert trajectories",
        },
        {
            "role": "stem",
            "evidence_status": "partial",
            "readout": {
                "leading_forward_median": leading_stem["median"],
                "leading_forward_p90": leading_stem["p90"],
                "leading_forward_max": leading_stem["max"],
                "first_after_stem": dict(sorted(first_after_stem_counts.items())),
            },
            "interpretation": "T-maze-like approach-to-choice segments are visible as leading forward runs followed by left/right choice, but they are short and not yet tied to shared spatial corridors",
        },
        {
            "role": "dead-end",
            "evidence_status": "partial",
            "readout": {
                "terminal_stop_episodes": terminal_stop_count,
                "immediate_stop_episodes": immediate_stop_count,
                "reversal_like_turn_episodes": episodes_with_reversal_like_turn,
                "reversal_like_turn_episode_fraction": _safe_fraction(
                    episodes_with_reversal_like_turn, episodes
                ),
            },
            "interpretation": "terminal leaves and reversal-like turn runs are visible, but spatial dead-end structure is not proven without pose or navmesh geometry",
        },
        {
            "role": "learned policy topology",
            "evidence_status": "insufficient",
            "readout": {
                "graph_manifest_available": graph_manifest_available,
                "policy_logits_available": False,
            },
            "interpretation": "latent/policy topology requires graph cache or trained-policy logits; the current canonical data lane exposes action/RGB episodes only",
        },
    ]


def _scope_from_config(cfg: TopoVLMConfig) -> str:
    return cfg.config_name.replace("/", "_")


def _write_result_manifest(cfg: TopoVLMConfig, result: dict[str, object]) -> Path | None:
    artifact_dir = os.environ.get("ARTIFACT_DIR")
    if artifact_dir is None:
        return None
    manifest_path = Path(artifact_dir) / "result_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "artifact_type": "topovlm_analysis_result_manifest",
        "schema_version": 1,
        "status": result["status"],
        "analysis_name": result["analysis_name"],
        "scope": result["scope"],
        "config_name": cfg.config_name,
        "run_name": cfg.run_name,
        "seed": cfg.seed,
        "source_commit": resolve_source_commit(),
        "data": result["data"],
        "operational_definition": result["operational_definition"],
        "metrics": result["metrics"],
        "evidence_axes": result["evidence_axes"],
        "topology_role_evidence": result["topology_role_evidence"],
        "figures": result.get("figures", []),
        "interpretation_limits": result["interpretation_limits"],
        "next_gates": result["next_gates"],
        "durable_lane_roots": {
            "artifact_dir": str(Path(artifact_dir)),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def _write_figures(result: dict[str, object]) -> list[Path]:
    artifact_dir = os.environ.get("ARTIFACT_DIR")
    if artifact_dir is None:
        return []
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    metrics = result["metrics"]
    figure_path = Path(artifact_dir) / "branch_structure_summary.png"
    topology_figure_path = Path(artifact_dir) / "policy_topology_summary.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), dpi=160)
    _plot_action_mix(axes[0, 0], metrics["action_fractions"])
    _plot_first_turns(axes[0, 1], metrics["first_turn"])
    _plot_episode_lengths(axes[1, 0], metrics["episode_length"])
    _plot_scene_object_diversity(axes[1, 1], metrics["scene_object_trajectory_diversity"])
    fig.suptitle("HM3D ObjectNav shortest-path branch-structure audit", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figure_path)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), dpi=160)
    topology_metrics = metrics["policy_topology"]
    _plot_stem_after_alignment(axes[0, 0], topology_metrics["first_after_stem"])
    _plot_forward_run_summary(axes[0, 1], topology_metrics["forward_run_length"])
    _plot_prefix_tree_roles(axes[1, 0], topology_metrics["action_mode_prefix_tree"])
    _plot_topology_role_evidence(axes[1, 1], result["topology_role_evidence"])
    fig.suptitle("HM3D policy-topology role proxy audit", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(topology_figure_path)
    plt.close(fig)
    return [figure_path, topology_figure_path]


def _plot_action_mix(ax, action_fractions: dict[str, float]) -> None:
    labels = list(action_fractions)
    values = [100.0 * action_fractions[label] for label in labels]
    colors = ["#5B6770", "#4C78A8", "#F58518", "#E45756"]
    ax.barh(labels, values, color=colors)
    ax.set_title("Action mix")
    ax.set_xlabel("steps (%)")
    ax.set_xlim(0, max(values) * 1.15 if values else 1.0)
    for index, value in enumerate(values):
        ax.text(value + 0.5, index, f"{value:.1f}%", va="center", fontsize=8)


def _plot_first_turns(ax, first_turn: dict[str, int]) -> None:
    labels = list(first_turn)
    values = [first_turn[label] for label in labels]
    colors = ["#72B7B2", "#F58518", "#E45756"]
    ax.bar(labels, values, color=colors[: len(labels)])
    ax.set_title("First turn direction")
    ax.set_ylabel("episodes")
    ax.tick_params(axis="x", rotation=20)
    for index, value in enumerate(values):
        ax.text(index, value, str(value), ha="center", va="bottom", fontsize=8)


def _plot_episode_lengths(ax, episode_length: dict[str, float | int]) -> None:
    labels = ["median", "mean", "p90", "max"]
    values = [float(episode_length[label]) for label in labels]
    ax.bar(labels, values, color="#54A24B")
    ax.set_title("Trajectory length")
    ax.set_ylabel("actions")
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.0f}", ha="center", va="bottom", fontsize=8)


def _plot_scene_object_diversity(ax, diversity: dict[str, object]) -> None:
    labels = ["multi-episode\nscene-object", "diverse first-turn\nscene-object"]
    values = [
        int(diversity["groups_with_multiple_episodes"]),
        int(diversity["groups_with_first_turn_diversity"]),
    ]
    ax.bar(labels, values, color=["#B279A2", "#FF9DA6"])
    ax.set_title("Scene-object trajectory diversity")
    ax.set_ylabel("groups")
    for index, value in enumerate(values):
        ax.text(index, value, str(value), ha="center", va="bottom", fontsize=8)
    fraction = 100.0 * float(diversity["first_turn_diversity_fraction"])
    ax.text(
        0.5,
        0.88,
        f"{fraction:.1f}% of multi-episode groups",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )


def _plot_stem_after_alignment(ax, first_after_stem: dict[str, int]) -> None:
    labels = list(first_after_stem)
    values = [first_after_stem[label] for label in labels]
    colors = ["#72B7B2", "#E45756", "#F58518", "#4C78A8"][: len(labels)]
    ax.bar(labels, values, color=colors)
    ax.set_title("First action after stem proxy")
    ax.set_ylabel("episodes")
    ax.tick_params(axis="x", rotation=20)
    for index, value in enumerate(values):
        ax.text(index, value, str(value), ha="center", va="bottom", fontsize=8)


def _plot_forward_run_summary(ax, summary: dict[str, float | int]) -> None:
    labels = ["median", "mean", "p90", "max"]
    values = [float(summary[label]) for label in labels]
    ax.bar(labels, values, color="#54A24B")
    ax.set_title("Edge proxy: forward-run length")
    ax.set_ylabel("MOVE_FORWARD actions")
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.0f}", ha="center", va="bottom", fontsize=8)


def _plot_prefix_tree_roles(ax, prefix_tree: dict[str, object]) -> None:
    labels = ["branch", "deterministic", "terminal"]
    values = [
        int(prefix_tree["branch_nodes"]),
        int(prefix_tree["deterministic_continuation_nodes"]),
        int(prefix_tree["terminal_nodes"]),
    ]
    ax.bar(labels, values, color=["#F58518", "#4C78A8", "#72B7B2"])
    ax.set_title("Action-mode prefix tree")
    ax.set_ylabel("nodes")
    ax.tick_params(axis="x", rotation=15)
    for index, value in enumerate(values):
        ax.text(index, value, str(value), ha="center", va="bottom", fontsize=8)


def _plot_topology_role_evidence(ax, role_evidence: list[dict[str, object]]) -> None:
    status_score = {"visible": 3, "partial": 2, "not visible": 1, "insufficient": 0}
    colors = {
        "visible": "#54A24B",
        "partial": "#F58518",
        "not visible": "#E45756",
        "insufficient": "#9D9DA1",
    }
    labels = [str(row["role"]) for row in role_evidence]
    statuses = [str(row["evidence_status"]) for row in role_evidence]
    values = [status_score[status] for status in statuses]
    ax.barh(labels, values, color=[colors[status] for status in statuses])
    ax.set_title("Topology role evidence")
    ax.set_xlim(0, 3.4)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["insufficient", "not visible", "partial", "visible"])
    for index, status in enumerate(statuses):
        ax.text(values[index] + 0.05, index, status, va="center", fontsize=8)
