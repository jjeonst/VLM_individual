"""Week 2 · Step 5-7 (positions) — Replay HM3D ObjectNav expert actions in the
Habitat simulator to recover per-step agent POSITIONS.

The PR2L cache stores only expert actions (0=STOP,1=FWD,2=LEFT,3=RIGHT). doorway,
bottleneck and wrong-turn/dead-end candidates are inherently spatial, so they
cannot be found from actions alone -- we need where the agent actually went.
This script replays each episode's action sequence headlessly (no RGB sensors)
and records the agent (x, y, z) at every step, grouped by scene.

Join: a PR2L record's ``source_trajectory_id`` = ``<scene>:<N>`` where N is the
original ObjectNav episode_id; the start pose comes from that episode.

Action params match the ObjectNav config (forward 0.25 m, turn 30 deg,
agent height 0.88, radius 0.18).

Outputs (survey/week2/results/objectnav_branching/positions/<split>/):
  - <scene_key>.npz     : arrays keyed by episode_id -> (T, 3) float32 positions
  - index.jsonl         : one line per replayed episode (ids, object, #points)

Heavy-ish (loads 124 scenes), so intended to run via slurm/objectnav_replay.slurm.

Usage:
  python -m survey.week2.objectnav_replay_positions --split train
  python -m survey.week2.objectnav_replay_positions --split train --limit-scenes 3
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("TOPOVLM_DATASET_ROOT", "/data/topovlm"))
HABITAT_ROOT = DATA_ROOT / "habitat"
SCENE_ROOT = HABITAT_ROOT / "scene_datasets" / "hm3d_v0.2"
SCENE_DATASET_CONFIG = SCENE_ROOT / "hm3d_annotated_basis.scene_dataset_config.json"
ORIG_OBJECTNAV = (HABITAT_ROOT / "datasets" / "objectnav" / "hm3d" / "v2"
                  / "objectnav_hm3d_v2")
RESULT_ROOT = REPO_ROOT / "survey" / "week2" / "results" / "objectnav_branching" / "positions"

FORWARD_AMOUNT_M = 0.25
TURN_ANGLE_DEG = 30.0
AGENT_HEIGHT = 0.88
AGENT_RADIUS = 0.18
ACTION_TO_NAME = {1: "move_forward", 2: "turn_left", 3: "turn_right"}


def scene_key(scene_id: str) -> str:
    return scene_id.split("/")[-2]  # e.g. "00016-qk9eeNeR4vw"


def scene_short_name(sc_key: str) -> str:
    return sc_key.split("-", 1)[-1]  # e.g. "qk9eeNeR4vw"


def load_records(split: str) -> list[dict[str, Any]]:
    mf = HABITAT_ROOT / "episodes" / "pr2l_hm3d_objectnav" / split / "manifest.jsonl"
    with mf.open() as handle:
        return [json.loads(line) for line in handle]


def load_actions(record: dict[str, Any]) -> np.ndarray | None:
    path = HABITAT_ROOT / record["actions_path"]
    return np.load(path) if path.exists() else None


def _make_sim(scene_glb: str):
    import habitat_sim

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene_glb
    backend.scene_dataset_config_file = str(SCENE_DATASET_CONFIG)
    backend.enable_physics = False

    agent = habitat_sim.AgentConfiguration()
    agent.height = AGENT_HEIGHT
    agent.radius = AGENT_RADIUS
    agent.sensor_specifications = []  # no RGB -> headless & fast
    agent.action_space = {
        "move_forward": habitat_sim.ActionSpec(
            "move_forward", habitat_sim.ActuationSpec(amount=FORWARD_AMOUNT_M)),
        "turn_left": habitat_sim.ActionSpec(
            "turn_left", habitat_sim.ActuationSpec(amount=TURN_ANGLE_DEG)),
        "turn_right": habitat_sim.ActionSpec(
            "turn_right", habitat_sim.ActuationSpec(amount=TURN_ANGLE_DEG)),
    }
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))


def replay_scene(sc_key: str, records: list[dict[str, Any]],
                 index_out: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    from habitat_sim.utils.common import quat_from_coeffs

    name = scene_short_name(sc_key)
    scene_glb = str(SCENE_ROOT / "train" / sc_key / f"{name}.basis.glb")
    content = ORIG_OBJECTNAV / "train" / "content" / f"{name}.json.gz"
    if not (Path(scene_glb).exists() and content.exists()):
        print(f"  [skip] {sc_key}: missing scene glb or content", flush=True)
        return {}

    with gzip.open(content, "rt") as handle:
        orig = {str(e["episode_id"]): e for e in json.load(handle)["episodes"]}

    sim = _make_sim(scene_glb)
    agent = sim.get_agent(0)
    out: dict[str, np.ndarray] = {}
    try:
        for rec in records:
            actions = load_actions(rec)
            if actions is None or len(actions) <= 1:
                continue  # degenerate (immediate STOP) -> no trajectory
            ep = orig.get(rec["source_trajectory_id"].split(":")[-1])
            if ep is None:
                continue
            state = agent.get_state()
            state.position = np.array(ep["start_position"], dtype=np.float32)
            state.rotation = quat_from_coeffs(np.array(ep["start_rotation"], dtype=np.float32))
            agent.set_state(state)

            positions = [agent.get_state().position.copy()]
            for act in actions:
                a = int(act)
                if a == 0:
                    break
                sim.step(ACTION_TO_NAME[a])
                positions.append(agent.get_state().position.copy())

            arr = np.asarray(positions, dtype=np.float32)
            out[rec["episode_id"]] = arr
            index_out.append({
                "episode_id": rec["episode_id"],
                "scene_key": sc_key,
                "object_category": rec.get("object_category"),
                "num_points": int(len(arr)),
                "num_actions": int(len(actions)),
            })
    finally:
        sim.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit-scenes", type=int, default=0,
                        help="If >0, only process this many scenes (for testing).")
    args = parser.parse_args()

    records = load_records(args.split)
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_scene[scene_key(rec["scene_id"])].append(rec)
    scenes = sorted(by_scene)
    if args.limit_scenes > 0:
        scenes = scenes[: args.limit_scenes]
    print(f"[replay] split={args.split}: {len(records)} records across {len(by_scene)} scenes; "
          f"processing {len(scenes)}", flush=True)

    out_dir = RESULT_ROOT / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    t0 = time.time()
    total_eps = 0
    for i, sc_key in enumerate(scenes, start=1):
        t = time.time()
        scene_out = replay_scene(sc_key, by_scene[sc_key], index)
        if scene_out:
            np.savez_compressed(out_dir / f"{sc_key}.npz", **scene_out)
            total_eps += len(scene_out)
        print(f"  [{i}/{len(scenes)}] {sc_key}: {len(scene_out)} episodes "
              f"in {time.time()-t:.1f}s (total {total_eps} eps)", flush=True)

    with (out_dir / "index.jsonl").open("w") as handle:
        for row in index:
            handle.write(json.dumps(row) + "\n")
    print(f"[replay] done: {total_eps} episodes, {len(scenes)} scenes, "
          f"{time.time()-t0:.1f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
