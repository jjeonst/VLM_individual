"""Experiment 1-A of Appendix W -- run the same policy inside habitat-lab's own environment.

Every success rate this project reports came out of `evaluate.py`, and almost all of that file is
ours: the rollout loop, the agent-state save and restore that lets one simulator serve many
episodes, the episode termination, the batching, the pose-to-sensor conversion. Two pieces were
checked against habitat-lab -- the success test and the episodic GPS/compass -- but a check of
two functions is not a check of the loop that calls them.

Here habitat-lab supplies the environment and we supply only the representation and the choice of
action:

    habitat-lab                          ours
    ------------------------------------ ---------------------------------
    episode iteration and reset          encode the frame
    rgb / gps / compass / objectgoal     run the policy
    stepping and termination             pick one action
    Success, SPL, DistanceToGoal

If the success rate agrees with our loop on the same episodes, the objection "the evaluation is
lenient" closes for every number in this project at once. If it disagrees, the size of the
disagreement is the bias of our loop, and the per-episode rows say where it lives.

**The image-encoder condition, not CoT.** Experiment 3-A showed that changing how frames are
batched changes the sampled text -- different kernels, different logits, a token near a
probability boundary falling the other way. habitat-lab hands over one episode at a time, so a
CoT comparison would batch differently from our loop by construction and the sampling difference
would sit inside the very number being compared. The image encoder generates nothing, so the two
loops can be compared without that confound. What 1-A tests -- rollout, termination, scoring --
is shared code, so clearing it here clears it for CoT too.

**Deviations from the stock benchmark config, and why each is deliberate.**

* `look_up` / `look_down` stay in the action space but are never emitted. Appendix C.1 gives the
  policy four actions; removing the other two from the config would be a change to the standard
  setup, and leaving them unused costs nothing.
* The depth sensor is switched off. `objectnav_hm3d.yaml` asks for an `rgbd_agent` and renders
  a 640x480 depth map every step that the policy never reads. Leaving it on was the first call
  here, on the reasoning that the stock file should be accepted as given -- but the paper
  excludes depth outright (Appendix C.1) and our own `make_sim` has no depth sensor, so keeping
  it made the run differ from *both* sides of the comparison. This experiment measures the
  difference between two loops; adding a rendering difference on top only muddies it. Success,
  SPL and DistanceToGoal come from the pose and the pathfinder, so nothing measured moves.
* Scene and episode paths are pointed at this cluster's copies. That is a path, not a setting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

# Four of these run side by side on one node, each asking slurm for eight CPUs. Nothing makes
# torch honour that -- left alone it sizes its pools from the machine and takes 16 intra-op and
# 32 inter-op threads, so four processes put 248 threads on 48 cores and spend their time taking
# the cores back off each other. The rollout is CPU-bound, so that is most of the cost.
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
torch.set_num_interop_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

from dataset import OBJECTS, OBJECT_INDEX
from evaluate import (CHECKPOINT_ROOT, HABITAT_ROOT,
                      load_episodes, stratified_subset)
from policy import NavigationPolicy
from vlm_features import encode_vision_batch, load_vision_backbone

BENCHMARK = "benchmark/nav/objectnav/objectnav_hm3d.yaml"
DATA_PATH = str(HABITAT_ROOT / "datasets/objectnav/hm3d/v1/objectnav_hm3d_v1/{split}/{split}.json.gz")
SCENE_DATASET = str(HABITAT_ROOT / "scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json")

# `evaluate.py` orders its head STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT. habitat-lab addresses
# actions by name, so the mapping is written out rather than assumed to line up with the order
# the actions happen to appear in the config.
ACTION_NAME = {0: "stop", 1: "move_forward", 2: "turn_left", 3: "turn_right"}


def pose_key(scene_id: str, start_position, start_rotation) -> str:
    """A name for an episode that both loaders agree on.

    `episode_uid` cannot be used to join the two sides. It ends in a digest of the start pose,
    which matches exactly, but it also carries `episode_id` -- and habitat-lab renumbers that
    field sequentially as it reads a scene file, while our loader keeps whatever the JSON held.
    The raw field was already known to be useless as an identifier (the 2,000 validation episodes
    share three values between them), which is why the digest exists; leaving the useless half
    inside the name is what made the join fail, 495 episodes matching none.

    The digest alone, with the building, is what actually identifies an episode.
    """
    payload = (np.asarray(start_position, dtype=np.float32).tobytes()
               + np.asarray(start_rotation, dtype=np.float32).tobytes())
    scene = Path(scene_id).parent.name or Path(scene_id).stem
    return f"{scene}-{hashlib.md5(payload).hexdigest()[:8]}"


def pose_key_of(episode) -> str:
    """`pose_key` for a habitat-lab episode object."""
    return pose_key(episode.scene_id, episode.start_position, episode.start_rotation)


def build_env(scenes: list[str], split: str, wanted: set[str]):
    """habitat-lab's ObjectNav environment, stock apart from where the files live.

    The dataset is built first and filtered down to the episodes being compared, then handed to
    `Env`. Letting `Env` load everything and skipping past the unwanted ones would work too, but
    only by reaching into its private episode cursor -- and a reset it does not expect still
    loads a scene.
    """
    import habitat
    from habitat.config import read_write
    from habitat.config.default import get_config
    from habitat.datasets import make_dataset

    # The paths are written into the config object rather than passed as hydra overrides. The
    # dataset path carries a literal `{split}` placeholder, and hydra's override grammar reads
    # the brace as syntax -- the first attempt died on `mismatched input '{'` before the model
    # had even loaded. Editing the resolved config leaves the benchmark file itself untouched,
    # which is the property this experiment needs.
    config = get_config(BENCHMARK)
    with read_write(config):
        config.habitat.dataset.split = split
        config.habitat.dataset.data_path = DATA_PATH
        config.habitat.dataset.scenes_dir = str(HABITAT_ROOT / "scene_datasets")
        config.habitat.simulator.scene_dataset = SCENE_DATASET
        if scenes:
            config.habitat.dataset.content_scenes = list(scenes)
        sensors = config.habitat.simulator.agents.main_agent.sim_sensors
        if "depth_sensor" in sensors:
            del sensors["depth_sensor"]

    dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    before = len(dataset.episodes)
    dataset.episodes = [e for e in dataset.episodes if pose_key_of(e) in wanted]
    print(f"[1-A] habitat-lab 데이터셋 {before} -> {len(dataset.episodes)} 에피소드로 제한",
          flush=True)
    if len(dataset.episodes) != len(wanted):
        missing = wanted - {pose_key_of(e) for e in dataset.episodes}
        raise SystemExit(f"[1-A] {len(missing)}개를 habitat-lab 쪽에서 찾지 못했다: "
                         f"{sorted(missing)[:3]}")
    return habitat.Env(config=config, dataset=dataset)


def goal_vector(category: str, device) -> torch.Tensor:
    """One-hot over this project's object order.

    habitat's `objectgoal` sensor reports an index into the task's own category list, which is
    not this project's list and would silently mean a different object. The episode carries the
    name, so the name is what gets mapped.
    """
    goal = torch.zeros(1, 1, len(OBJECTS), device=device)
    goal[0, 0, OBJECT_INDEX[category]] = 1.0
    return goal


@torch.inference_mode()
def run_episode(env, vlm, policy, device) -> dict:
    """One episode, stepped by habitat-lab, acted by our policy."""
    observations = env.reset()
    episode = env.current_episode
    memory = None
    previous = torch.zeros(1, 1, 4, device=device)
    goal = goal_vector(episode.object_category, device)
    steps = 0

    while not env.episode_over:
        frame = np.asarray(observations["rgb"])[..., :3][None]
        # The image encoder's tokens are already [16, 2176] -- no PCA, no layer stacking, both
        # being PR2L-specific (Appendix C.2, PR2L items 2 and 3). `evaluate.py` hands the array
        # straight through here too.
        reduced = encode_vision_batch(vlm, frame)[0].tokens

        tokens = torch.zeros(1, 1, reduced.shape[0], policy.summary.token_dim, device=device)
        tokens[0, 0, :reduced.shape[0]] = torch.from_numpy(reduced).to(device)
        padding = torch.zeros(1, 1, reduced.shape[0], dtype=torch.bool, device=device)

        # habitat's own episodic sensors, not our reimplementation of them. That substitution is
        # part of what this experiment is for.
        gps = torch.as_tensor(observations["gps"], dtype=torch.float32,
                              device=device).reshape(1, 1, 2)
        compass = torch.as_tensor(observations["compass"], dtype=torch.float32,
                                  device=device).reshape(1, 1)

        logits, memory = policy(tokens, padding, gps, compass, previous, goal, memory)
        action = int(logits[0, 0].argmax())

        previous = torch.zeros(1, 1, 4, device=device)
        previous[0, 0, action] = 1.0
        observations = env.step(ACTION_NAME[action])
        steps += 1

    metrics = env.get_metrics()
    return {"pose_key": pose_key_of(episode),
            "object_category": episode.object_category,
            "scene_id": episode.scene_id,
            "start_geodesic": float(episode.info.get("geodesic_distance", float("nan"))),
            "steps": steps,
            "success": bool(metrics["success"]),
            "spl": round(float(metrics["spl"]), 4),
            "final_geodesic": round(float(metrics["distance_to_goal"]), 4)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="stageB_image")
    parser.add_argument("--split", default="val")
    parser.add_argument("--episodes", type=int, default=None,
                        help="how many of the split to run; leave unset for all of it")
    parser.add_argument("--subset", type=int, default=None,
                        help="draw from this subset of the split rather than the whole; unset "
                             "means the whole split, which is what the paper reports on")
    # Scenes are the seam, as in `evaluate.py`: no episode spans two of them and the selection
    # happens before the split, so concatenating the shards reproduces the single-process result
    # exactly rather than approximately.
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--compare", type=Path,
                        default=CHECKPOINT_ROOT / "stageB_image_full_eval_val.json")
    parser.add_argument("--out", type=Path,
                        default=CHECKPOINT_ROOT / "validate_1a.json")
    args = parser.parse_args()

    # Draw the episodes from the subset our own loop already ran, so every row here has a
    # counterpart to be compared against rather than a number standing on its own.
    pool = load_episodes(args.split)
    ours = stratified_subset(pool, args.subset) if args.subset else pool
    wanted = (stratified_subset(ours, args.episodes)
              if args.episodes and args.episodes < len(ours) else ours)
    if args.shard is not None:
        chosen = sorted({e["scene_id"] for e in wanted})[args.shard::args.shards]
        wanted = [e for e in wanted if e["scene_id"] in set(chosen)]
        print(f"[1-A] 샤드 {args.shard}/{args.shards}: 장면 {len(chosen)}개, "
              f"에피소드 {len(wanted)}개", flush=True)
    by_pose = {pose_key(e["scene_id"], e["start_position"], e["start_rotation"]):
               e["episode_id"] for e in wanted}
    if len(by_pose) != len(wanted):
        raise SystemExit(f"[1-A] 출발 자세가 겹치는 에피소드가 있다 "
                         f"({len(wanted)} -> {len(by_pose)}). 이름으로 쓸 수 없다")
    scenes = sorted({Path(e["scene_id"]).stem.split(".")[0] for e in wanted})
    print(f"[1-A] 우리 결과에서 {len(by_pose)} 에피소드 / 장면 {len(scenes)}개를 고름", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_ROOT / f"{args.run_name}.pt", map_location="cpu")
    policy = NavigationPolicy(heading_encoding=checkpoint.get("heading_encoding", "angle"),
                              num_heads=checkpoint.get("num_heads", 1),
                              token_dim=checkpoint.get("token_dim", 2176),
                              lstm_hidden=checkpoint.get("lstm_hidden", 2048),
                              lstm_layers=checkpoint.get("lstm_layers", 2),
                              side_dim=checkpoint.get("side_dim", 32))
    policy.load_state_dict(checkpoint["policy"])
    policy.to(device).eval()
    vlm = load_vision_backbone()

    env = build_env(scenes, args.split, set(by_pose))

    rows, started = [], time.time()
    for _ in range(len(env.episodes)):
        row = run_episode(env, vlm, policy, device)
        # Carry our own name for the episode so the comparison joins on it, not on the pose key.
        row["episode_id"] = by_pose[row.pop("pose_key")]
        rows.append(row)
        if len(rows) % 10 == 0:
            done = sum(r["success"] for r in rows)
            print(f"[1-A] {len(rows)}/{len(by_pose)} | 성공 {done} "
                  f"({100 * done / len(rows):.1f}%) | {time.time() - started:.0f}s", flush=True)
    env.close()

    args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    report(rows, args.compare)
    return 0


def report(rows: list[dict], compare: Path) -> None:
    """Ours against habitat-lab's, overall and episode by episode."""
    theirs = {r["episode_id"]: r for r in rows}
    print(f"\n=== habitat-lab 환경 {len(rows)} 에피소드 ===")
    print(f"  성공률 {100 * np.mean([r['success'] for r in rows]):.1f}%  "
          f"SPL {np.mean([r['spl'] for r in rows]):.3f}")

    if not compare.exists():
        print(f"\n[1-A] 비교 대상 {compare} 이 없어 대조를 건너뛴다")
        return

    mine = {r["episode_id"]: r for r in json.loads(compare.read_text())
            if r["episode_id"] in theirs}
    if not mine:
        print(f"\n[1-A] {compare.name} 과 에피소드 이름이 맞지 않아 대조할 수 없다")
        return

    shared = sorted(mine)
    ours_rate = 100 * np.mean([mine[k]["success"] for k in shared])
    them_rate = 100 * np.mean([theirs[k]["success"] for k in shared])
    disagree = [k for k in shared if mine[k]["success"] != theirs[k]["success"]]

    print(f"\n=== 같은 {len(shared)} 에피소드 대조 ===")
    print(f"  우리 루프        {ours_rate:5.1f}%")
    print(f"  habitat-lab     {them_rate:5.1f}%")
    print(f"  차이            {them_rate - ours_rate:+5.1f}%p   (판정 기준 ±3%p)")
    print(f"  성공/실패가 갈린 에피소드 {len(disagree)}개")
    for key in disagree[:8]:
        print(f"    {key}: 우리 {mine[key]['success']} / lab {theirs[key]['success']} | "
              f"최종거리 {mine[key].get('final_geodesic')} vs {theirs[key]['final_geodesic']} | "
              f"스텝 {mine[key]['steps']} vs {theirs[key]['steps']}")
    print(f"\n  판정: {'통과' if abs(them_rate - ours_rate) <= 3.0 else '실패'}")


if __name__ == "__main__":
    raise SystemExit(main())
