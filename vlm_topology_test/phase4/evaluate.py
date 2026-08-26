"""Put the trained policy in a building it has never seen and see whether it finds the object.

Everything before this point could be checked against something -- the replay against recorded
poses, the extraction against a property the architecture guarantees. This cannot. The policy
acts, the world responds, and the only measure is whether it ended up in front of the right
object. That is the number the paper reports and the one this reproduction exists to produce.

**The model runs inside the loop.** At every step the agent's camera view has to be encoded
before the policy can choose, and the next view depends on what it chose, so the frames cannot
be prepared in advance the way they were for training. Several episodes are therefore stepped
side by side and their frames encoded together, which is the only batching available here.

**Scene meshes are cached on the node's local disk.** Evaluation opens and closes buildings
constantly, and loading one is many small scattered reads, where the round-trip to the shared
filesystem dominates. That is the access pattern local disk helps with -- unlike the large
sequential reads elsewhere in this pipeline, where the shared filesystem is the faster of the
two.

**Success is Habitat's own criterion**, which the paper adopts: the agent stopped of its own
accord within 0.1 m, by walkable distance, of a viewpoint from which the target is visible.
Stopping anywhere else, or never stopping within 500 steps, is a failure.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import dataset
import pca
from dataset import OBJECTS, OBJECT_INDEX, episodic_pose
from policy import NavigationPolicy
from render import (AGENT_HEIGHT, AGENT_RADIUS, CAMERA_HEIGHT, HFOV_DEG, MAX_CLIMB,
                    MAX_SLOPE, RGB_HEIGHT, RGB_WIDTH, SUCCESS_DISTANCE_M, TURN_DEG,
                    FORWARD_M, ALLOW_SLIDING, agent_pose, geodesic_to_viewpoints)
from vlm_features import (LAYERS, encode_batch, encode_vision_batch, load_vlm,
                          load_vision_backbone)

HABITAT_ROOT = Path("/data/topovlm/habitat")
EPISODE_ROOT = HABITAT_ROOT / "datasets/objectnav/hm3d/v1/objectnav_hm3d_v1"
SCENE_ROOT = HABITAT_ROOT / "scene_datasets"
SCENE_CONFIG_NAME = "hm3d/hm3d_annotated_basis.scene_dataset_config.json"
CHECKPOINT_ROOT = Path("/data/topovlm/checkpoints/pr2l_phase4")

MAX_EPISODE_STEPS = 500          # objectnav_hm3d.yaml: max_episode_steps
STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2, 3
SIM_ACTION = {MOVE_FORWARD: "move_forward", TURN_LEFT: "turn_left", TURN_RIGHT: "turn_right"}
DISTANCE_BINS = ((0, 2), (2, 4), (4, 6), (6, float("inf")))


def cache_scene(scene_id: str, scratch: Path) -> Path:
    """Copy one building's files to local disk, and report where they landed.

    Only the mesh files move. The scene dataset configuration stays where it is, because it is
    read once and refers to paths relative to the collection root.
    """
    source = SCENE_ROOT / scene_id
    target = scratch / scene_id
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    for path in source.parent.glob(f"{source.stem.split('.')[0]}*"):
        destination = target.parent / path.name
        if not destination.exists():
            shutil.copy2(path, destination)
    return target


def episode_uid(episode: dict) -> str:
    """A name that identifies one episode, which `episode_id` does not.

    In this dataset the field is close to useless as an identifier: the 2,000 validation
    episodes carry three distinct values between them, and the training split has 3.97 million
    episodes sharing 8,334. Ids repeat inside a single building, so neither the scene nor the
    target object rescues them.

    It matters because decoding is seeded per episode, so that a frame's generated answer does
    not depend on which other episodes happened to be batched beside it. Seeding from a name
    shared by hundreds of episodes gives the frames at step k of all of them one generator --
    they still decode differently, since their logits differ, but their draws are correlated
    across the evaluation set, which is not the independence the seeding was there to provide.

    The start pose is what actually distinguishes two episodes with the same target in the same
    house, so the name is derived from it.
    """
    payload = (np.asarray(episode["start_position"], dtype=np.float32).tobytes()
               + np.asarray(episode["start_rotation"], dtype=np.float32).tobytes())
    scene = Path(episode["scene_id"]).parent.name or Path(episode["scene_id"]).stem
    return f"{scene}_{episode['episode_id']}-{hashlib.md5(payload).hexdigest()[:8]}"


def load_episodes(split: str = "val", scenes: set[str] | None = None,
                  objects: tuple[str, ...] = OBJECTS) -> list[dict]:
    """Episodes with the viewpoints that count as having found the target.

    The validation split is the paper's evaluation set and the default. The training split is
    there for a different purpose: rolling out in a building the policy was trained on removes
    the demand to generalise, so a policy that behaves sensibly there and not in an unseen
    scene has a generalisation problem, while one that behaves badly in both has something
    more basic wrong. `scenes` restricts the load to the buildings actually trained on.
    """
    episodes = []
    for shard in sorted((EPISODE_ROOT / split / "content").glob("*.json.gz")):
        if scenes is not None and shard.name.replace(".json.gz", "") not in scenes:
            continue
        payload = json.loads(gzip.open(shard, "rt").read())
        viewpoints = {
            key: np.array([point["agent_state"]["position"]
                           for goal in goals for point in goal.get("view_points", [])],
                          dtype=np.float32)
            for key, goals in payload["goals_by_category"].items()}
        for episode in payload["episodes"]:
            if episode["object_category"] not in objects:
                continue
            key = f"{Path(episode['scene_id']).name}_{episode['object_category']}"
            episodes.append({
                "episode_id": episode_uid(episode),
                "source_episode_id": str(episode["episode_id"]),
                "scene_id": episode["scene_id"],
                "object_category": episode["object_category"],
                "start_position": np.asarray(episode["start_position"], dtype=np.float32),
                "start_rotation": np.asarray(episode["start_rotation"], dtype=np.float32),
                "geodesic_distance": float(episode["info"]["geodesic_distance"]),
                "viewpoints": viewpoints.get(key, np.empty((0, 3), dtype=np.float32)),
            })
    return episodes


def check_success_against_habitat(episodes: list[dict], sim, samples: int = 400,
                                 seed: int = 0) -> dict[str, float]:
    """Compare this file's success test with habitat-lab's own, on the same positions.

    The paper does not define success itself. Appendix C.1 says the specifications are "largely
    the same as the defaults provided by Habitat, as specified in the HM3D ObjectNav
    configuration file", which makes habitat-lab the definition and this a checkable claim
    rather than a matter of reading. What that configuration asks for is

        distance_to: VIEW_POINTS        success_distance: 0.1

    and the two measures that consume it are, in `habitat/tasks/nav/nav.py`:

        DistanceToGoal:  view_points = [vp.agent_state.position
                                        for goal in episode.goals
                                        for vp in goal.view_points]
                         distance    = sim.geodesic_distance(position, view_points, episode)
        Success:         is_stop_called and distance_to_target < success_distance

    Three things had to line up for our version to mean the same, and each is checked here
    rather than assumed. `episode.goals` is every instance of the target category in the
    building, not the one nearest the agent -- the dataset assigns
    `episode.goals = goals_by_category[episode.goals_key]` and the key is
    `f"{basename(scene_id)}_{object_category}"`, which is the key this file builds too. The
    distance is a single multi-goal query, `habitat_sim.MultiGoalShortestPath` with every
    viewpoint as an end, which is what `geodesic_to_viewpoints` issues. And the threshold is
    0.1 metres, strict in habitat-lab and inclusive here -- a difference that can only matter
    for a position sitting on the boundary to floating-point exactness.

    Positions are drawn from the navigable area rather than from a policy's rollout, because
    what is being compared is the measurement, not the behaviour, and scattered points cover
    more of the range than one agent's path would.
    """
    import habitat_sim

    rng = np.random.default_rng(seed)
    worst, checked, disagreements = 0.0, 0, 0

    for episode in episodes:
        points = episode["viewpoints"]
        if len(points) == 0:
            continue
        # habitat-lab's own formulation, transcribed from HabitatSim.geodesic_distance.
        reference_path = habitat_sim.MultiGoalShortestPath()
        reference_path.requested_ends = np.array(points, dtype=np.float32)

        for _ in range(max(1, samples // max(len(episodes), 1))):
            position = sim.pathfinder.get_random_navigable_point()
            reference_path.requested_start = np.array(position, dtype=np.float32)
            sim.pathfinder.find_path(reference_path)
            reference = float(reference_path.geodesic_distance)

            ours = geodesic_to_viewpoints(sim.pathfinder, position, points)
            if not (np.isfinite(reference) and np.isfinite(ours)):
                continue
            worst = max(worst, abs(reference - ours))
            # Both sides use the same comparison the scoring uses. Written with `<` on one side
            # and `<=` on the other, the two would differ exactly at the boundary and agree
            # everywhere else, so the check would report agreement it had not actually tested.
            disagreements += int((reference <= SUCCESS_DISTANCE_M)
                                 != (ours <= SUCCESS_DISTANCE_M))
            checked += 1

    return {"positions": checked, "max_distance_error": worst,
            "success_disagreements": disagreements}


def limit_scenes(episodes: list[dict], count: int | None, seed: int = 0) -> list[dict]:
    """Keep episodes from at most `count` buildings, chosen at random.

    Spreading a small sample thinly over many buildings is the worst way to spend the time.
    Every building has to be opened and its navigation mesh rebuilt, and episodes are stepped
    in parallel only within one building, so one episode per building pays the whole opening
    cost and then runs the model one frame at a time. Concentrating the same number of episodes
    into fewer buildings pays that cost a handful of times and lets the frames batch.

    It costs breadth: a success rate over eight buildings says less about how the policy fares
    across houses than one over forty. That is the right trade for a run whose question is
    whether the loop works at all, and the wrong one for the paper's comparison, which is why
    the validation runs leave this unset.
    """
    if count is None:
        return episodes
    names = sorted({episode["scene_id"] for episode in episodes})
    if count >= len(names):
        return episodes
    rng = np.random.default_rng(seed)
    keep = {names[i] for i in rng.choice(len(names), size=count, replace=False)}
    return [episode for episode in episodes if episode["scene_id"] in keep]


def stratified_subset(episodes: list[dict], count: int | None, seed: int = 0) -> list[dict]:
    """Take a sample that keeps each target object's share of the whole."""
    if count is None or count >= len(episodes):
        return episodes
    rng = np.random.default_rng(seed)
    by_object = defaultdict(list)
    for episode in episodes:
        by_object[episode["object_category"]].append(episode)

    chosen = []
    for name, group in sorted(by_object.items()):
        take = max(1, round(count * len(group) / len(episodes)))
        picked = rng.choice(len(group), size=min(take, len(group)), replace=False)
        chosen.extend(group[i] for i in sorted(picked))
    return chosen


def make_sim(glb: Path, scene_config: Path):
    """A simulator matching the one the demonstrations were replayed in."""
    import habitat_sim

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(glb)
    backend.scene_dataset_config_file = str(scene_config)
    backend.enable_physics = False
    backend.allow_sliding = ALLOW_SLIDING

    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [RGB_HEIGHT, RGB_WIDTH]
    rgb.hfov = HFOV_DEG
    rgb.position = [0.0, CAMERA_HEIGHT, 0.0]

    agent = habitat_sim.AgentConfiguration()
    agent.height = AGENT_HEIGHT
    agent.radius = AGENT_RADIUS
    agent.sensor_specifications = [rgb]
    agent.action_space = {
        "move_forward": habitat_sim.ActionSpec(
            "move_forward", habitat_sim.ActuationSpec(amount=FORWARD_M)),
        "turn_left": habitat_sim.ActionSpec(
            "turn_left", habitat_sim.ActuationSpec(amount=TURN_DEG)),
        "turn_right": habitat_sim.ActionSpec(
            "turn_right", habitat_sim.ActuationSpec(amount=TURN_DEG)),
    }
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))

    settings = habitat_sim.nav.NavMeshSettings()
    settings.set_defaults()
    settings.agent_radius = AGENT_RADIUS
    settings.agent_height = AGENT_HEIGHT
    settings.agent_max_climb = MAX_CLIMB
    settings.agent_max_slope = MAX_SLOPE
    settings.include_static_objects = False
    if not sim.recompute_navmesh(sim.pathfinder, settings):
        raise RuntimeError("navmesh recomputation failed")
    return sim


class Rollout:
    """One episode in progress: where its agent stands, what it has done, and how it ended.

    **A rollout does not own a simulator.** habitat-sim allows one per process -- building a
    second takes the GL context away from the first, and the next render aborts with
    "GL::Context::current(): no current context". Several episodes are still stepped together,
    because batching the frames is what keeps the card busy, so they share one simulator and
    take turns: each rollout carries its own agent state and hands it to the simulator just
    before it renders or moves. That is exact rather than approximate. Habitat's dynamics
    depend on the agent's current pose and the action alone (Appendix C.1: "All observations,
    actions, and associated dynamics are deterministic"), so restoring a pose restores the
    episode, and collisions and sliding resolve as they would have.
    """

    def __init__(self, episode: dict) -> None:
        import habitat_sim
        from habitat_sim.utils.common import quat_from_coeffs

        self.episode = episode
        self.state = habitat_sim.AgentState()
        self.state.position = np.asarray(episode["start_position"], dtype=np.float32)
        self.state.rotation = quat_from_coeffs(episode["start_rotation"])

        self.poses: list[np.ndarray] | None = None   # filled on the first resume
        self.previous_action = np.zeros(4, dtype=np.float32)
        self.memory = None
        self.steps = 0
        self.walked = 0.0
        self.finished = False
        self.stopped = False

    def resume(self, sim) -> None:
        """Put this episode's agent back into the shared simulator."""
        sim.get_agent(0).set_state(self.state, reset_sensors=True)
        if self.poses is None:
            self.poses = [np.asarray(agent_pose(sim), dtype=np.float32)]

    def observation(self, sim) -> np.ndarray:
        self.resume(sim)
        return np.asarray(sim.get_sensor_observations()["rgb"])[..., :3]

    def side_inputs(self) -> tuple[np.ndarray, float]:
        gps, compass = episodic_pose(np.stack(self.poses, axis=0))
        return gps[-1], float(compass[-1])

    def apply(self, sim, action: int) -> None:
        self.steps += 1
        if action == STOP:
            self.finished = self.stopped = True
            return
        self.resume(sim)
        before = self.poses[-1][:3]
        sim.step(SIM_ACTION[action])
        self.state = sim.get_agent(0).get_state()
        pose = np.asarray(agent_pose(sim), dtype=np.float32)
        self.walked += float(np.linalg.norm(pose[:3] - before))
        self.poses.append(pose)
        self.previous_action = np.eye(4, dtype=np.float32)[action]
        if self.steps >= MAX_EPISODE_STEPS:
            self.finished = True

    def result(self, sim) -> dict:
        distance = geodesic_to_viewpoints(sim.pathfinder, self.poses[-1][:3],
                                          self.episode["viewpoints"])
        success = bool(self.stopped and distance <= SUCCESS_DISTANCE_M)
        shortest = self.episode["geodesic_distance"]
        spl = float(success * shortest / max(self.walked, shortest, 1e-6))
        return {"episode_id": self.episode["episode_id"],
                "object_category": self.episode["object_category"],
                "scene_id": self.episode["scene_id"],
                "start_geodesic": shortest, "steps": self.steps,
                "stopped": self.stopped, "final_geodesic": None if not np.isfinite(distance)
                else round(distance, 4), "success": success, "spl": round(spl, 4)}


@torch.inference_mode()
def run_scene(vlm, policy, basis, episodes: list[dict], scene_config: Path, scratch: Path,
              parallel: int, condition: str, device, rng: np.random.Generator | None = None
              ) -> list[dict]:
    """Roll out every episode of one building, stepping several at a time.

    One simulator serves them all -- see `Rollout` for why there cannot be more than one -- and
    the episodes take turns inside it, so what runs in parallel is the model, not the physics.
    That is where the time goes anyway: a step is one render and one pass through a 7B model.
    """
    glb = cache_scene(episodes[0]["scene_id"], scratch)
    sim = make_sim(glb, scene_config)
    results = []
    try:
        # Take episodes looking for the same object together. The model is asked one question
        # per call, so a wave of episodes hunting six different objects splits into six small
        # calls, and small calls waste most of the card. Sorting costs nothing and turns those
        # into one full-width call whenever a building has several episodes for an object.
        queue = sorted(episodes, key=lambda episode: episode["object_category"])
        while queue:
            take = min(parallel, len(queue))
            active = [Rollout(queue.pop(0)) for _ in range(take)]
            while any(not r.finished for r in active):
                live = [r for r in active if not r.finished]

                # The random control (Appendix W, 1-B) asks what the task itself concedes to an
                # agent that sees nothing and remembers nothing. It never renders a frame, never
                # builds a representation and never loads a policy, so it shares only the
                # simulator, the episode set and the success test with the real runs -- which is
                # exactly the part of the pipeline it is meant to put a floor under.
                if condition == "random":
                    for rollout in live:
                        # `resume` is what records the opening pose, and the normal path gets it
                        # for free from `observation`. Skipping the render skips that too, and an
                        # episode whose first draw is STOP would then be scored with no poses at
                        # all -- the fastest way for this control to fail is the one where it
                        # stops immediately, which is also the likeliest, at one draw in four.
                        rollout.resume(sim)
                        rollout.apply(sim, int(rng.integers(0, 4)))
                    continue

                frames = np.stack([r.observation(sim) for r in live], axis=0)
                # One goal per episode, so a batch may mix goals; the encoder requires one
                # prompt per batch, hence the grouping.
                # Each condition builds the frame's representation its own way, exactly as its
                # training data was built. Getting this wrong is not a crash but a silent
                # mismatch between what the policy learned on and what it is now shown.
                features = {}
                if condition == "image_encoder":
                    # The vision backbone alone: no prompt, no generation, no reduction, no
                    # stacking. One batch serves every episode because there is no question to
                    # ask, which also makes this by far the cheapest condition to roll out.
                    for slot, item in enumerate(encode_vision_batch(vlm, frames)):
                        features[slot] = item
                elif condition != "zero":
                    by_goal = defaultdict(list)
                    for index, rollout in enumerate(live):
                        by_goal[rollout.episode["object_category"]].append(index)
                    for goal, indices in by_goal.items():
                        # Each episode seeds its own decoding, exactly as during training. A
                        # batch here spans episodes, so one shared name would hand two episodes
                        # standing at the same step the same seed.
                        encoded = encode_batch(vlm, frames[indices], goal,
                                               [live[i].episode["episode_id"] for i in indices],
                                               [live[i].steps for i in indices],
                                               with_cot=condition == "cot")
                        for slot, item in zip(indices, encoded):
                            features[slot] = item

                # The blank control never calls the model, which is what makes it cheap enough
                # to be worth running: one empty token stands in for the frame, and the policy
                # is left with the pose, the heading, the previous action and the goal.
                blank = condition == "zero"
                width = 1 if blank else max(features[i].tokens.shape[0]
                                            for i in range(len(live)))
                tokens = torch.zeros(len(live), 1, width, policy.summary.token_dim,
                                     device=device)
                padding = torch.full((len(live), 1, width), not blank,
                                     dtype=torch.bool, device=device)
                gps = torch.zeros(len(live), 1, 2, device=device)
                compass = torch.zeros(len(live), 1, device=device)
                previous = torch.zeros(len(live), 1, 4, device=device)
                goals = torch.zeros(len(live), 1, len(OBJECTS), device=device)

                for slot, rollout in enumerate(live):
                    if not blank:
                        # PR2L reduces each layer and stacks the two; the image encoder does
                        # neither, both being PR2L-specific (Appendix C.2, PR2L items 2 and 3).
                        reduced = (features[slot].tokens if basis is None else
                                   np.concatenate([basis.apply(features[slot].tokens[:, layer])
                                                   for layer in range(len(LAYERS))], axis=1))
                        count = reduced.shape[0]
                        tokens[slot, 0, :count] = torch.from_numpy(reduced).to(device)
                        padding[slot, 0, :count] = False
                    position, heading = rollout.side_inputs()
                    gps[slot, 0] = torch.from_numpy(position).to(device)
                    compass[slot, 0] = heading
                    previous[slot, 0] = torch.from_numpy(rollout.previous_action).to(device)
                    goals[slot, 0, OBJECT_INDEX[rollout.episode["object_category"]]] = 1.0

                states = [r.memory for r in live]
                if any(s is None for s in states):
                    memory = None
                else:
                    memory = (torch.cat([s[0] for s in states], dim=1),
                              torch.cat([s[1] for s in states], dim=1))
                logits, memory = policy(tokens, padding, gps, compass, previous, goals, memory)
                for slot, rollout in enumerate(live):
                    rollout.memory = (memory[0][:, slot:slot + 1].contiguous(),
                                      memory[1][:, slot:slot + 1].contiguous())
                    rollout.apply(sim, int(logits[slot, 0].argmax()))
            results.extend(r.result(sim) for r in active)
    finally:
        sim.close()
    return results


def summarise(results: list[dict]) -> None:
    """Report as the paper does, plus a breakdown by how far the goal started."""
    total = len(results)
    success = sum(r["success"] for r in results)
    print(f"\n=== 전체 {total} 에피소드 ===")
    print(f"  성공률 {100 * success / max(total, 1):.1f}%  "
          f"SPL {np.mean([r['spl'] for r in results]):.3f}  "
          f"자발적 정지 {100 * np.mean([r['stopped'] for r in results]):.1f}%")

    print("\n=== 목표 물체별 ===")
    by_object = defaultdict(list)
    for row in results:
        by_object[row["object_category"]].append(row)
    for name, group in sorted(by_object.items()):
        rate = 100 * sum(r["success"] for r in group) / len(group)
        print(f"  {name:12s} {len(group):4d} 에피소드  성공률 {rate:5.1f}%")

    print("\n=== 출발 측지거리별 ===")
    for low, high in DISTANCE_BINS:
        group = [r for r in results if low <= r["start_geodesic"] < high]
        if not group:
            continue
        rate = 100 * sum(r["success"] for r in group) / len(group)
        label = f"{low}-{high} m" if np.isfinite(high) else f"{low}+ m"
        print(f"  {label:10s} {len(group):4d} 에피소드  성공률 {rate:5.1f}%")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--condition", default="cot",
                        choices=["cot", "nocot", "image_encoder"])
    parser.add_argument("--split", default="val", choices=["val", "train"],
                        help="val is the paper's evaluation set; train removes the demand to "
                             "generalise and only asks whether the loop behaves")
    parser.add_argument("--episodes", type=int, default=None,
                        help="stratified subset size; default is every episode in the split")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--check-success", action="store_true",
                        help="compare this file's success test against habitat-lab's and exit")
    parser.add_argument("--zero-tokens", action="store_true",
                        help="blank the frame representations and skip the model entirely: "
                             "the control that measures what the VLM actually contributes")
    parser.add_argument("--random-actions", action="store_true",
                        help="draw actions uniformly and load no checkpoint at all: the floor "
                             "the task concedes to an agent that sees nothing (experiment 1-B)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seeds the random control so its number is reproducible")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="concentrate the sample into this many buildings; leave unset "
                             "for the validation runs, where breadth is the point")
    parser.add_argument("--scratch", type=Path, default=Path("/scratch/jonghoon/hm3d"))
    # Scenes are the natural seam: an episode never spans two of them, and the subset that
    # decides which 500 episodes run is computed before the split, so every shard sees the same
    # selection and takes a disjoint slice of it. Concatenating the shards' rows reproduces the
    # single-process result exactly rather than approximately.
    parser.add_argument("--shard", type=int, default=None,
                        help="which slice of the scenes this process runs")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    # Comparing the two success tests needs a pathfinder and nothing else, so it runs
    # before any model or checkpoint is touched -- otherwise it fails for reasons that
    # have nothing to do with what it is asking.
    if args.check_success:
        args.scratch.mkdir(parents=True, exist_ok=True)
        sample = stratified_subset(load_episodes(args.split), args.episodes)
        first = sorted({e["scene_id"] for e in sample})[0]
        group = [e for e in sample if e["scene_id"] == first]
        sim = make_sim(cache_scene(first, args.scratch), SCENE_ROOT / SCENE_CONFIG_NAME)
        try:
            report = check_success_against_habitat(group, sim)
        finally:
            sim.close()
        print(f"\n=== 성공 판정을 habitat-lab과 대조 ({report['positions']} 위치) ===")
        print(f"  최대 거리 오차   {report['max_distance_error']:.3e} m")
        print(f"  성공 판정 불일치 {report['success_disagreements']}")
        ok = report["max_distance_error"] < 1e-4 and report["success_disagreements"] == 0
        print("  → " + ("habitat-lab의 판정과 동일" if ok
                        else "불일치 — 판정 기준을 재검토할 것"))
        return 0 if ok else 1

    embedding_root = HABITAT_ROOT / "embeddings" / f"pr2l_habitat_web_hd_{args.condition}"
    # `zero` is not a stored condition -- it blanks whatever the run was trained on -- so the
    # embeddings it names are only there to find the reduction, and it has none of its own.
    condition = ("random" if args.random_actions else
                 "zero" if args.zero_tokens else args.condition)
    # The image encoder's representation was never reduced, so there is no basis to load.
    basis = (None if condition in ("random", "zero", "image_encoder")
             else pca.load(embedding_root / "pca.npz"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    if condition == "random":
        # No checkpoint is read. Loading one and then ignoring its outputs would leave the
        # control quietly depending on a trained artefact existing, which is the opposite of
        # what a floor is for.
        policy = None
    else:
        checkpoint = torch.load(CHECKPOINT_ROOT / f"{args.run_name}.pt", map_location="cpu")
        # The checkpoint carries the two settings the network's shape depends on, so a run
        # started with different ones is rebuilt the way it was trained rather than the way the
        # defaults say.
        policy = NavigationPolicy(heading_encoding=checkpoint.get("heading_encoding", "angle"),
                                  num_heads=checkpoint.get("num_heads", 1),
                                  token_dim=checkpoint.get("token_dim", 2048))
        policy.load_state_dict(checkpoint["policy"])
        policy.to(device).eval()
    # Only as much model as the condition needs: nothing for the two controls, the vision half
    # for the image encoder, the whole thing for PR2L.
    if condition in ("random", "zero"):
        vlm = None
    elif condition == "image_encoder":
        vlm = load_vision_backbone()
    else:
        vlm = load_vlm()

    scenes = None
    if args.split == "train":
        # Restrict to the buildings the policy actually saw, which the encoding manifest lists.
        # It names them the way the scene assets are laid out, "00386-b3WpMbPFB6q", while the
        # episode shards are named by the bare hash, "b3WpMbPFB6q.json.gz". Drop the prefix.
        scenes = {row["scene_key"].split("-", 1)[-1]
                  for row in dataset.read_manifest(embedding_root)}
        print(f"[eval] 학습에 쓴 장면 {len(scenes)}개로 제한", flush=True)
    episodes = stratified_subset(
        limit_scenes(load_episodes(args.split, scenes), args.max_scenes), args.episodes)
    by_scene = defaultdict(list)
    for episode in episodes:
        by_scene[episode["scene_id"]].append(episode)
    print(f"[eval] {len(episodes)} 에피소드 / {len(by_scene)} 장면", flush=True)

    if args.shard is not None:
        chosen = sorted(by_scene)[args.shard::args.shards]
        by_scene = {scene: by_scene[scene] for scene in chosen}
        kept = sum(len(group) for group in by_scene.values())
        print(f"[eval] 샤드 {args.shard}/{args.shards}: 장면 {len(by_scene)}개, "
              f"에피소드 {kept}개", flush=True)

    args.scratch.mkdir(parents=True, exist_ok=True)


    results, started = [], time.time()
    for index, (scene, group) in enumerate(sorted(by_scene.items()), start=1):
        results.extend(run_scene(vlm, policy, basis, group,
                                 SCENE_ROOT / SCENE_CONFIG_NAME, args.scratch,
                                 args.parallel, condition, device, rng))
        done = sum(r["success"] for r in results)
        print(f"[eval] 장면 {index}/{len(by_scene)} | 누적 {len(results)} 에피소드, "
              f"성공 {done} ({100 * done / len(results):.1f}%), "
              f"{time.time() - started:.0f}s", flush=True)

    summarise(results)
    default = (f"{args.run_name}_eval.json" if args.shard is None
               else f"{args.run_name}_eval_shard{args.shard:02d}.json")
    out = args.out or CHECKPOINT_ROOT / default
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[eval] {out} 저장")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
