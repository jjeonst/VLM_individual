"""Assemble one trajectory's stored pieces into what the policy expects to read.

Three files were written per trajectory during the replay -- the images' representations, the
actions the person took, and the pose the agent held at each step -- and this module turns
them into batches.

**Poses need converting.** The replay stored absolute world coordinates, because the relative
form can be derived from them and not the other way round. What the paper's observation space
actually specifies, and what Habitat's sensors report, is the pose *relative to where the
episode began*: position becomes a displacement from the starting point expressed in the frame
the agent started facing, and heading becomes a turn away from the starting heading. That
conversion is done here and checked against Habitat's own sensor code rather than trusted.

**Frames hold different numbers of tokens**, because the model's answer runs to a different
length each time, and trajectories hold different numbers of frames. Both get padded, and both
paddings are handed to the policy so that nothing averages the padding in by mistake.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

HABITAT_ROOT = Path("/data/topovlm/habitat")
RENDER_SET = "pr2l_habitat_web_hd"
OBJECTS = ("bed", "chair", "plant", "sofa", "toilet", "tv_monitor")
OBJECT_INDEX = {name: index for index, name in enumerate(OBJECTS)}
NUM_ACTIONS = 4

# Appendix C.2, general item 6: the loss upweights the moments where the action changes, and
# additionally weights stopping and turning, "due to them being uncommon but important".
STOP_TURN_ACTIONS = (0, 2, 3)
STOP_TURN_WEIGHT = 1.5


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def episodic_pose(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Absolute (x, y, z, yaw) per step -> Habitat's episodic gps and compass.

    Reproduces `EpisodicGPSSensor` and `EpisodicCompassSensor`. The agent only ever rotates
    about the vertical axis in this task -- the actions that tilt the camera were removed --
    so a single yaw fixes the rotation and the sensors' quaternion algebra collapses to plane
    trigonometry. `check_against_habitat` confirms that collapse rather than assuming it.
    """
    start = pose[0]
    delta = pose[:, :3] - start[:3]
    yaw0 = float(start[3])

    # Rotate the displacement back into the frame the agent started facing, then take the two
    # horizontal components in the order the sensor reports them.
    forward = delta[:, 0] * np.cos(yaw0) - delta[:, 2] * np.sin(yaw0)
    sideways = delta[:, 0] * np.sin(yaw0) + delta[:, 2] * np.cos(yaw0)
    gps = np.stack([-sideways, forward], axis=1).astype(np.float32)

    compass = wrap_to_pi(pose[:, 3] - yaw0).astype(np.float32)
    return gps, compass


def check_against_habitat(samples: int = 200, seed: int = 0) -> dict[str, float]:
    """Compare this module's conversion with the sensors it is imitating."""
    import quaternion  # noqa: F401  (registers the dtype used below)
    from habitat.tasks.utils import cartesian_to_polar
    from habitat.utils.geometry_utils import quaternion_rotate_vector
    from habitat_sim.utils.common import quat_from_angle_axis

    rng = np.random.default_rng(seed)
    up = np.array([0.0, 1.0, 0.0])
    gps_error, compass_error = 0.0, 0.0

    for _ in range(samples):
        yaw0, yaw = rng.uniform(-np.pi, np.pi, size=2)
        p0 = rng.normal(size=3)
        p = rng.normal(size=3)
        pose = np.stack([[p0[0], p0[1], p0[2], yaw], [p[0], p[1], p[2], yaw]], axis=0)
        pose[0, 3] = yaw0

        gps, compass = episodic_pose(pose.astype(np.float32))

        q0 = quat_from_angle_axis(yaw0, up)
        q = quat_from_angle_axis(yaw, up)
        relative = quaternion_rotate_vector(q0.inverse(), p - p0)
        reference_gps = np.array([-relative[2], relative[0]], dtype=np.float32)

        direction = quaternion_rotate_vector(q.inverse() * q0, np.array([0.0, 0.0, -1.0]))
        reference_compass = cartesian_to_polar(-direction[2], direction[0])[1]

        gps_error = max(gps_error, float(np.abs(gps[1] - reference_gps).max()))
        compass_error = max(
            compass_error, float(abs(wrap_to_pi(np.array([compass[1] - reference_compass]))[0])))

    return {"gps_max_error": gps_error, "compass_max_error": compass_error}


def inflection_weights(actions: np.ndarray) -> np.ndarray:
    """Weight the steps where the demonstrated action changes, and the rare actions.

    Two thirds of every demonstration is "move forward". Left alone, a policy trained on that
    learns to walk straight and never turn or stop, which is exactly the behaviour that cannot
    solve the task. Habitat-Web's remedy, which the paper adopts, is to scale up the moments
    where the action differs from the previous one: those are the decisions, and the runs of
    identical actions between them are their consequences.

    The scale is the ratio of all steps to changing steps, so that the two groups end up
    contributing equally -- this is Habitat-Web's definition, which the paper cites without
    restating. On top of it, stopping and turning are weighted by 1.5 (Appendix C.2, item 6).
    """
    changed = np.ones(len(actions), dtype=bool)
    changed[1:] = actions[1:] != actions[:-1]
    ratio = len(actions) / max(int(changed.sum()), 1)

    weights = np.where(changed, ratio, 1.0).astype(np.float32)
    weights[np.isin(actions, STOP_TURN_ACTIONS)] *= STOP_TURN_WEIGHT
    return weights


@dataclass
class Trajectory:
    """One demonstration, ready for the policy."""

    episode_id: str
    tokens: np.ndarray          # [T, N, 2048] float16, padded across frames
    token_padding: np.ndarray   # [T, N] bool, True where padded
    actions: np.ndarray         # [T] int64
    gps: np.ndarray             # [T, 2] float32
    compass: np.ndarray         # [T] float32
    previous_action: np.ndarray  # [T, 4] float32, all zeros on the first step
    goal: np.ndarray            # [T, 6] float32
    weights: np.ndarray         # [T] float32


def trajectory_paths(record: dict, embedding_root: Path,
                     habitat_root: Path | None = None) -> tuple[Path, Path, Path]:
    """The three files one trajectory is spread across: tokens, actions, pose."""
    root = habitat_root or HABITAT_ROOT
    name = record["episode_id"]
    return (embedding_root / "train" / f"{name}.npz",
            root / "actions" / RENDER_SET / "train" / f"{name}.npy",
            root / "pose" / RENDER_SET / "train" / f"{name}.npy")


def read_tokens(token_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """One trajectory's token block, from plain arrays when they exist.

    The encoder writes an uncompressed `.npz`, and reading it means going through Python's
    `zipfile`: a directory parse and a chunked copy per member. With the page cache dropped
    between reads that path sustains 42 MB/s on this node against 120 MB/s for the same bytes as
    a bare `.npy`, and 969 against 4,489 MB/s when the pages are resident. `convert_npy.py`
    writes the plain pair beside the archive; this prefers it when present and falls back
    otherwise, so a partially converted directory still trains.
    """
    base = token_path.name[: -len(token_path.suffix)] if token_path.suffix else token_path.name
    plain = token_path.with_name(base + ".tokens.npy")
    if plain.exists():
        return np.load(plain), np.load(token_path.with_name(base + ".off.npy"))
    payload = np.load(token_path)
    return payload["tokens"], payload["offsets"]


def load_trajectory(record: dict, embedding_root: Path,
                    habitat_root: Path | None = None,
                    zero_tokens: bool = False) -> Trajectory:
    """One demonstration, ready for the policy.

    `zero_tokens` builds the blank control's input without reading the token files at all. The
    control blanks the representation anyway, so reading 365 GB an epoch to overwrite it with
    zeros costs thirteen hours for nothing -- and the width it needs is already recorded in the
    manifest. One token stands in for the frame, which is also what `evaluate.py` hands the
    policy in this condition, so the two match exactly rather than approximately.
    """
    token_path, action_path, pose_path = trajectory_paths(
        record, embedding_root, habitat_root)
    actions = np.load(action_path)
    pose = np.load(pose_path)
    if zero_tokens:
        steps = len(actions)
        offsets = np.arange(steps + 1, dtype=np.int64)
        tokens = np.zeros((steps, int(record["width"])), dtype=np.float16)
    else:
        tokens, offsets = read_tokens(token_path)

    steps = len(offsets) - 1
    if not (steps == len(actions) == len(pose)):
        raise ValueError(f"{record['episode_id']}: {steps} frames, {len(actions)} actions, "
                         f"{len(pose)} poses")

    counts = np.diff(offsets)
    width = int(counts.max())
    padded = np.zeros((steps, width, tokens.shape[1]), dtype=tokens.dtype)
    padding = np.ones((steps, width), dtype=bool)
    for step in range(steps):
        span = slice(offsets[step], offsets[step + 1])
        padded[step, :counts[step]] = tokens[span]
        padding[step, :counts[step]] = False

    gps, compass = episodic_pose(pose)

    previous = np.zeros((steps, NUM_ACTIONS), dtype=np.float32)
    previous[np.arange(1, steps), actions[:-1]] = 1.0

    goal = np.zeros((steps, len(OBJECTS)), dtype=np.float32)
    goal[:, OBJECT_INDEX[record["object_category"]]] = 1.0

    return Trajectory(record["episode_id"], padded, padding, actions.astype(np.int64),
                      gps, compass, previous, goal, inflection_weights(actions))


def collate(batch: list[Trajectory]) -> dict[str, torch.Tensor]:
    """Pad a set of trajectories to a common length and stack them.

    The token block keeps the half precision it was stored in, and is widened on the GPU
    instead. Casting here cost 38% of the time spent building a batch and doubled every byte
    afterwards -- through shared memory, over PCIe, and, because the queued batches crowd out
    the page cache, back onto the disk the next epoch has to read from. Widening float16 to
    float32 is exact, so where it happens changes the traffic and not the numbers.
    """
    steps = max(t.actions.shape[0] for t in batch)
    width = max(t.tokens.shape[1] for t in batch)
    size, dim = len(batch), batch[0].tokens.shape[2]

    tokens = torch.zeros(size, steps, width, dim,
                         dtype=torch.from_numpy(batch[0].tokens[:0]).dtype)
    token_padding = torch.ones(size, steps, width, dtype=torch.bool)
    actions = torch.zeros(size, steps, dtype=torch.long)
    gps = torch.zeros(size, steps, 2)
    compass = torch.zeros(size, steps)
    previous = torch.zeros(size, steps, NUM_ACTIONS)
    goal = torch.zeros(size, steps, len(OBJECTS))
    weights = torch.zeros(size, steps)
    valid = torch.zeros(size, steps, dtype=torch.bool)

    for index, item in enumerate(batch):
        length, count = item.actions.shape[0], item.tokens.shape[1]
        tokens[index, :length, :count] = torch.from_numpy(item.tokens)
        token_padding[index, :length, :count] = torch.from_numpy(item.token_padding)
        actions[index, :length] = torch.from_numpy(item.actions)
        gps[index, :length] = torch.from_numpy(item.gps)
        compass[index, :length] = torch.from_numpy(item.compass)
        previous[index, :length] = torch.from_numpy(item.previous_action)
        goal[index, :length] = torch.from_numpy(item.goal)
        weights[index, :length] = torch.from_numpy(item.weights)
        valid[index, :length] = True

    return {"tokens": tokens, "token_padding": token_padding, "actions": actions,
            "gps": gps, "compass": compass, "previous_action": previous, "goal": goal,
            "weights": weights, "valid": valid}


def read_manifest(embedding_root: Path) -> list[dict]:
    """Trajectories that have actually been encoded, once each, in a fixed order.

    Shard manifests accumulate across runs, and a run that shards the work differently from an
    earlier one leaves both layouts side by side. Concatenating them then lists some trajectories
    twice, and a duplicate row is not a harmless one: the trajectory is drawn twice per epoch, so
    a slice of the data is silently upweighted. The CoT condition had 8,085 rows for 7,824
    trajectories this way -- a stale `shard020.jsonl` from a superseded layout, every one of whose
    261 rows was already covered by the current twelve shards.

    Later shards win, on the reasoning that a re-encoded trajectory supersedes an older one; when
    the rows agree, as they did there, the choice does not matter.
    """
    unique: dict[str, dict] = {}
    for path in sorted((embedding_root / "manifests").glob("*.jsonl")):
        for line in path.open():
            row = json.loads(line)
            unique[row["episode_id"]] = row
    return sorted(unique.values(), key=lambda row: row["episode_id"])


if __name__ == "__main__":
    errors = check_against_habitat()
    print("Habitat 센서와의 최대 오차:")
    for name, value in errors.items():
        print(f"  {name:20s} {value:.3e}")
    assert max(errors.values()) < 1e-4, "상대 자세 변환이 Habitat 구현과 다르다"
    print("  → 상대 자세 변환이 Habitat 구현과 일치")

    demo = np.array([0, 1, 1, 1, 2, 2, 0], dtype=np.int64)
    print(f"\ninflection 가중 예시 (행동 {demo.tolist()}):")
    print(f"  {np.round(inflection_weights(demo), 2).tolist()}")
