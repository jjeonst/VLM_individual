"""Why do the two evaluation loops walk different paths? (follow-up to experiment 1-A)

1-A put the same policy inside habitat-lab's `Env` and got different trajectories -- 380 steps
against 202 on one episode, 500 against 117 on another. Two explanations fit that, and they call
for opposite responses:

  1. The loops feed the policy different things. Some convention differs -- the episodic GPS,
     the heading sign, when `previous_action` is updated, how the frame is read -- and the two
     runs part company at the first step where it matters. That is a bug, and it would sit under
     every number this project has reported.

  2. The loops feed the policy the same things, and a four-way argmax flips somewhere on a near
     tie. Our loop encodes eight frames in one batch, habitat-lab hands over one; half precision
     moves the logits in the last bits. One flipped step sends a closed loop somewhere else
     entirely and it never comes back. That is not a bug, but it does mean episodes cannot be
     compared one against one, and 1-A has to be read as an aggregate over many.

**Driving both loops with one action sequence separates them.** habitat-lab runs the episode
freely and its actions are recorded; our loop then replays that exact sequence. With the action
history forced identical, any difference in what the policy sees is a difference between the
loops rather than a consequence of an earlier disagreement. If the observations agree step for
step, explanation 1 is out.

The policy's own choice is recorded at each step as well, so the first place the two loops *would*
have parted is visible even while they are being kept together.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dataset import OBJECTS, OBJECT_INDEX
from evaluate import (CHECKPOINT_ROOT, Rollout, SCENE_CONFIG_NAME, SCENE_ROOT,
                      cache_scene, load_episodes, make_sim, stratified_subset)
from policy import NavigationPolicy
from validate_env import ACTION_NAME, build_env, pose_key, pose_key_of
from vlm_features import encode_vision_batch, load_vision_backbone


def digest(frame: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(frame).tobytes()).hexdigest()[:8]


def act(vlm, policy, frame, gps, compass, previous, goal, memory, device):
    """One policy step, written once so both loops go through identical code."""
    reduced = encode_vision_batch(vlm, frame[None])[0].tokens
    tokens = torch.zeros(1, 1, reduced.shape[0], policy.summary.token_dim, device=device)
    tokens[0, 0, :reduced.shape[0]] = torch.from_numpy(reduced).to(device)
    padding = torch.zeros(1, 1, reduced.shape[0], dtype=torch.bool, device=device)
    gps_t = torch.as_tensor(gps, dtype=torch.float32, device=device).reshape(1, 1, 2)
    compass_t = torch.as_tensor(compass, dtype=torch.float32, device=device).reshape(1, 1)
    logits, memory = policy(tokens, padding, gps_t, compass_t, previous, goal, memory)
    return logits[0, 0].float().cpu().numpy(), memory


@torch.inference_mode()
def run_habitat(env, vlm, policy, device, limit: int) -> list[dict]:
    """habitat-lab drives the episode; every input and choice is written down."""
    observations = env.reset()
    episode = env.current_episode
    goal = torch.zeros(1, 1, len(OBJECTS), device=device)
    goal[0, 0, OBJECT_INDEX[episode.object_category]] = 1.0
    previous = torch.zeros(1, 1, 4, device=device)
    memory, trace = None, []

    while not env.episode_over and len(trace) < limit:
        frame = np.asarray(observations["rgb"])[..., :3]
        gps = np.asarray(observations["gps"], dtype=np.float32).reshape(2)
        compass = float(np.asarray(observations["compass"]).reshape(-1)[0])
        logits, memory = act(vlm, policy, frame, gps, compass, previous, goal, memory, device)
        action = int(logits.argmax())
        trace.append({"step": len(trace), "action": action, "gps": gps.tolist(),
                      "compass": compass, "rgb": digest(frame),
                      "logits": [round(float(x), 4) for x in logits]})
        previous = torch.zeros(1, 1, 4, device=device)
        previous[0, 0, action] = 1.0
        if action == 0:
            break
        observations = env.step(ACTION_NAME[action])
    return trace


@torch.inference_mode()
def run_ours(episode: dict, actions: list[int], vlm, policy, device, scratch: Path) -> list[dict]:
    """Our loop, replaying habitat-lab's actions so the histories cannot drift apart."""
    glb = cache_scene(episode["scene_id"], scratch)
    sim = make_sim(glb, SCENE_ROOT / SCENE_CONFIG_NAME)
    try:
        rollout = Rollout(episode)
        goal = torch.zeros(1, 1, len(OBJECTS), device=device)
        goal[0, 0, OBJECT_INDEX[episode["object_category"]]] = 1.0
        previous = torch.zeros(1, 1, 4, device=device)
        memory, trace = None, []

        for step, action in enumerate(actions):
            frame = rollout.observation(sim)
            gps, compass = rollout.side_inputs()
            logits, memory = act(vlm, policy, frame, gps, compass, previous, goal, memory,
                                 device)
            trace.append({"step": step, "action": int(logits.argmax()),
                          "gps": np.asarray(gps).reshape(2).tolist(), "compass": float(compass),
                          "rgb": digest(frame),
                          "logits": [round(float(x), 4) for x in logits]})
            previous = torch.zeros(1, 1, 4, device=device)
            previous[0, 0, action] = 1.0
            if action == 0:
                break
            rollout.apply(sim, action)
    finally:
        sim.close()
    return trace


def compare(theirs: list[dict], ours: list[dict]) -> None:
    print(f"\n{'스텝':>4s} {'rgb 일치':>8s} {'gps 차이':>10s} {'나침반 차이':>12s} "
          f"{'행동 우리/lab':>14s} {'로짓 최대차':>12s}")
    first_input, first_action = None, None
    for a, b in zip(theirs, ours):
        rgb_same = a["rgb"] == b["rgb"]
        dgps = float(np.abs(np.array(a["gps"]) - np.array(b["gps"])).max())
        dcomp = abs(a["compass"] - b["compass"])
        dlog = float(np.abs(np.array(a["logits"]) - np.array(b["logits"])).max())
        mark = "" if a["action"] == b["action"] else "  <- 행동 갈림"
        print(f"{a['step']:4d} {'O' if rgb_same else 'X':>8s} {dgps:10.2e} {dcomp:12.2e} "
              f"{b['action']:6d}/{a['action']:<6d} {dlog:12.2e}{mark}")
        if first_input is None and (not rgb_same or dgps > 1e-3 or dcomp > 1e-3):
            first_input = a["step"]
        if first_action is None and a["action"] != b["action"]:
            first_action = a["step"]

    print(f"\n  입력이 처음 갈린 스텝   {first_input if first_input is not None else '없음'}")
    print(f"  행동이 처음 갈린 스텝   {first_action if first_action is not None else '없음'}")
    if first_input is None:
        print("\n  → 같은 행동열에서 두 루프가 정책에 **같은 것**을 먹인다.")
        print("    행동이 갈렸다면 반정밀도 로짓 위의 argmax 뒤집힘이며, 폐루프가 그것을 증폭한다.")
        print("    개별 에피소드를 짝지어 비교할 수 없다는 뜻이고, 1-A는 집계로 읽어야 한다.")
    else:
        print(f"\n  → 스텝 {first_input}에서 입력 자체가 다르다. 두 루프 사이의 계통적 차이이며,")
        print("    그 줄의 rgb/gps/나침반 중 어느 것이 어긋났는지가 원인을 가리킨다.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="stageB_image")
    parser.add_argument("--index", type=int, default=0, help="비교할 에피소드 (부분집합 내 순번)")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--subset", type=int, default=500)
    parser.add_argument("--scratch", type=Path, default=Path("/scratch/jonghoon/hm3d"))
    parser.add_argument("--out", type=Path, default=CHECKPOINT_ROOT / "diagnose_1a.json")
    args = parser.parse_args()

    pool = stratified_subset(load_episodes("val"), args.subset)
    episode = pool[args.index]
    key = pose_key(episode["scene_id"], episode["start_position"], episode["start_rotation"])
    print(f"[진단] {episode['episode_id']} | {episode['object_category']} | "
          f"출발거리 {episode['geodesic_distance']:.2f} m", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_ROOT / f"{args.run_name}.pt", map_location="cpu")
    policy = NavigationPolicy(heading_encoding=checkpoint.get("heading_encoding", "angle"),
                              num_heads=checkpoint.get("num_heads", 1),
                              token_dim=checkpoint.get("token_dim", 2176))
    policy.load_state_dict(checkpoint["policy"])
    policy.to(device).eval()
    vlm = load_vision_backbone()

    scene = Path(episode["scene_id"]).stem.split(".")[0]
    env = build_env([scene], "val", {key})
    theirs = run_habitat(env, vlm, policy, device, args.steps)
    env.close()
    print(f"[진단] habitat-lab {len(theirs)} 스텝 기록", flush=True)

    args.scratch.mkdir(parents=True, exist_ok=True)
    ours = run_ours(episode, [row["action"] for row in theirs], vlm, policy, device,
                    args.scratch)
    print(f"[진단] 우리 루프 {len(ours)} 스텝 재생", flush=True)

    compare(theirs, ours)
    args.out.write_text(json.dumps({"episode": episode["episode_id"],
                                    "habitat": theirs, "ours": ours},
                                   indent=2, ensure_ascii=False) + "\n")
    print(f"\n[진단] {args.out} 저장")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
