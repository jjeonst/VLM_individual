"""Why do episodes time out? Sorting the policy's 224 timeouts by what the agent was doing.

The design document's motivation reads timeouts as failures of memory: the agent keeps walking
through places it has already searched. `evaluate.py --save-states` kept every pose and action of
the 500 evaluation episodes, so that reading can be checked from the walks themselves.

**Kinds of timeout**, decided in this order from poses and actions alone:

    stuck     100 or more steps in a row within about 0.5 m
    wander    not stuck, and either 2 or more returns or 25 % or more of its steps on old ground
    other     neither

A *return* is arriving within 1 m, on the same floor, of a place left more than 30 steps ago after
having gone at least 3 m from it.

**The navmesh matters.** `evaluate.make_sim` recomputes the navmesh for the agent's radius and
height, and the `.navmesh` shipped with a scene is not the same surface. Clearance measured on the
shipped one disagreed with what the agent ran into, so `stuck` saves the recomputed mesh and
everything else reads that.

    python probe_timeouts.py stuck --scene 00800-TEEsavR23oF   # one scene per process, GL + GPU
    python probe_timeouts.py classify                          # kinds and strata -> report.json
    python probe_timeouts.py demos                             # blocked steps in demos and walks
    python probe_timeouts.py figures                           # the three figures of the tab

`stuck` builds one simulator, which habitat-sim allows once per process, so it takes one scene.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate  # noqa: E402
import render  # noqa: E402

ROLLOUTS = Path("/data/topovlm/probes/lstm/stageB_nocotMin0C3_rollout")
DEMOS = Path("/data/topovlm/probes/lstm/stageB_nocotMin0C3")
CHECKPOINT = evaluate.CHECKPOINT_ROOT / "stageB_nocotMin0C3.pt"
OUT = Path("/data/topovlm/probes/timeouts")

STILL_RADIUS, STILL_STEPS = 0.5, 100
LOW_CAMERAS = (0.1, 0.3, 0.5, 0.7, render.CAMERA_HEIGHT)   # narrow cameras looking straight ahead
NEAR, LEFT, GAP, SAME_FLOOR = 1.0, 3.0, 30, 1.0
BLOCKED = 1e-3                     # a forward step that moved less than this did not move
FORWARD = evaluate.MOVE_FORWARD


def longest_stay(pose: np.ndarray) -> tuple[int, int]:
    """Where the longest run of steps within `STILL_RADIUS` of its last step begins, and its length."""
    flat = pose[:, [0, 2]]
    best = start = left = 0
    for t in range(len(flat)):
        while np.linalg.norm(flat[t] - flat[left:t + 1], axis=1).max() >= STILL_RADIUS:
            left += 1
        if t - left + 1 > best:
            best, start = t - left + 1, left
    return start, best


def walk_metrics(pose: np.ndarray) -> dict:
    """Longest stay, and the steps that stand on ground the agent had left, for one walk."""
    flat = pose[:, [0, 2]]
    start, best = longest_stay(pose)
    gaps = np.linalg.norm(flat[:, None] - flat[None], axis=2)
    same = np.abs(pose[:, None, 1] - pose[None, :, 1]) < SAME_FLOOR
    back = np.zeros(len(flat), bool)
    for t in range(len(flat)):
        for s in range(max(t - GAP, 0)):
            if same[s, t] and gaps[s, t] < NEAR and gaps[s, s:t].max() >= LEFT:
                back[t] = True
                break
    returns = int(back[0]) + int((back[1:] & ~back[:-1]).sum())
    return {"still_start": start, "still_run": best, "returns": returns,
            "back_share": float(back.mean()),
            "path": float(np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1).sum())}


def blocked_steps(pose: np.ndarray, actions: np.ndarray) -> np.ndarray:
    moved = np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1)
    return (actions[:len(moved)] == FORWARD) & (moved < BLOCKED)


def kind_of(row: dict) -> str:
    if row["stopped"]:
        return "stopped"
    if row["still_run"] >= STILL_STEPS:
        return "stuck"
    return "wander" if row["returns"] >= 2 or row["back_share"] >= 0.25 else "other"


def load_mesh(scene: str):
    import habitat_sim

    finder = habitat_sim.PathFinder()
    if not finder.load_nav_mesh(str(OUT / "navmesh" / f"{scene}.navmesh")):
        raise SystemExit(f"[timeouts] no navmesh for {scene}: run `stuck --scene {scene}` first")
    return finder


def path_shape(finder, episode: dict) -> tuple[float, float]:
    """Tortuosity to the nearest viewpoint v* (design document 6.2), and the height up to it."""
    import habitat_sim

    start = np.asarray(episode["start_position"], dtype=np.float32)
    query = habitat_sim.MultiGoalShortestPath()
    query.requested_start = start
    query.requested_ends = [np.asarray(v, dtype=np.float32) for v in episode["viewpoints"]]
    finder.find_path(query)
    target = episode["viewpoints"][query.closest_end_point_index]
    across = np.linalg.norm((target - start)[[0, 2]])
    return float(query.geodesic_distance / max(across, 1e-3)), float(abs(target[1] - start[1]))


def classify() -> list[dict]:
    meta = {r["episode_id"]: r for r in json.loads(
        (evaluate.CHECKPOINT_ROOT / "stageB_nocotMin0C3_eval_val.json").read_text())}
    episodes = {e["episode_id"]: e for e in evaluate.load_episodes("val")}
    meshes, rows = {}, []
    for path in sorted(ROLLOUTS.glob("*.npz")):
        walk = np.load(path)
        pose, actions = walk["pose"].astype(float), walk["actions"]
        uid = path.stem
        episode = episodes[uid]
        scene = uid.split("_")[0]
        finder = meshes.setdefault(scene, load_mesh(scene))
        block = blocked_steps(pose, actions)
        row = {"episode": uid, "scene": scene, "goal": episode["object_category"],
               "success": bool(walk["success"]), "stopped": bool(walk["stopped"]),
               "start": meta[uid]["start_geodesic"], "end": meta[uid]["final_geodesic"],
               "blocked_share": float(block.sum() / max((actions[:-1] == FORWARD).sum(), 1)),
               **walk_metrics(pose)}
        row["kind"] = kind_of(row)
        row["closest"] = float(np.nanmin([evaluate.geodesic_to_viewpoints(
            finder, point, episode["viewpoints"]) for point in pose[::5, :3]]))
        row["tortuosity"], row["height"] = path_shape(finder, episode)
        rows.append(row)
    (OUT / "episodes.json").write_text(json.dumps(rows, indent=1))
    return rows


def stratum(rows: list[dict]) -> dict:
    count = len(rows)
    kinds = Counter(r["kind"] for r in rows)
    wrong = sum(r["stopped"] and not r["success"] for r in rows)
    return {"n": count, "success": sum(r["success"] for r in rows), "wrong_stop": wrong,
            **{k: kinds[k] for k in ("stuck", "wander", "other")}}


def report(rows: list[dict]) -> dict:
    tortuosity = {"<1.2": lambda r: r["tortuosity"] < 1.2,
                  "1.2-1.6": lambda r: 1.2 <= r["tortuosity"] < 1.6,
                  ">=1.6": lambda r: r["tortuosity"] >= 1.6}
    floors = {"same": lambda r: r["height"] < SAME_FLOOR, "other": lambda r: r["height"] >= SAME_FLOOR}
    bands = ((2, 4), (4, 6), (6, 10), (10, np.inf))
    within = {}
    for low, high in bands:
        band = [r for r in rows if low <= r["start"] < high]
        within[f"{low}-{high}"] = {"straight": stratum([r for r in band if r["tortuosity"] < 1.2]),
                                   "winding": stratum([r for r in band if r["tortuosity"] >= 1.2])}
    timeouts = [r for r in rows if not r["stopped"]]
    stuck = [r for r in timeouts if r["kind"] == "stuck"]
    summary = {
        "all": stratum(rows),
        "tortuosity": {k: stratum([r for r in rows if f(r)]) for k, f in tortuosity.items()},
        "floor": {k: stratum([r for r in rows if f(r)]) for k, f in floors.items()},
        "start": {f"{a}-{b}": stratum([r for r in rows if a <= r["start"] < b])
                  for a, b in ((2, 4), (4, 6), (6, np.inf))},
        "within_distance": within,
        "stuck_onset_under_50": sum(r["still_start"] < 50 for r in stuck),
        "stuck_onset_median": float(np.median([r["still_start"] for r in stuck])),
        "stuck_share_of_timeout_steps": sum(r["still_run"] for r in stuck) / (500 * len(timeouts)),
        "blocked_share": {"stuck": float(np.mean([r["blocked_share"] for r in stuck])),
                          "success": float(np.mean([r["blocked_share"] for r in rows if r["success"]]))},
        "reached_a_viewpoint_and_timed_out": dict(Counter(
            r["kind"] for r in timeouts if r["closest"] < 0.1)),
        "stuck_with_heavy_return_before": sum(r["back_share"] >= 0.25 for r in stuck),
    }
    (OUT / "report.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return summary


def make_sim(scene: str, glb: Path):
    """`evaluate.make_sim` with the extra cameras: depth and semantics at the policy's camera."""
    import habitat_sim

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(glb)
    backend.scene_dataset_config_file = str(evaluate.SCENE_ROOT / evaluate.SCENE_CONFIG_NAME)
    backend.enable_physics = False
    backend.allow_sliding = render.ALLOW_SLIDING

    def camera(uuid, kind):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid, spec.sensor_type = uuid, kind
        spec.resolution = [render.RGB_HEIGHT, render.RGB_WIDTH]
        spec.hfov = render.HFOV_DEG
        spec.position = [0.0, render.CAMERA_HEIGHT, 0.0]
        return spec

    agent = habitat_sim.AgentConfiguration()
    agent.height, agent.radius = render.AGENT_HEIGHT, render.AGENT_RADIUS
    agent.sensor_specifications = [camera("rgb", habitat_sim.SensorType.COLOR),
                                   camera("depth", habitat_sim.SensorType.DEPTH),
                                   camera("semantic", habitat_sim.SensorType.SEMANTIC)]
    for i, height in enumerate(LOW_CAMERAS):
        for prefix, kind in (("low_depth", habitat_sim.SensorType.DEPTH), ("low_semantic", habitat_sim.SensorType.SEMANTIC)):
            spec = camera(f"{prefix}{i}", kind)
            spec.resolution, spec.hfov = [33, 33], 30.0
            spec.position = [0.0, height, 0.0]
            agent.sensor_specifications.append(spec)
    agent.action_space = {"move_forward": habitat_sim.ActionSpec(
        "move_forward", habitat_sim.ActuationSpec(amount=render.FORWARD_M))}
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
    settings = habitat_sim.nav.NavMeshSettings()
    settings.set_defaults()
    settings.agent_radius, settings.agent_height = render.AGENT_RADIUS, render.AGENT_HEIGHT
    settings.agent_max_climb, settings.agent_max_slope = render.MAX_CLIMB, render.MAX_SLOPE
    settings.include_static_objects = False
    if not sim.recompute_navmesh(sim.pathfinder, settings):
        raise RuntimeError("navmesh recomputation failed")
    (OUT / "navmesh").mkdir(parents=True, exist_ok=True)
    sim.pathfinder.save_nav_mesh(str(OUT / "navmesh" / f"{scene}.navmesh"))
    return sim


def look_at_stuck(scene: str) -> None:
    """Stand where each stuck agent stood, retry the step, and see what is in front of it."""
    import habitat_sim

    walks = sorted(ROLLOUTS.glob(f"{scene}_*.npz"))
    if not walks:
        return
    sim = make_sim(scene, evaluate.SCENE_ROOT / str(np.load(walks[0])["scene"]))
    names = {int(o.id.split("_")[-1]): o.category.name()
             for o in sim.semantic_scene.objects if o is not None}
    (OUT / "frames").mkdir(parents=True, exist_ok=True)

    def stand(position, yaw):
        state = habitat_sim.AgentState()
        state.position = np.asarray(position, dtype=np.float32)
        state.rotation = habitat_sim.utils.common.quat_from_angle_axis(
            float(yaw), np.array([0.0, 1.0, 0.0]))
        sim.get_agent(0).set_state(state, reset_sensors=True)
        return sim.get_sensor_observations()

    def step_length(position, yaw):
        stand(position, yaw)
        sim.step("move_forward")
        return float(np.linalg.norm(sim.get_agent(0).get_state().position - position))

    rows = []
    for path in walks:
        walk = np.load(path)
        pose, actions = walk["pose"].astype(float), walk["actions"]
        metrics = walk_metrics(pose)
        if bool(walk["stopped"]) or metrics["still_run"] < STILL_STEPS:
            continue
        first, run = metrics["still_start"], metrics["still_run"]
        block = blocked_steps(pose, actions)
        steps = [t for t in range(first, min(first + run, len(block))) if block[t]]
        if not steps:
            continue
        t = steps[len(steps) // 2]
        position, yaw = pose[t, :3].astype(np.float32), pose[t, 3]
        seen = stand(position, yaw)
        depth = seen["depth"]
        rows_mid, cols_mid = depth.shape[0] // 2, depth.shape[1] // 2
        centre = depth[rows_mid - 40:rows_mid + 40, cols_mid - 40:cols_mid + 40]
        # The strip of the frame the body would sweep through. Anything lower than the bottom of
        # the frame, or beside the agent nearer than the frame's edge, cannot appear here at all.
        focal = cols_mid / np.tan(np.deg2rad(render.HFOV_DEG / 2))
        across = (np.arange(depth.shape[1]) - cols_mid + 0.5) / focal
        strip = (depth > 0) & (depth < 0.5) & (np.abs(across[None] * depth) <= render.AGENT_RADIUS + 0.02)
        labels = Counter(names.get(int(i), "?") for i in seen["semantic"][strip])
        ahead = []
        for i in range(len(LOW_CAMERAS)):
            near = seen[f"low_depth{i}"][12:21, 12:21]
            near = near[near > 0]
            if len(near) and np.percentile(near, 10) < 0.6:
                found = Counter(seen[f"low_semantic{i}"][12:21, 12:21].ravel().tolist())
                ahead.append(names.get(int(found.most_common(1)[0][0]), "?"))
        finder, metres = sim.pathfinder, 0.01
        floor = finder.get_topdown_view(metres, float(position[1]))
        low, _ = finder.get_bounds()
        column, row = int((position[0] - low[0]) / metres), int((position[2] - low[2]) / metres)
        crop = np.pad(floor, 120)[row:row + 240, column:column + 240]
        np.savez_compressed(OUT / "frames" / f"{path.stem}.npz", rgb=np.asarray(seen["rgb"])[..., :3],
                            crop=crop, heading=yaw, t=t, centre=float(np.median(centre[centre > 0])))
        rows.append({"episode": path.stem, "t": int(t), "retry": step_length(position, yaw),
                     "free_headings": sum(step_length(position, yaw + k * np.pi / 6) > 0.2 for k in range(12)),
                     "centre_depth": float(np.median(centre[centre > 0])), "strip_pixels": int(strip.sum()),
                     "strip_label": labels.most_common(1)[0][0] if labels else None,
                     "low_ahead": ahead[0] if ahead else None})
        print(rows[-1], flush=True)
    (OUT / "stuck").mkdir(parents=True, exist_ok=True)
    (OUT / "stuck" / f"{scene}.json").write_text(json.dumps(rows, indent=1))
    sim.close()


def tally(paths: list[Path]) -> dict:
    after_blocked, after_moved, runs, stays = np.zeros(4), np.zeros(4), [], []
    forward = blocked = 0
    for path in paths:
        walk = np.load(path)
        pose, actions = walk["pose"].astype(float), walk["actions"]
        block = blocked_steps(pose, actions)
        moved = (actions[:len(block)] == FORWARD) & ~block
        forward, blocked = forward + int(moved.sum() + block.sum()), blocked + int(block.sum())
        for t in range(len(block) - 1):
            if block[t]:
                after_blocked[actions[t + 1]] += 1
            if moved[t]:
                after_moved[actions[t + 1]] += 1
        length = 0
        for flag in list(block) + [False]:
            if flag:
                length += 1
            elif length:
                runs.append(length)
                length = 0
        stays.append(longest_stay(pose)[1])
    runs, stays = np.array(runs), np.array(stays)
    return {"walks": len(paths), "blocked_share_of_forward": blocked / forward,
            "forward_after_blocked": float(after_blocked[FORWARD] / after_blocked.sum()),
            "forward_after_moved": float(after_moved[FORWARD] / after_moved.sum()),
            "blocked_run_percentiles_50_99": np.percentile(runs, [50, 99]).tolist(),
            "blocked_runs_50_or_more": int((runs >= 50).sum()), "stays_100_or_more": int((stays >= 100).sum())}


def demos() -> None:
    """How people and the policy behave after a blocked step, and what the policy would pick."""
    import torch

    rows = json.loads((OUT / "episodes.json").read_text())
    pick = lambda test: [ROLLOUTS / f"{r['episode']}.npz" for r in rows if test(r)]  # noqa: E731
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = state.get("policy", state.get("model", state.get("state_dict", state)))
    weight, bias = state["action.weight"].float(), state["action.bias"].float()
    stuck = []
    for row in (r for r in rows if r["kind"] == "stuck"):
        hidden = torch.from_numpy(np.load(ROLLOUTS / f"{row['episode']}.npz")["h"].astype(np.float32))
        stuck.append(torch.softmax(hidden @ weight.T + bias, 1)[row["still_start"]:
                                                                 row["still_start"] + row["still_run"]])
    stuck = torch.cat(stuck)
    summary = {"demos": tally(sorted(DEMOS.glob("*.npz"))),
               "success": tally(pick(lambda r: r["success"])),
               "timeout": tally(pick(lambda r: not r["stopped"])),
               "stuck_probabilities_stop_forward_left_right": stuck.mean(0).tolist(),
               "stuck_mean_top_probability": float(stuck.max(1).values.mean())}
    (OUT / "demos.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


WANDER = ("00843-DYehNKdT76V_0-43ff3e66", "00848-ziup5kvtCCR_0-7dbf2b31", "00829-QaLdnwvtxbs_0-a48c6670",
          "00839-zt1RVoi7PcG_0-30237331", "00814-p53SfW6mjZe_0-fc2cd8cf", "00824-Dd4bFSTQ8gi_0-053cf046")
STUCK = ("00848-ziup5kvtCCR_0-9ba72bec", "00824-Dd4bFSTQ8gi_0-f6ccee05", "00843-DYehNKdT76V_0-f54a83ea",
         "00839-zt1RVoi7PcG_0-7768497b", "00800-TEEsavR23oF_0-0c7ec6ee", "00880-Nfvxx8J5NCo_0-e0fdb117",
         "00890-6s7QHgap2fW_0-98265c56", "00802-wcojb4TFT35_0-0b20e8be")


def figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Circle, Patch

    rows = {r["episode"]: r for r in json.loads((OUT / "episodes.json").read_text())}
    episodes = {e["episode_id"]: e for e in evaluate.load_episodes("val")}
    here = Path(__file__).resolve().parent

    def floor(ax, uid):
        pose = np.load(ROLLOUTS / f"{uid}.npz")["pose"].astype(float)
        finder, metres = load_mesh(rows[uid]["scene"]), 0.05
        height = float(np.median(pose[:, 1]))
        view = finder.get_topdown_view(metres, height)
        low, _ = finder.get_bounds()
        x, z = (pose[:, 0] - low[0]) / metres, (pose[:, 2] - low[2]) / metres
        ax.imshow(view, cmap="Greys", vmin=-0.6, vmax=4, interpolation="nearest")
        goals = episodes[uid]["viewpoints"]
        goals = goals[np.abs(goals[:, 1] - height) < SAME_FLOOR]
        if len(goals):
            ax.scatter((goals[:, 0] - low[0]) / metres, (goals[:, 2] - low[2]) / metres, s=3,
                       c="#d62728", alpha=0.6, linewidths=0, zorder=2)
        ys, xs = np.where(view)
        ax.set_xlim(min(xs.min(), x.min()) - 8, max(xs.max(), x.max()) + 8)
        ax.set_ylim(max(ys.max(), z.max()) + 8, min(ys.min(), z.min()) - 8)
        ax.set_xticks([])
        ax.set_yticks([])
        return pose, x, z, len(goals)

    def goal_line(row):
        return (f"{row['scene'][:5]} {row['goal']}: goal {row['start']:.1f} m at start, "
                f"{row['closest']:.1f} m closest, {row['end']:.1f} m at end")

    fig = plt.figure(figsize=(16, 10.5))
    grid = fig.add_gridspec(4, 3, height_ratios=[3.4, 0.8, 3.4, 0.8], hspace=0.5, wspace=0.14)
    for i, uid in enumerate(WANDER):
        row = rows[uid]
        ax = fig.add_subplot(grid[(i // 3) * 2, i % 3])
        pose, x, z, goals = floor(ax, uid)
        points = np.c_[x, z]
        ax.add_collection(LineCollection(np.stack([points[:-1], points[1:]], 1), cmap="viridis",
                                         array=np.arange(len(x) - 1), linewidth=1.4, zorder=3))
        ax.scatter(x[0], z[0], marker="o", s=50, c="white", edgecolors="k", zorder=4)
        ax.scatter(x[-1], z[-1], marker="X", s=60, c="#ff7f0e", edgecolors="k", zorder=4)
        note = "" if goals else " (goal on another floor)"
        ax.set_title(f"{goal_line(row)}\nwalked {row['path']:.0f} m, {row['back_share'] * 100:.0f} % of steps "
                     f"on old ground{note}", fontsize=9)
        strip = fig.add_subplot(grid[(i // 3) * 2 + 1, i % 3])
        strip.plot(np.linalg.norm(pose[:, [0, 2]] - pose[0, [0, 2]], axis=1), lw=1, color="#1f4e9c")
        strip.set_xlim(0, 500)
        strip.set_ylabel("m from start", fontsize=8)
        strip.set_xlabel("step", fontsize=8, labelpad=0)
        strip.tick_params(labelsize=7)
    fig.savefig(here / "timeout_wander.png", dpi=120, bbox_inches="tight")

    colours = ListedColormap(["#2a9d55", "#d62828", "#9db4d6", "#3f5f93", "#000000"])
    fig = plt.figure(figsize=(14, 13.5))
    outer = fig.add_gridspec(4, 2, hspace=0.36, wspace=0.1, top=0.95)
    for i, uid in enumerate(STUCK):
        row = rows[uid]
        inner = outer[i].subgridspec(2, 1, height_ratios=[6, 1], hspace=0.28)
        ax = fig.add_subplot(inner[0])
        pose, x, z, _ = floor(ax, uid)
        first, run = row["still_start"], row["still_run"]
        actions = np.load(ROLLOUTS / f"{uid}.npz")["actions"][:len(pose)]
        ax.plot(x[:first + 1], z[:first + 1], color="#1f4e9c", lw=1.5, zorder=3)
        ax.scatter(x[0], z[0], marker="o", s=45, c="white", edgecolors="k", zorder=4)
        ax.add_patch(Circle((x[first:first + run].mean(), z[first:first + run].mean()), 18,
                            fill=False, ec="#ff7f0e", lw=2, zorder=5))
        ax.set_title(f"{goal_line(row)}\nstuck from step {first} for {run} steps", fontsize=9)
        ribbon = fig.add_subplot(inner[1])
        moved = np.r_[np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1) > BLOCKED, False]
        code = np.where(actions == FORWARD, np.where(moved, 0, 1),
                        np.where(actions == evaluate.TURN_LEFT, 2, np.where(actions == evaluate.TURN_RIGHT, 3, 4)))
        ribbon.imshow(code[None], aspect="auto", cmap=colours, vmin=0, vmax=4, interpolation="nearest")
        ribbon.axvline(first, color="#ff7f0e", lw=2)
        ribbon.set_yticks([])
        ribbon.tick_params(labelsize=7)
        ribbon.set_xlabel("step | after onset: " + "".join("SFLR"[a] for a in actions[first:first + 48]),
                          fontsize=7.5, labelpad=1, family="monospace")
    fig.legend(handles=[Patch(color="#2a9d55", label="forward, moved"), Patch(color="#d62828", label="forward, blocked"),
                        Patch(color="#9db4d6", label="turn left"), Patch(color="#3f5f93", label="turn right")],
               loc="upper center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, 0.995))
    fig.savefig(here / "timeout_stuck.png", dpi=120, bbox_inches="tight")

    fig = plt.figure(figsize=(15, 13))
    grid = fig.add_gridspec(4, 4, hspace=0.42, wspace=0.05)
    for i, uid in enumerate(STUCK):
        saved, row = np.load(OUT / "frames" / f"{uid}.npz"), rows[uid]
        top = fig.add_subplot(grid[(i // 4) * 2, i % 4])
        below = fig.add_subplot(grid[(i // 4) * 2 + 1, i % 4])
        top.imshow(saved["rgb"])
        top.set_title(f"{row['scene'][:5]} {row['goal']}, step {int(saved['t'])}\n"
                      f"depth straight ahead {float(saved['centre']):.1f} m", fontsize=9)
        below.imshow(saved["crop"], cmap="Greys", vmin=-0.6, vmax=4, interpolation="nearest")
        heading = float(saved["heading"])
        ahead = np.array([-np.sin(heading), -np.cos(heading)]) * 25
        below.add_patch(Circle((120, 120), 18, fill=False, ec="#1f4e9c", lw=2))
        below.arrow(120, 120, ahead[0], ahead[1], color="#d62828", width=2, head_width=8, length_includes_head=True)
        below.set_xlabel("walkable floor, 2.4 m wide", fontsize=8)
        for ax in (top, below):
            ax.set_xticks([])
            ax.set_yticks([])
    fig.savefig(here / "timeout_views.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("stuck", "classify", "demos", "figures"))
    parser.add_argument("--scene")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "stuck":
        look_at_stuck(args.scene)
    elif args.command == "classify":
        report(classify())
    elif args.command == "demos":
        demos()
    else:
        figures()
