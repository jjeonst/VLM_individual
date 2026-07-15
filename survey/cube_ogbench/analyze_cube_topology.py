"""Cube benchmark (OGBench) — manipulation topology analysis.

OGBench (Park et al., "OGBench: Benchmarking Offline Goal-Conditioned RL") ships
`cube-{single,double,triple,quadruple}` tasks: a robot arm pick-and-places N cubes
to goal configurations. Unlike navigation, "topology" here is NOT a physical map —
it is the **task structure**: which cube to manipulate (choice point), the
stacking / object-on-object relations (precondition graph), and the pick→place→
stack skill sequence (task-stage graph). This is the assignment's robotics
expansion example, done on real data.

We use `cube-double` (2 cubes) — the smallest task with genuine ordering
structure. State layout (from ogbench.relabel_utils): qpos[0:14] = arm, then each
cube = 7 (xyz + quaternion); so cube i xyz = qpos[:, 14+i*7 : 14+i*7+3].
The dataset is task-agnostic "play" data, so it explores BOTH stacking orders —
we report that honestly (a specific goal-conditioned task would fix one order).

Downloads the small val split (~30 MB) if absent; no MuJoCo/env needed.

Outputs (survey/cube_ogbench/results/):
  - cube_topology.json          : all stats + a chosen episode timeline
  - cube_branch_candidates.md   : robotics topology table (real numbers)
  - cube_topology.png           : 3-panel figure (choice point / episode stages / stacking graph)

Usage:
  python -m survey.cube_ogbench.analyze_cube_topology --dataset cube-double-play-v0
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "survey" / "cube_ogbench" / "results"
DATASET_URL = "https://rail.eecs.berkeley.edu/datasets/ogbench"
DATA_DIR = Path(os.path.expanduser("~/.ogbench/data"))

QPOS_ARM = 14
QPOS_CUBE = 7
CUBE_SIZE = 0.04      # cube edge / success tolerance
TABLE_MAX_Z = 0.045   # a cube resting on the table sits below this
LIFT_Z = 0.07         # above this the cube is clearly lifted / being manipulated


def ensure_dataset(dataset: str, use_val: bool = True) -> Path:
    """Download the (val) split if missing; return the local path."""
    fname = f"{dataset}-val.npz" if use_val else f"{dataset}.npz"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dst = DATA_DIR / fname
    if not dst.exists():
        url = f"{DATASET_URL}/{fname}"
        print(f"downloading {url} ...", flush=True)
        urllib.request.urlretrieve(url, dst)
    return dst


def cube_xyz(qpos: np.ndarray, i: int) -> np.ndarray:
    s = QPOS_ARM + i * QPOS_CUBE
    return qpos[:, s : s + 3]


def episode_slices(terminals: np.ndarray) -> list[tuple[int, int]]:
    ends = np.where(terminals)[0]
    starts = np.concatenate([[0], ends[:-1] + 1])
    return list(zip(starts.tolist(), ends.tolist()))


def stacked(top: np.ndarray, bot: np.ndarray) -> np.ndarray:
    """Boolean per-step: `top` cube resting on `bot` cube (bot on table)."""
    xy = np.linalg.norm(top[:, :2] - bot[:, :2], axis=1)
    dz = top[:, 2] - bot[:, 2]
    return (xy < 0.045) & (dz > 0.025) & (dz < 0.055) & (bot[:, 2] < TABLE_MAX_Z)


def analyse(qpos: np.ndarray, terminals: np.ndarray) -> dict[str, Any]:
    c = [cube_xyz(qpos, 0), cube_xyz(qpos, 1)]
    slices = episode_slices(terminals)
    n = len(slices)

    stack_eps = both_lift = 0
    dir_01 = dir_10 = 0            # cube0-on-cube1 / cube1-on-cube0
    first_moved = [0, 0]
    manipulated = [0, 0]
    stacked_step_frac = []
    best_ep = None                 # a representative episode for the timeline panel

    for (s, e) in slices:
        a, b = c[0][s : e + 1], c[1][s : e + 1]
        s01, s10 = stacked(a, b), stacked(b, a)
        if s01.any():
            dir_01 += 1
        if s10.any():
            dir_10 += 1
        if s01.any() or s10.any():
            stack_eps += 1
        stacked_step_frac.append(float((s01 | s10).mean()))

        lift0 = (a[:, 2] > LIFT_Z).any()
        lift1 = (b[:, 2] > LIFT_Z).any()
        manipulated[0] += int(lift0)
        manipulated[1] += int(lift1)
        if lift0 and lift1:
            both_lift += 1

        d0 = np.linalg.norm(a - a[0], axis=1)
        d1 = np.linalg.norm(b - b[0], axis=1)
        t0 = int(np.argmax(d0 > 0.03)) if (d0 > 0.03).any() else 10**9
        t1 = int(np.argmax(d1 > 0.03)) if (d1 > 0.03).any() else 10**9
        if t0 < t1:
            first_moved[0] += 1
        elif t1 < t0:
            first_moved[1] += 1

        # pick a nice timeline episode: has a stack AND both cubes lifted
        score = (s01 | s10).sum()
        if (s01.any() or s10.any()) and lift0 and lift1:
            if best_ep is None or score > best_ep["score"]:
                best_ep = {"s": s, "e": e, "score": int(score)}

    return {
        "num_episodes": n,
        "episode_len": int(slices[0][1] - slices[0][0] + 1),
        "stacking_episodes": stack_eps,
        "stacking_pct": round(100 * stack_eps / n, 1),
        "cube0_on_cube1_eps": dir_01,
        "cube1_on_cube0_eps": dir_10,
        "avg_stacked_step_pct": round(float(np.mean(stacked_step_frac)) * 100, 1),
        "both_cubes_lifted_eps": both_lift,
        "cube0_manipulated_eps": manipulated[0],
        "cube1_manipulated_eps": manipulated[1],
        "first_moved_cube0": first_moved[0],
        "first_moved_cube1": first_moved[1],
        "timeline_episode": best_ep,
    }


def draw(qpos: np.ndarray, stats: dict[str, Any], out_path: Path) -> None:
    fig = plt.figure(figsize=(16, 5.4))
    ax_a = fig.add_subplot(1, 3, 1)
    ax_b = fig.add_subplot(1, 3, 2)
    ax_c = fig.add_subplot(1, 3, 3)

    n = stats["num_episodes"]

    # ---- A: choice point — which cube to manipulate first ----
    ax_a.axis("off")
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 10)
    ax_a.set_title("A. Choice point\n(which cube to manipulate first)", fontsize=10)
    ax_a.scatter([5], [8.5], s=800, color="#444", zorder=3)
    ax_a.text(5, 8.5, "START", color="white", ha="center", va="center", fontsize=9, fontweight="bold")
    for x, lbl, cnt, col in [(2.4, "pick\ncube 0", stats["first_moved_cube0"], "#4c78a8"),
                             (7.6, "pick\ncube 1", stats["first_moved_cube1"], "#f58518")]:
        ax_a.annotate("", xy=(x, 4.6), xytext=(5 + (x - 5) * 0.15, 8.0),
                      arrowprops=dict(arrowstyle="-|>", lw=3, color=col))
        ax_a.scatter([x], [3.9], s=1500, color=col, zorder=3)
        ax_a.text(x, 3.9, lbl, color="white", ha="center", va="center", fontsize=9, fontweight="bold")
        ax_a.text(x, 2.4, f"{cnt}/{n} eps\n({100*cnt/n:.0f}%)", ha="center", fontsize=9, color="#333")
    ax_a.text(5, 0.6, f"both cubes get lifted in {stats['both_cubes_lifted_eps']}/{n} eps\n"
                      f"→ the pick order is a real decision", ha="center", fontsize=8, color="#555")

    # ---- B: one episode's cube-height timeline (task stages) ----
    ep = stats["timeline_episode"]
    s, e = ep["s"], ep["e"]
    z0 = cube_xyz(qpos, 0)[s : e + 1, 2]
    z1 = cube_xyz(qpos, 1)[s : e + 1, 2]
    t = np.arange(len(z0))
    ax_b.plot(t, z0, color="#4c78a8", lw=1.5, label="cube 0 height")
    ax_b.plot(t, z1, color="#f58518", lw=1.5, label="cube 1 height")
    ax_b.axhline(LIFT_Z, color="gray", ls="--", lw=0.8)
    ax_b.text(len(t) * 0.01, LIFT_Z + 0.005, "lift threshold", fontsize=7, color="gray")
    a = cube_xyz(qpos, 0)[s : e + 1]
    b = cube_xyz(qpos, 1)[s : e + 1]
    stack_mask = stacked(a, b) | stacked(b, a)
    ax_b.fill_between(t, 0, np.maximum(z0, z1), where=stack_mask, color="#54a24b",
                      alpha=0.25, label="stacked (one on other)")
    ax_b.set_title("B. One play episode — cube heights\n(lift → place → stack stages)", fontsize=10)
    ax_b.set_xlabel("step")
    ax_b.set_ylabel("cube z height (m)")
    ax_b.legend(loc="upper right", fontsize=7)

    # ---- C: stacking / precondition graph ----
    ax_c.axis("off")
    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(0, 10)
    ax_c.set_title("C. Stacking / precondition graph\n(who ends up on top of whom)", fontsize=10)
    ax_c.scatter([2.5], [5], s=2600, color="#4c78a8", zorder=3)
    ax_c.text(2.5, 5, "cube 0", color="white", ha="center", va="center", fontsize=10, fontweight="bold")
    ax_c.scatter([7.5], [5], s=2600, color="#f58518", zorder=3)
    ax_c.text(7.5, 5, "cube 1", color="white", ha="center", va="center", fontsize=10, fontweight="bold")
    ax_c.annotate("", xy=(6.7, 5.8), xytext=(3.3, 5.8),
                  arrowprops=dict(arrowstyle="-|>", lw=3, color="#4c78a8",
                                  connectionstyle="arc3,rad=0.3"))
    ax_c.text(5, 7.4, f"cube0 on cube1\n{stats['cube0_on_cube1_eps']}/{n} eps",
              ha="center", fontsize=9, color="#4c78a8", fontweight="bold")
    ax_c.annotate("", xy=(3.3, 4.2), xytext=(6.7, 4.2),
                  arrowprops=dict(arrowstyle="-|>", lw=3, color="#f58518",
                                  connectionstyle="arc3,rad=0.3"))
    ax_c.text(5, 2.4, f"cube1 on cube0\n{stats['cube1_on_cube0_eps']}/{n} eps",
              ha="center", fontsize=9, color="#f58518", fontweight="bold")
    ax_c.text(5, 0.6, f"{stats['stacking_pct']:.0f}% of play episodes contain a stack;\n"
                      f"both orders occur → precondition, not a fixed order", ha="center",
              fontsize=8, color="#555")

    fig.suptitle("OGBench cube-double — manipulation topology (task structure, not a metric map)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_table(dataset: str, stats: dict[str, Any], path: Path) -> None:
    n = stats["num_episodes"]
    md = f"""# OGBench cube benchmark — manipulation topology (robotics expansion)

Source: OGBench `{dataset}` (val split, {n} play episodes × {stats['episode_len']} steps).
State: qpos[0:14]=arm, cube i xyz = qpos[:, 14+i*7 : 14+i*7+3]. Action: 5-D continuous
(end-effector Δxyz + wrist + gripper). Observation: 37-D state (no image in this dataset).

Topology here is **task structure**, not a physical map: which cube to act on, the
object-on-object stacking (precondition), and the pick→place→stack skill sequence.

| Candidate | Observation cue | Action / option choices | Why topology (choice changes future)? | Valid / invalid | How to visualize | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| **Choice point — which cube** | cube xyz + gripper state | pick cube 0 vs cube 1 | the cube you grasp first sets the rest of the plan | valid = the cube the goal needs first; invalid = the other (must be undone) | choice-point graph (Panel A) | first-moved: cube0 {stats['first_moved_cube0']}/{n}, cube1 {stats['first_moved_cube1']}/{n}; both lifted in {stats['both_cubes_lifted_eps']}/{n} |
| **Precondition / stacking** | relative cube heights & xy | place-on-table vs stack-on-other | to stack, the base cube must be placed first — an ordering constraint | valid = base before top; invalid = stack before base is set | object-on-object graph (Panel C) | {stats['stacking_pct']:.0f}% of episodes stack; cube0-on-1 in {stats['cube0_on_cube1_eps']}, cube1-on-0 in {stats['cube1_on_cube0_eps']} |
| **Skill / task-stage** | gripper open/close + cube z | reach→grasp→lift→transport→place | each episode is a sequence of pick-place skills; stage order defines the task | valid = correct stage order; invalid = e.g. release before transport | cube-height timeline (Panel B) | avg {stats['avg_stacked_step_pct']}% of steps in a stacked config; both cubes manipulated ({stats['cube0_manipulated_eps']}, {stats['cube1_manipulated_eps']} eps) |

## Honest framing
- This is **play** (task-agnostic) data, so it explores **both** stacking orders (cube0-on-1 ≈ cube1-on-0). A specific goal-conditioned task fixes one order → then the precondition graph is directed. So the *dataset* shows the space of options; a *task* selects a valid branch through it.
- "cube" topology is **task/precondition topology**, distinct from R2R/ObjectNav route topology — kept separate per the assignment (dataset vs task vs representation topology).
- Figure: `results/cube_topology.png`. Larger stacking towers (precondition chains) live in `cube-triple` / `cube-quadruple` (1–1.9 GB), not downloaded here.
"""
    path.write_text(md)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cube-double-play-v0")
    parser.add_argument("--use-train", action="store_true",
                        help="use the full train split instead of the small val split")
    args = parser.parse_args()

    path = ensure_dataset(args.dataset, use_val=not args.use_train)
    data = np.load(path)
    qpos, terminals = data["qpos"], data["terminals"]
    print(f"loaded {args.dataset}: {len(qpos)} transitions")

    stats = analyse(qpos, terminals)
    print(f"  {stats['num_episodes']} eps | stacking {stats['stacking_pct']}% "
          f"(0-on-1 {stats['cube0_on_cube1_eps']}, 1-on-0 {stats['cube1_on_cube0_eps']}) | "
          f"first-move {stats['first_moved_cube0']}/{stats['first_moved_cube1']}")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    save = {k: v for k, v in stats.items()}
    (RESULT_ROOT / "cube_topology.json").write_text(json.dumps(save, indent=2) + "\n")
    write_table(args.dataset, stats, RESULT_ROOT / "cube_branch_candidates.md")
    draw(qpos, stats, RESULT_ROOT / "cube_topology.png")
    print(f"  wrote cube_topology.json / cube_branch_candidates.md / cube_topology.png")


if __name__ == "__main__":
    main()
