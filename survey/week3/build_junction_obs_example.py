"""Week 3 (rev) — Observation example for Obs-Probe 3 (branch choice from view+goal).

At a decision junction (from objectnav_junctions), show the agent's egocentric RGB
AT the junction and a few steps down each of the two branches. Given a target
object, only one branch leads to the goal — that is what the model must choose.
Coordinates are used only to pick frames; the model would see only the images.

Usage:
  python -m survey.week3.build_junction_obs_example
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from survey.week2.objectnav_junctions import scene_junctions, _arc, _idx_at_dist

REPO = Path(__file__).resolve().parents[2]
HAB = Path("/data/topovlm/habitat")
POS = REPO / "survey/week2/results/objectnav_branching/positions/train"
OUT = REPO / "survey/week3/results/probes"
SCENE = "00463-URjpCob8MGw"
CELL_XZ = (-2 * 0.5 + 0.25, 16 * 0.5 + 0.25)  # chosen junction cell (-2,16) centre


def rgb_path(ep):
    return HAB / "rgb" / "pr2l_hm3d_objectnav" / "train" / f"{ep}.npy"


def obj_of(ep):
    mf = HAB / "episodes" / "pr2l_hm3d_objectnav" / "train" / "manifest.jsonl"
    for line in mf.open():
        r = json.loads(line)
        if r["episode_id"] == ep:
            return r.get("object_category")
    return "?"


def frame(ep, t):
    a = np.load(rgb_path(ep), mmap_mode="r")
    return np.asarray(a[min(max(t, 0), a.shape[0] - 1)])


def main() -> None:
    trajs = {k: v for k, v in np.load(POS / f"{SCENE}.npz").items()}
    juncs = [j for j in scene_junctions(trajs) if j["n_branches"] == 2 and 80 <= j["separation_deg"] <= 140]
    juncs.sort(key=lambda r: (r["balance"], r["n_from_incoming"]), reverse=True)
    j = juncs[0]
    branches = sorted(j["branches"].items(), key=lambda kv: -len(kv[1]))[:2]

    # one representative episode per branch (that has RGB)
    def pick(members):
        for ep, t in members:
            if rgb_path(ep).exists():
                return ep, t
        return members[0]
    (ep1, t1), (ep2, t2) = pick(branches[0][1]), pick(branches[1][1])

    # step index ~2 m ahead along each branch
    def ahead(ep, t):
        cum = _arc(trajs[ep][:, [0, 2]])
        return _idx_at_dist(cum, t, 2.0, forward=True)

    obj1 = obj_of(ep1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    ax[0].imshow(frame(ep1, t1)); ax[0].axis("off")
    ax[0].set_title("① AT the junction (egocentric view)\n— agent must choose a branch", fontsize=9)
    ax[1].imshow(frame(ep1, ahead(ep1, t1))); ax[1].axis("off")
    ax[1].set_title(f"② Branch 1 view (what you'd see going branch 1)\ngold: leads to target '{obj1}'", fontsize=9)
    ax[2].imshow(frame(ep2, ahead(ep2, t2))); ax[2].axis("off")
    ax[2].set_title("③ Branch 2 view (the other branch)\n→ leads to a different region", fontsize=9)

    fig.suptitle(
        f"Obs-Probe 3 example (scene {SCENE}): at the junction ①, target = '{obj1}'. "
        f"Which branch leads to the goal — ② or ③?\n"
        f"Model sees only these egocentric views (no coordinates); gold = the branch the expert to '{obj1}' took.",
        fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "obs_branch_choice_00463.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    (OUT / "obs_branch_choice_00463.json").write_text(json.dumps({
        "scene": SCENE, "junction_cell_xz": list(CELL_XZ),
        "branch1_episode": ep1, "branch1_target": obj1, "branch2_episode": ep2,
        "question": f"At the junction, target='{obj1}'. Which branch view leads to the goal?",
        "gold": "branch 1 (the expert to the target took it)", "figure": str(out.relative_to(REPO)),
    }, indent=2) + "\n")
    print("wrote", out, "| target:", obj1, "| eps:", ep1[-6:], ep2[-6:])


if __name__ == "__main__":
    main()
