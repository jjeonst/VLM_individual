"""Week 2 (add) — Decision junctions in HM3D ObjectNav recovered positions.

A junction/fork is a location where trajectories ARRIVE from a common direction
and then SPLIT into different outgoing directions (→ different regions/goals) —
exactly the R2R branch point ("go right → goal A, go down → goal B") but recovered
in ObjectNav. This differs from a bottleneck (many pass, same direction): here the
outgoing choice diverges, so the point is decision-relevant.

Detection (on positions from objectnav_replay_positions.py):
  - quantise floor plane into ``CELL`` m cells; for each trajectory passing a cell,
    record its incoming bearing (~LOOK m before) and outgoing bearing (~LOOK m after).
  - within a cell, group passers by incoming bearing; if one incoming group (>= KMIN
    trajectories) splits into >= 2 outgoing branches separated by >= MIN_SEP degrees,
    the cell is a fork. Score prefers clean 2–3 way forks with balanced branches.

Outputs (results/objectnav_branching/train/):
  - junctions.json            : ranked junctions + the chosen example's branches
  - objectnav_junction.png    : the chosen fork drawn (trajectories coloured by branch)

Usage:
  python -m survey.week2.objectnav_junctions --split train
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
POS = REPO / "survey/week2/results/objectnav_branching/positions"
RESULT = REPO / "survey/week2/results/objectnav_branching"

CELL = 0.5
LOOK = 1.5           # metres before/after the cell to measure bearings
KMIN = 6             # min trajectories from a common incoming direction
MIN_SEP = 80         # min angular separation (deg) between outgoing branches
BRANCH_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]


def _bearing(p, q):
    return math.atan2(q[1] - p[1], q[0] - p[0])


def _arc(xz):
    seg = np.sqrt((np.diff(xz, axis=0) ** 2).sum(1))
    return np.concatenate([[0.0], np.cumsum(seg)])


def scene_junctions(trajs: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    # cell -> list of (ep, t, in_bearing, out_bearing)
    cell_io: dict[tuple, list] = defaultdict(list)
    for ep, pos in trajs.items():
        xz = pos[:, [0, 2]]
        cum = _arc(xz)
        seen = set()
        for t in range(len(xz)):
            c = (int(math.floor(xz[t, 0] / CELL)), int(math.floor(xz[t, 1] / CELL)))
            if c in seen:
                continue
            seen.add(c)
            t0, t2 = t, t
            while t0 > 0 and cum[t] - cum[t0] < LOOK:
                t0 -= 1
            while t2 < len(xz) - 1 and cum[t2] - cum[t] < LOOK:
                t2 += 1
            if t0 == t or t2 == t:
                continue
            cell_io[c].append((ep, t, _bearing(xz[t0], xz[t]), _bearing(xz[t], xz[t2])))

    results = []
    for c, lst in cell_io.items():
        if len({e for e, *_ in lst}) < KMIN:
            continue
        # group by incoming bearing (8 sectors)
        groups: dict[int, list] = defaultdict(list)
        for ep, t, ib, ob in lst:
            groups[int(((ib + math.pi) / (2 * math.pi)) * 8) % 8].append((ep, t, ob))
        for members in groups.values():
            if len({e for e, *_ in members}) < KMIN:
                continue
            # cluster outgoing bearings into 8 sectors
            ob_bins: dict[int, list] = defaultdict(list)
            for ep, t, ob in members:
                ob_bins[int(((ob + math.pi) / (2 * math.pi)) * 8) % 8].append((ep, t))
            branches = {b: m for b, m in ob_bins.items() if len(m) >= 2}
            if len(branches) < 2:
                continue
            angs = [b * 45 for b in branches]
            sep = max(min(abs(a - b), 360 - abs(a - b)) for a in angs for b in angs if a != b)
            if sep < MIN_SEP:
                continue
            counts = sorted((len(m) for m in branches.values()), reverse=True)
            balance = counts[1] / counts[0]  # 2nd branch / 1st branch
            results.append({
                "scene": None, "cell": list(c), "cell_xz": [(c[0] + 0.5) * CELL, (c[1] + 0.5) * CELL],
                "n_from_incoming": len(members), "n_branches": len(branches),
                "separation_deg": int(sep), "balance": round(balance, 2),
                "branches": {int(b): [(e, int(t)) for e, t in m] for b, m in branches.items()},
                # cleanliness score: prefer 2–3 balanced branches, moderate separation
                "score": len(members) * balance * (1.0 if len(branches) <= 3 else 0.5),
            })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def _idx_at_dist(cum, t, dist, forward=True):
    if forward:
        t2 = t
        while t2 < len(cum) - 1 and cum[t2] - cum[t] < dist:
            t2 += 1
        return t2
    t0 = t
    while t0 > 0 and cum[t] - cum[t0] < dist:
        t0 -= 1
    return t0


def draw_junction(scene: str, trajs: dict[str, np.ndarray], junc: dict[str, Any], out_path: Path):
    fig, ax = plt.subplots(figsize=(8.0, 7.5))
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, lw=0.3, alpha=0.3)
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")

    jx, jz = junc["cell_xz"]
    WIN = 3.0  # metres of path to show on each side of the junction
    branch_items = sorted(junc["branches"].items(), key=lambda kv: -len(kv[1]))[:3]

    # common arrival (incoming) drawn grey, from a few members
    for b, members in branch_items:
        for ep, t in members[:6]:
            pos = trajs[ep]; cum = _arc(pos[:, [0, 2]])
            t0 = _idx_at_dist(cum, t, WIN, forward=False)
            ax.plot(pos[t0:t + 1, 0], pos[t0:t + 1, 2], "-", color="#999", lw=1.0, alpha=0.5)

    # outgoing branches, coloured, short window after the junction
    for k, (b, members) in enumerate(branch_items):
        col = BRANCH_COLORS[k % len(BRANCH_COLORS)]
        for ep, t in members[:8]:
            pos = trajs[ep]; cum = _arc(pos[:, [0, 2]])
            t2 = _idx_at_dist(cum, t, WIN, forward=True)
            ax.plot(pos[t:t2 + 1, 0], pos[t:t2 + 1, 2], "-", color=col, lw=1.8, alpha=0.8)
            ax.scatter(*pos[t2, [0, 2]], color=col, s=30, zorder=5)  # branch endpoint
        ax.plot([], [], "-", color=col, lw=2.5, label=f"branch {k+1} ({len(members)} routes)")

    ax.scatter([jx], [jz], s=360, marker="P", color="white", edgecolor="black",
               linewidths=2, zorder=6)
    ax.annotate("junction\n(arrive → choose branch)", (jx, jz), textcoords="offset points",
                xytext=(8, 8), fontsize=10, fontweight="bold")
    ax.plot([], [], "-", color="#999", lw=1.5, label="common arrival (incoming)")

    ax.set_title(f"Decision junction — scene {scene}\n"
                 f"{junc['n_from_incoming']} routes arrive together, split into "
                 f"{junc['n_branches']} branches (sep {junc['separation_deg']}°). ±{WIN:.0f} m shown.")
    ax.legend(loc="best", fontsize=9)
    fig.suptitle("HM3D ObjectNav — decision junction (arrive together → diverge to different regions)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="train")
    ap.add_argument("--sep-min", type=int, default=80)
    ap.add_argument("--sep-max", type=int, default=140)
    args = ap.parse_args()

    pos_dir = POS / args.split
    all_junc = []
    scene_trajs = {}
    for npz in sorted(pos_dir.glob("*.npz")):
        scene = npz.stem
        trajs = {k: v for k, v in np.load(npz).items()}
        scene_trajs[scene] = trajs
        for j in scene_junctions(trajs):
            j["scene"] = scene
            all_junc.append(j)
    # prefer clean 2-branch forks with a moderate (T-like) separation, balanced
    clean = [j for j in all_junc if j["n_branches"] == 2
             and args.sep_min <= j["separation_deg"] <= args.sep_max]
    clean.sort(key=lambda r: (r["balance"], r["n_from_incoming"]), reverse=True)
    chosen = clean[0]
    print(f"junctions found: {len(all_junc)} (clean 2–3 branch: {len(clean)})")
    print(f"chosen: scene {chosen['scene']} cell {chosen['cell']} — "
          f"{chosen['n_from_incoming']} routes → {chosen['n_branches']} branches, "
          f"sep {chosen['separation_deg']}°, balance {chosen['balance']}")

    RESULT_dir = RESULT / args.split
    RESULT_dir.mkdir(parents=True, exist_ok=True)
    draw_junction(chosen["scene"], scene_trajs[chosen["scene"]], chosen,
                  RESULT_dir / "objectnav_junction.png")
    summary = {
        "num_junctions": len(all_junc),
        "num_clean_forks": len(clean),
        "chosen": {k: chosen[k] for k in
                   ["scene", "cell", "cell_xz", "n_from_incoming", "n_branches",
                    "separation_deg", "balance"]},
        "top_forks": [{k: j[k] for k in ["scene", "cell", "n_from_incoming", "n_branches",
                                         "separation_deg", "balance"]} for j in clean[:10]],
    }
    (RESULT_dir / "junctions.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {RESULT_dir}/objectnav_junction.png / junctions.json")


if __name__ == "__main__":
    main()
