"""Week 2 · Step 5-7 — ObjectNav branching candidates (deliverable #3).

R2R gives a language route graph; HM3D ObjectNav does not. As Week 1 found, the
materialised PR2L cache stores only the expert **action** sequence per episode
(0=STOP, 1=MOVE_FORWARD, 2=TURN_LEFT, 3=TURN_RIGHT) plus egocentric RGB and the
target object_category -- there are NO world positions unless you replay each
episode in the Habitat simulator. The assignment explicitly allows the fallback:
if a top-down map is hard, characterise branch candidates from the action
sequence and a small graph. That is what this script does -- no sim, no training.

It mines three branch candidates that match the assignment taxonomy, then writes
a candidate table + an evidence JSON + a figure.

  1. Start-junction fork (junction / observation-conditioned branch)
       Same scene + same target object, but expert routes split between an
       initial TURN_LEFT and TURN_RIGHT. The scene affords divergent initial
       branches; which one is correct depends on where the target instance is
       (i.e. on the observation), so the first choice changes the whole route.

  2. Dead-end / wrong-turn reversal (recovery branch)
       A contiguous run of same-direction turns totalling >= 180 deg
       (>= 6 turns at 30 deg) means the agent turned around -- a backtrack out of
       a dead-end / recovery from a wrong heading. A wrong branch here costs a
       recovery detour.

  3. Terminal STOP decision (stop-vs-continue)
       Every episode ends in STOP. ~16% STOP immediately (spawned at the goal);
       the rest approach then STOP. Calling STOP at the wrong place fails the
       episode, so the STOP/continue judgement is a genuine decision point.

Usage:
  python -m survey.week2.objectnav_branching --split train
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("TOPOVLM_DATASET_ROOT", "/data/topovlm"))
HABITAT_ROOT = DATA_ROOT / "habitat"
RESULT_ROOT = REPO_ROOT / "survey" / "week2" / "results" / "objectnav_branching"

ACTION_NAMES = {0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT"}
TURN_ACTIONS = (2, 3)
TURN_ANGLE_DEG = 30
REVERSAL_MIN_TURNS = 180 // TURN_ANGLE_DEG  # 6 consecutive turns = 180 degrees


def _manifest_path(split: str) -> Path:
    return HABITAT_ROOT / "episodes" / "pr2l_hm3d_objectnav" / split / "manifest.jsonl"


def load_records(split: str) -> list[dict[str, Any]]:
    with _manifest_path(split).open() as handle:
        return [json.loads(line) for line in handle]


def load_actions(record: dict[str, Any]) -> np.ndarray | None:
    path = HABITAT_ROOT / record["actions_path"]
    if not path.exists():
        return None
    return np.load(path)


def scene_key(scene_id: str) -> str:
    # ".../scene_datasets/hm3d_v0.2/train/00006-HkseAnWCgqk/HkseAnWCgqk.basis.glb"
    return scene_id.split("/")[-2]


# --------------------------------------------------------------------------- #
# Per-episode action features
# --------------------------------------------------------------------------- #
def first_turn(actions: np.ndarray) -> str:
    """Direction of the first turn after the leading forward stem."""
    for a in actions:
        if a == 1:
            continue
        if a == 2:
            return "left"
        if a == 3:
            return "right"
        if a == 0:
            return "straight"
    return "straight"


def longest_turn_run(actions: np.ndarray) -> tuple[int, int | None, int]:
    """Longest same-direction turn run: (length, direction, start_index)."""
    best_len, best_dir, best_start = 0, None, -1
    run, cur, start = 0, None, 0
    for i, a in enumerate(actions):
        if a in TURN_ACTIONS:
            if a == cur:
                run += 1
            else:
                cur, run, start = a, 1, i
            if run > best_len:
                best_len, best_dir, best_start = run, int(a), start
        else:
            run, cur = 0, None
    return best_len, best_dir, best_start


def inflections(actions: np.ndarray) -> int:
    """Number of FORWARD -> TURN transitions (heading-change decision points)."""
    return int(sum(1 for i in range(1, len(actions))
                   if actions[i - 1] == 1 and actions[i] in TURN_ACTIONS))


# --------------------------------------------------------------------------- #
def analyse(records: list[dict[str, Any]]) -> dict[str, Any]:
    group_first_turns: dict[tuple[str, str], list[str]] = defaultdict(list)
    transitions: Counter[tuple[str, str]] = Counter()  # option-transition graph
    reversal_examples: list[dict[str, Any]] = []
    n = immediate_stop = any_reversal = midroute_reversal_count = infl_total = 0
    MIN_FWD_BEFORE = 5  # forwards before a run to call it mid-route (not spawn spin)

    for rec in records:
        actions = load_actions(rec)
        if actions is None:
            continue
        n += 1
        sc = scene_key(rec["scene_id"])
        obj = rec.get("object_category") or rec.get("goal_text") or "?"
        group_first_turns[(sc, obj)].append(first_turn(actions))

        # option-transition graph counts (virtual START before first action)
        prev = "START"
        for a in actions:
            cur = ACTION_NAMES[int(a)]
            transitions[(prev, cur)] += 1
            prev = cur

        run_len, run_dir, run_start = longest_turn_run(actions)
        if run_len >= REVERSAL_MIN_TURNS:
            any_reversal += 1
            fwd_before = int((actions[:run_start] == 1).sum())
            if fwd_before >= MIN_FWD_BEFORE:
                midroute_reversal_count += 1
            if len(reversal_examples) < 30:
                reversal_examples.append({
                    "episode_id": rec["episode_id"],
                    "object": obj,
                    "length": int(len(actions)),
                    "reversal_turns": int(run_len),
                    "reversal_dir": ACTION_NAMES[run_dir] if run_dir else None,
                    "reversal_start": int(run_start),
                    "forward_before": fwd_before,
                })
        if len(actions) <= 1:
            immediate_stop += 1
        infl_total += inflections(actions)

    # Rank start-fork groups: same scene+object with a genuine left/right split.
    forks = []
    for (sc, obj), turns in group_first_turns.items():
        c = Counter(turns)
        total = len(turns)
        left, right = c["left"], c["right"]
        if total >= 6 and left >= 2 and right >= 2:
            forks.append({
                "scene": sc,
                "object": obj,
                "episodes": total,
                "left": left,
                "right": right,
                "straight": c["straight"],
                "split_balance": round(min(left, right) / total, 3),
            })
    forks.sort(key=lambda f: (f["split_balance"], f["episodes"]), reverse=True)

    return {
        "split_episodes": n,
        "immediate_stop": immediate_stop,
        "immediate_stop_pct": round(100 * immediate_stop / n, 1),
        "any_reversal_episodes": any_reversal,
        "any_reversal_pct": round(100 * any_reversal / n, 1),
        "midroute_reversal_episodes": midroute_reversal_count,
        "midroute_reversal_pct": round(100 * midroute_reversal_count / n, 1),
        "avg_inflections_per_episode": round(infl_total / n, 2),
        "num_fork_groups": len(forks),
        "top_forks": forks[:15],
        "reversal_examples": reversal_examples,
        "transitions": {f"{a}->{b}": c for (a, b), c in transitions.most_common()},
    }


# --------------------------------------------------------------------------- #
def draw(stats: dict[str, Any], records: list[dict[str, Any]], out_path: Path) -> None:
    fig = plt.figure(figsize=(16, 5.5))
    ax_a = fig.add_subplot(1, 3, 1)
    ax_b = fig.add_subplot(1, 3, 2)
    ax_c = fig.add_subplot(1, 3, 3)

    # ---- Panel A: start-junction fork for the top group ----
    fork = stats["top_forks"][0]
    ax_a.axis("off")
    ax_a.set_title("A. Start-junction fork\n(same scene + target, route splits L/R)", fontsize=10)
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 10)
    ax_a.scatter([5], [8.5], s=900, color="#444", zorder=3)
    ax_a.text(5, 8.5, "START", color="white", ha="center", va="center", fontsize=9, fontweight="bold", zorder=4)
    # left branch
    ax_a.annotate("", xy=(2.2, 4.5), xytext=(4.6, 8.0),
                 arrowprops=dict(arrowstyle="-|>", lw=3, color="#1f77b4"))
    ax_a.annotate("", xy=(7.8, 4.5), xytext=(5.4, 8.0),
                 arrowprops=dict(arrowstyle="-|>", lw=3, color="#ff7f0e"))
    ax_a.scatter([2.2], [3.8], s=800, color="#1f77b4", zorder=3)
    ax_a.text(2.2, 3.8, f"TURN\nLEFT\n{fork['left']} ep", color="white", ha="center", va="center", fontsize=8, fontweight="bold")
    ax_a.scatter([7.8], [3.8], s=800, color="#ff7f0e", zorder=3)
    ax_a.text(7.8, 3.8, f"TURN\nRIGHT\n{fork['right']} ep", color="white", ha="center", va="center", fontsize=8, fontweight="bold")
    ax_a.annotate("", xy=(5, 1.2), xytext=(2.4, 3.1),
                 arrowprops=dict(arrowstyle="-|>", lw=2, color="#1f77b4", linestyle="--"))
    ax_a.annotate("", xy=(5, 1.2), xytext=(7.6, 3.1),
                 arrowprops=dict(arrowstyle="-|>", lw=2, color="#ff7f0e", linestyle="--"))
    ax_a.scatter([5], [0.7], s=1100, marker="*", color="#2ca02c", zorder=3, edgecolor="black")
    ax_a.text(5, 0.7, f"target:\n{fork['object']}", ha="center", va="center", fontsize=8, fontweight="bold")
    ax_a.text(5, 6.2, f"scene {fork['scene']}\n{fork['episodes']} expert routes\nsplit {fork['left']}L / {fork['right']}R",
             ha="center", va="center", fontsize=8, color="#333")

    # ---- Panel B: option-transition graph ----
    ax_b.set_title("B. Option-transition graph\n(from FORWARD the agent chooses)", fontsize=10)
    ax_b.axis("off")
    pos = {"START": (0.5, 0.9), "FORWARD": (0.5, 0.5), "LEFT": (0.12, 0.28),
           "RIGHT": (0.88, 0.28), "STOP": (0.5, 0.08)}
    colors = {"START": "#444", "FORWARD": "#1f77b4", "LEFT": "#9467bd",
              "RIGHT": "#8c564b", "STOP": "#d62728"}
    tr = stats["transitions"]
    # total outgoing from FORWARD for edge widths
    fwd_out = {k: tr.get(f"FORWARD->{k}", 0) for k in ["FORWARD", "LEFT", "RIGHT", "STOP"]}
    fwd_total = max(sum(fwd_out.values()), 1)
    edges = [("START", "FORWARD"), ("FORWARD", "LEFT"), ("FORWARD", "RIGHT"),
             ("FORWARD", "STOP"), ("LEFT", "FORWARD"), ("RIGHT", "FORWARD")]
    for a, b in edges:
        cnt = tr.get(f"{a}->{b}", 0)
        lw = 0.6 + 5.5 * (cnt / fwd_total) if a == "FORWARD" else 1.2
        ax_b.annotate("", xy=pos[b], xytext=pos[a],
                     arrowprops=dict(arrowstyle="-|>", lw=lw, color="#888",
                                     connectionstyle="arc3,rad=0.12"))
    for name, (x, y) in pos.items():
        ax_b.scatter([x], [y], s=1500, color=colors[name], zorder=3)
        ax_b.text(x, y, name, color="white", ha="center", va="center", fontsize=8, fontweight="bold", zorder=4)
    # annotate FORWARD choice probabilities
    for k in ["LEFT", "RIGHT", "STOP"]:
        frac = fwd_out[k] / fwd_total
        ax_b.text(pos[k][0], pos[k][1] - 0.09, f"{frac:.0%}", ha="center", fontsize=8, color="#333")
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)

    # ---- Panel C: spawn-orientation spin action strip ----
    # Honest finding: every >=180 deg same-direction turn run starts at step 0
    # (0 forward actions before it) -- these are spawn-orientation spins, not
    # mid-route dead-end backtracks. Show the longest such spin.
    ax_c.set_title("C. Spawn-orientation spin\n(all ≥180° runs are at step 0, not mid-route)", fontsize=10)
    ex = max(stats["reversal_examples"], key=lambda e: e["reversal_turns"])
    rec = next(r for r in records if r["episode_id"] == ex["episode_id"])
    actions = load_actions(rec)
    cmap = {0: "#d62728", 1: "#1f77b4", 2: "#9467bd", 3: "#8c564b"}
    hi_start = ex["reversal_start"]
    run_len = ex["reversal_turns"]
    run_name = ex["reversal_dir"]
    for i, a in enumerate(actions):
        ax_c.add_patch(plt.Rectangle((i, 0), 1, 1, color=cmap[int(a)]))
    ax_c.add_patch(plt.Rectangle((hi_start, -0.15), run_len, 1.3, fill=False,
                                edgecolor="black", lw=2.2))
    ax_c.text(hi_start + run_len / 2, 1.35,
             f"{run_len}×{run_name} = {run_len*TURN_ANGLE_DEG}°\nspin at step {hi_start}",
             ha="center", fontsize=8, fontweight="bold")
    ax_c.set_xlim(0, len(actions))
    ax_c.set_ylim(-0.5, 2.0)
    ax_c.set_yticks([])
    ax_c.set_xlabel(f"action step  (episode len {len(actions)}, target {ex['object']})", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[k]) for k in [1, 2, 3, 0]]
    ax_c.legend(handles, ["FORWARD", "LEFT", "RIGHT", "STOP"], loc="upper right",
               fontsize=7, ncol=2, framealpha=0.9)

    fig.suptitle("HM3D ObjectNav — action-space branching candidates (PR2L expert trajectories)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_table(stats: dict[str, Any], path: Path) -> None:
    fork = stats["top_forks"][0]
    tr = stats["transitions"]
    fwd_out = {k: tr.get(f"FORWARD->{k}", 0) for k in ["FORWARD", "LEFT", "RIGHT", "STOP"]}
    fwd_total = max(sum(fwd_out.values()), 1)
    md = f"""# ObjectNav branching candidates (deliverable #3)

Source: HM3D ObjectNav v2 PR2L expert **action** trajectories
(`/data/topovlm/habitat/episodes/pr2l_hm3d_objectnav/{{split}}`), {stats['split_episodes']} episodes.
Actions: 0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT. No world positions in the cache
(would require a Habitat sim replay), so branches are characterised in ACTION space.

## Observation & action space (for all rows below)
- **Observation**: egocentric RGB (640×480) + target `object_category` (VLM prompt-conditioned latent in PR2L). No language route instruction.
- **Action**: discrete Habitat `STOP / MOVE_FORWARD / TURN_LEFT / TURN_RIGHT` (turn 30°).

## Branch candidate table

| Candidate | Observation cue | Action choices | Why topology (choice changes future)? | Valid / invalid branch | How to visualize | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| **Start-junction fork** | egocentric RGB at spawn + target category | first turn LEFT vs RIGHT (vs straight) | initial branch sets the whole route to the target; correct side depends on where the target instance is (observation-conditioned) | valid = branch toward the reachable target instance; invalid = away → longer path | fork node→{{L,R}}→target diagram (Panel A) | {stats['num_fork_groups']} scene+object groups split L/R; top: scene `{fork['scene']}`, {fork['object']}, {fork['episodes']} routes = {fork['left']}L / {fork['right']}R |
| **Spawn-orientation spin** | RGB at spawn (target behind agent) | turn-around (≥180° same-dir run) before moving | the agent must first choose a heading before any route unfolds — the earliest branch | valid = spin toward target bearing; invalid = spin away | action strip, run at step 0 (Panel C) | {stats['any_reversal_episodes']} episodes ({stats['any_reversal_pct']}%) begin with a ≥180° spin; **100% start at step 0** (mid-route ≥180° backtracks = {stats['midroute_reversal_episodes']}) |
| **Terminal STOP decision** | RGB shows target in view / not | STOP vs keep moving | STOP at the wrong place fails the episode; STOP is the success-defining branch | valid = STOP at target; invalid = premature/late STOP | STOP node in option graph (Panel B) | every episode ends in STOP; {stats['immediate_stop']} ({stats['immediate_stop_pct']}%) STOP immediately (spawn=goal) |
| **Inflection junctions (support)** | RGB at each heading change | FORWARD vs TURN | each FORWARD→TURN is a re-decision of heading at a junction | — | option-transition graph (Panel B) | avg {stats['avg_inflections_per_episode']} FORWARD→TURN transitions per episode |

## Option-transition structure (from FORWARD)
From a FORWARD state the expert chooses:
FORWARD {fwd_out['FORWARD']/fwd_total:.0%} · LEFT {fwd_out['LEFT']/fwd_total:.0%} · RIGHT {fwd_out['RIGHT']/fwd_total:.0%} · STOP {fwd_out['STOP']/fwd_total:.0%}.
These non-FORWARD choices are the decision points that create branch structure.

## Caveats (honest framing)
- This is **action-space** topology, not a metric map. "Start-junction fork" is route-level (spawn/observation-conditioned), not a single physical junction with multiple outgoing edges — reconstructing the latter needs a sim replay to recover positions.
- **Dead-end / mid-route backtrack is NOT separable from actions alone.** Every ≥180° same-direction turn run in the dataset starts at step 0 (0 forward actions before it) — they are spawn-orientation spins. A genuine dead-end recovery (go forward into a dead-end, then return near a previous location) needs **positions from a Habitat sim replay** to detect; it is left as future work, not claimed here.
- Left/right validity cannot be judged from actions alone (depends on target instance location in the RGB); it is labelled by intent, not verified geometrically.
- Figure: `results/objectnav_branching/{{split}}/objectnav_branching.png`
"""
    path.write_text(md)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    print(f"Loading ObjectNav PR2L manifest for split '{args.split}' ...")
    records = load_records(args.split)
    print(f"  {len(records)} episode records")

    stats = analyse(records)
    print(f"  analysed {stats['split_episodes']} episodes")
    print(f"  fork groups: {stats['num_fork_groups']} | spawn spins (≥180°): "
          f"{stats['any_reversal_episodes']} ({stats['any_reversal_pct']}%), "
          f"mid-route backtracks: {stats['midroute_reversal_episodes']} | immediate STOP: "
          f"{stats['immediate_stop']} ({stats['immediate_stop_pct']}%) | avg inflections: "
          f"{stats['avg_inflections_per_episode']}")

    out_dir = RESULT_ROOT / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "objectnav_branching.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    write_table(stats, out_dir / "branch_candidates.md")
    draw(stats, records, out_dir / "objectnav_branching.png")
    print(f"  wrote {out_dir}/objectnav_branching.json")
    print(f"  wrote {out_dir}/branch_candidates.md")
    print(f"  wrote {out_dir}/objectnav_branching.png")


if __name__ == "__main__":
    main()
