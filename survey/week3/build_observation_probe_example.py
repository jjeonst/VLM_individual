"""Week 3 (rev) — Observation-based probe example: same doorway, different routes.

The coordinate probes reduce to geometry. To test genuine topology understanding
we ground the question in the agent's EGOCENTRIC RGB: "do these first-person views
depict the same doorway/place?" (place recognition), with the gold coming from the
coordinate oracle (Week 2 bottleneck cell). Coordinates are used only to SELECT
frames; the model would see only the images.

This proves the observation-based benchmark is constructible: it pulls real RGB
frames from the PR2L ObjectNav cache at the bottleneck cell of scene 00440
(the doorway 80% of routes pass through) across different episodes, plus a
distractor frame from elsewhere.

Usage:
  python -m survey.week3.build_observation_probe_example
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
HAB = Path("/data/topovlm/habitat")
POS_DIR = REPO / "survey/week2/results/objectnav_branching/positions/train"
SPATIAL = REPO / "survey/week2/results/objectnav_branching/train/spatial_branch.json"
OUT = REPO / "survey/week3/results/probes"
CELL = 0.5


def rgb_path(episode_id: str) -> Path:
    return HAB / "rgb" / "pr2l_hm3d_objectnav" / "train" / f"{episode_id}.npy"


def main() -> None:
    stats = json.loads(SPATIAL.read_text())
    scene = stats["best_bottleneck_scene"]            # 00440-...
    cx, cz = stats["best_scene_top_cells"][0]["xz"]   # bottleneck centre
    frac = stats["best_scene_top_cells"][0]["fraction"]
    trajs = dict(np.load(POS_DIR / f"{scene}.npz"))

    # For each episode: closest approach to the bottleneck centre.
    passes = []
    for ep, pos in trajs.items():
        xz = pos[:, [0, 2]]
        d = np.hypot(xz[:, 0] - cx, xz[:, 1] - cz)
        t = int(np.argmin(d))
        if d[t] < CELL and rgb_path(ep).exists():
            passes.append((float(d[t]), ep, t, pos))
    passes.sort(key=lambda x: x[0])

    # Pick 3 distinct episodes that clearly pass the doorway.
    chosen = passes[:3]
    # Distractor: a frame far from the doorway (from a 4th episode's start).
    far_ep, far_pos = passes[5][1], passes[5][3]
    far_t = int(np.argmax(np.hypot(far_pos[:, 0] - cx, far_pos[:, 2] - cz)))

    def load_frame(ep, t):
        arr = np.load(rgb_path(ep), mmap_mode="r")
        return np.asarray(arr[min(max(t, 0), arr.shape[0] - 1)])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for k, (dist, ep, t, _) in enumerate(chosen):
        # a few steps BEFORE closest approach -> connector tends to be ahead in view
        axes[k].imshow(load_frame(ep, t - 4))
        axes[k].set_title(f"route {k+1} · ep …{ep[-6:]}\n(at the shared junction, approaching from its own room)", fontsize=8.5)
        axes[k].axis("off")
    axes[3].imshow(load_frame(far_ep, far_t))
    axes[3].set_title(f"distractor · ep …{far_ep[-6:]}\n(elsewhere in the same scene)", fontsize=8.5)
    axes[3].axis("off")

    fig.suptitle(
        f"Observation grounding is available (scene {scene}): egocentric RGB of 3 routes passing the SAME "
        f"bottleneck junction ({frac:.0%} of routes),\neach approaching from a different room so the views differ — "
        f"+ a distractor elsewhere. A coordinate solver calls 'same cell = same place' trivially;\n"
        f"a VLM must recognise the shared connector across genuinely different first-person views. Model sees only images.",
        fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    OUT.mkdir(parents=True, exist_ok=True)
    out_png = OUT / "obs_same_place_00440.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)

    meta = {
        "scene": scene, "bottleneck_xz": [cx, cz], "traffic_fraction": frac,
        "same_place_episodes": [ep for _, ep, _, _ in chosen],
        "distractor_episode": far_ep,
        "question": "Which of these egocentric views depict the SAME place (doorway)?",
        "gold": "views 1-3 = same doorway; view 4 = different location",
        "figure": str(out_png.relative_to(REPO)),
    }
    (OUT / "obs_same_place_00440.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", out_png)
    print("same-place episodes:", meta["same_place_episodes"], "| distractor:", far_ep[-6:])


if __name__ == "__main__":
    main()
