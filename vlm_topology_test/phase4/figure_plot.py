"""Draw the representation figure, and measure what it is supposed to show (Appendix H.2).

Figure 8 is an argument made in pictures: if prompting shapes the VLM's states around the task,
then states the model would call the same room should land together, and the states close to the
goal should land in the room the goal object lives in. The image-encoder row is the control --
the same frames, the same room labels, a representation that never went through the language
model -- and the paper's reading is that its points form one undifferentiated blob.

**The labels come from one place for both rows.** A vision backbone emits no text, so the boxes
over the lower row are drawn from the *same* frames' answers to "What room is this?" as the
upper row. That is what makes the two rows comparable: the grouping is held fixed and only the
coordinates change, so a difference between the rows is a difference between representations.

**The picture is scored, not just shown.** "Points are more clustered by room" is the kind of
claim a chosen viewing angle can manufacture, so the same grouping is also measured, with the
silhouette coefficient of the room labels in each panel's own 2-D coordinates. It runs from -1
to 1: near zero means the room labels tell you nothing about where a point sits, and higher
means points sit near others carrying the same label. Since the labels are identical between the
rows, the difference between their scores is attributable to the representation alone.

**Two colourings, because the paper's own definition contradicts its caption.** The caption says
the colour is the oracle's value and that "more yellow is better", and the text reads high-value
points as being in the target's room -- that is, close to the goal. But Appendix D.1's reward is
a terminal `+10*SPL` plus dense shaping equal to the drop in geodesic distance, and Appendix
D.2's discount is 0.99; computed that way the value *rises* with remaining distance (measured
+0.685 on these rollouts), because a rollout that starts far away collects more shaping on the
way in. Both are drawn: `distance`, which is what the caption's reading requires, and `return`,
which is what its definition literally says.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HABITAT_ROOT = Path("/data/topovlm/habitat")
EMBED_ROOT = HABITAT_ROOT / "figure_embeddings"
OUT_ROOT = HABITAT_ROOT / "figure_plots"

FIGURE_OBJECTS = ("toilet", "bed", "sofa")
TARGET_ROOM = {"toilet": "bathroom", "bed": "bedroom", "sofa": "living room"}
NEAR_METRES = 1.5                    # "close to the goal", for the summary table only
GAMMA = 0.99                         # Appendix D.2
STOP_REWARD = 10.0                   # Appendix D.1: +10 * SPL on stopping

# The vocabulary is drawn from the answers themselves rather than fixed in advance: the model
# writes sentences, not labels, and these are the rooms it actually named across 1,545 frames.
ROOMS = {
    "bathroom": ["bathroom", "restroom", "washroom", "powder room"],
    "bedroom": ["bedroom"],
    "living room": ["living room", "lounge", "sitting room", "family room", "living area"],
    "kitchen": ["kitchen"],
    "dining room": ["dining room", "dining area"],
    "hallway": ["hallway", "corridor", "foyer", "entryway"],
    "office": ["office"],
    "closet": ["closet"],
    "laundry": ["laundry", "utility room"],
    "garage": ["garage"],
    "outdoor": ["patio", "balcony", "terrace", "garden", "courtyard"],
    "stairway": ["stairway", "staircase", "stairwell", "stairs"],
}
UNLABELLED = "라벨없음"

# "A hallway or stairwell. It's not a bedroom." Rare -- two answers in 1,545 -- but a negated
# mention is the opposite of a label, and counting it as one would put a point in the wrong box.
NEGATION = re.compile(r"(not|isn't|n't) (a |an |the )?$")


def room_of(text: str) -> str:
    """The first room the answer names, ignoring ones it names only to deny."""
    haystack = " " + text.lower() + " "
    best, position = UNLABELLED, len(haystack) + 1
    for name, keys in ROOMS.items():
        for key in keys:
            index = haystack.find(key)
            if index < 0 or NEGATION.search(haystack[max(0, index - 12):index]):
                continue
            if index < position:
                best, position = name, index
    return best


def oracle_return(distances: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """The discounted return of the follower from each step, under Appendix D's reward."""
    walked = float(np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1).sum())
    spl = float(distances[0]) / max(float(distances[0]), walked, 1e-9)
    shaping = distances[:-1] - distances[1:]
    value = np.zeros(len(distances), dtype=np.float64)
    value[-1] = STOP_REWARD * spl
    for step in range(len(shaping) - 1, -1, -1):
        value[step] = shaping[step] + GAMMA * value[step + 1]
    return value


def load(condition: str) -> dict[str, dict]:
    """Every frame of every trajectory, grouped by target object."""
    root = EMBED_ROOT / condition
    rows = [json.loads(line) for line in (root / "manifest.jsonl").open()]
    labels_from = EMBED_ROOT / "room"          # both rows are grouped by the same answers
    grouped: dict[str, dict] = {
        name: {"mean": [], "distance": [], "value": [], "room": []} for name in FIGURE_OBJECTS}

    for row in rows:
        payload = np.load(root / f"{row['name']}.npz")
        mean = payload["mean"].astype(np.float32)
        mean = mean.reshape(len(mean), -1)      # [T, 2, 4096] -> [T, 8192]; [T, 2176] unchanged
        distances = payload["distances"].astype(np.float64)
        answers = json.loads((labels_from / f"{row['name']}.answers.json").read_text())

        bucket = grouped[row["object_category"]]
        bucket["mean"].append(mean)
        bucket["distance"].append(distances)
        bucket["value"].append(oracle_return(distances, payload["poses"].astype(np.float64)))
        bucket["room"].extend(room_of(text) for text in answers)

    for bucket in grouped.values():
        bucket["mean"] = np.concatenate(bucket["mean"])
        bucket["distance"] = np.concatenate(bucket["distance"])
        bucket["value"] = np.concatenate(bucket["value"])
        bucket["room"] = np.asarray(bucket["room"])
    return grouped


def project(mean: np.ndarray) -> np.ndarray:
    """Two dimensions of maximum variance, fitted on this panel alone.

    Appendix H.1: "principal component analysis (PCA) on the tokenwise average of all embeddings
    for each observation, thereby projecting the embeddings to a 2D space with maximum variance."
    Each panel of Figure 8 carries its own PC1/PC2 axes, so each is fitted separately.
    """
    centred = mean - mean.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(centred, full_matrices=False)
    return centred @ components[:2].T


def silhouette(points: np.ndarray, labels: np.ndarray, minimum: int = 10) -> float:
    """How well the room labels explain where points sit, in this panel's own coordinates.

    Rooms named fewer than `minimum` times are left out: a cluster of three points has a
    silhouette dominated by which three they are. Returns NaN when fewer than two rooms survive.
    """
    counts = Counter(labels)
    keep = np.array([counts[label] >= minimum and label != UNLABELLED for label in labels])
    points, labels = points[keep], labels[keep]
    names = sorted(set(labels))
    if len(names) < 2:
        return float("nan")

    scores = np.zeros(len(points))
    gaps = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    for index, label in enumerate(labels):
        same = labels == label
        same[index] = False
        if not same.any():
            continue
        inside = gaps[index, same].mean()
        outside = min(gaps[index, labels == other].mean()
                      for other in names if other != label)
        scores[index] = (outside - inside) / max(inside, outside, 1e-12)
    return float(scores.mean())


def target_separation(points: np.ndarray, labels: np.ndarray, target: str) -> float:
    """How far the target object's room sits from every other labelled room, as one number.

    The plain silhouette scores *all* rooms against each other, which is stricter than the claim
    being tested: the paper does not say kitchens separate from hallways, it says the states near
    the goal group with the room the goal object lives in. So this collapses the question to two
    classes -- the target's room against everything else that carried a label -- and reports the
    silhouette of that split. Same scale, same reading: 0 means the two are interleaved.
    """
    keep = labels != UNLABELLED
    points, labels = points[keep], labels[keep]
    inside = labels == target
    if inside.sum() < 10 or (~inside).sum() < 10:
        return float("nan")
    gaps = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    scores = np.zeros(len(points))
    for index in range(len(points)):
        same = inside == inside[index]
        same[index] = False
        if not same.any():
            continue
        near = gaps[index, same].mean()
        far = gaps[index, ~(inside == inside[index])].mean()
        scores[index] = (far - near) / max(near, far, 1e-12)
    return float(scores.mean())


def draw(conditions: list[str], colour: str, out: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    # Every glyph in the figure has to render on a machine without a CJK font, so the plot's own
    # text is English. The report printed to the console is not subject to that.
    import matplotlib.pyplot as plt

    data = {name: load(name) for name in conditions}
    rows, columns = len(conditions), len(FIGURE_OBJECTS)
    figure, axes = plt.subplots(rows, columns, figsize=(5.4 * columns, 4.8 * rows),
                                squeeze=False)
    report: dict = {}

    for row, condition in enumerate(conditions):
        for column, target in enumerate(FIGURE_OBJECTS):
            bucket = data[condition][target]
            points = project(bucket["mean"])
            shade = bucket["distance"] if colour == "distance" else bucket["value"]
            axis = axes[row][column]

            # Yellow is the good end in both colourings, which for distance means reversing it.
            handle = axis.scatter(points[:, 0], points[:, 1],
                                  c=-shade if colour == "distance" else shade,
                                  s=9, cmap="viridis", alpha=0.75, linewidths=0)
            bar = figure.colorbar(handle, ax=axis, fraction=0.046, pad=0.03)
            bar.set_label("geodesic distance to goal (m, reversed)" if colour == "distance"
                          else f"oracle discounted return (gamma={GAMMA})", fontsize=8)

            counts = Counter(bucket["room"])
            for name, count in counts.most_common():
                if count < 10 or name == UNLABELLED:
                    continue
                inside = bucket["room"] == name
                centre = points[inside].mean(axis=0)
                spread = points[inside].std(axis=0)
                edge = "crimson" if name == TARGET_ROOM[target] else "0.35"
                axis.add_patch(plt.Rectangle(centre - spread, 2 * spread[0], 2 * spread[1],
                                             fill=False, edgecolor=edge,
                                             linewidth=2.0 if edge == "crimson" else 1.0))
                axis.annotate(f"{name} ({count})", centre + spread * [0, 1], fontsize=7,
                              color=edge, ha="center", va="bottom")

            score = silhouette(points, bucket["room"])
            focus = target_separation(points, bucket["room"], TARGET_ROOM[target])
            near = bucket["distance"] < NEAR_METRES
            hit = float((bucket["room"][near] == TARGET_ROOM[target]).mean()) if near.any() else float("nan")
            report[f"{condition}/{target}"] = {
                "frames": int(len(points)), "silhouette": score,
                "target_room_separation": focus,
                f"목표방비율_{NEAR_METRES}m이내": hit,
                "rooms": {k: v for k, v in counts.most_common() if v >= 10}}

            axis.set_title(f"find {target} — {condition} reps\n"
                           f"all rooms {score:+.3f} | {TARGET_ROOM[target]} vs rest "
                           f"{focus:+.3f} | within {NEAR_METRES}m {100 * hit:.0f}%",
                           fontsize=10)
            axis.set_xlabel("PC1", fontsize=8)
            axis.set_ylabel("PC2", fontsize=8)

    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=140)
    plt.close(figure)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upper", default="cot", choices=["cot", "room"],
                        help="cot = the main text's condition; room = Appendix D's prompt")
    parser.add_argument("--colour", default="both", choices=["distance", "value", "both"])
    args = parser.parse_args()

    conditions = [args.upper, "image"]
    wanted = ["distance", "value"] if args.colour == "both" else [args.colour]
    report = {}
    for colour in wanted:
        out = OUT_ROOT / f"figure_{args.upper}_{colour}.png"
        report[colour] = draw(conditions, colour, out)
        print(f"[fig-plot] {out} 저장", flush=True)

    first = report[wanted[0]]
    print(f"\n{'패널':22s} {'프레임':>7s} {'전체방':>8s} {'목표방대나머지':>14s} {'가까움':>8s}")
    for key, values in first.items():
        print(f"{key:22s} {values['frames']:7d} {values['silhouette']:+8.3f} "
              f"{values['target_room_separation']:+14.3f} "
              f"{100 * values[f'목표방비율_{NEAR_METRES}m이내']:7.0f}%")
    (OUT_ROOT / f"report_{args.upper}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
