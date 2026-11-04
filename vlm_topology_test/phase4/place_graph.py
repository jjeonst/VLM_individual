"""The place graph of TopoPrompt (design document, Appendix F): nodes laid along a walked path.

A node is dropped wherever the agent stands at least `NODE_SPACING` from every node already on
its floor, so a straight corridor gets one every 1.5 m and a room walked in circles keeps
reusing the same few. The current node is the nearest one. A step is a *revisit* when that
nearest node is close and was made long ago -- more than `REVISIT_EXCLUDE` nodes back -- which
is what "You have been here before" (slot S2) says.

This module is the one place those rules live. The LSTM memory probe uses it to label the
demonstrations, and `describe` writes the prefix of conditions C and D (and the slot vector of
condition B) from the same graph, so the label a probe is scored on and the sentence a model is
shown cannot drift apart.

**The prefix** (design document, 4.4 and Appendix F). Each node carries the room name the VLM
gave the frame that made it (`ROOM_QUESTION`, mapped onto `ROOMS` by `normalize_room`). The
visited nodes, in order, fall into runs of one name -- *room segments* -- and slot S1 lists the
segments before the current one, most recent first, skipping the current room's name, `other`
and names already listed, up to `MAX_ROOMS`, each with the side its last node lies on now.
Slot S2 is the revisit flag above.

    You passed: kitchen behind you, hallway to your left. You have been here before.

Positions here are absolute simulator coordinates `(x, height, z)`. The agent itself will only
ever be given its 2-D episodic GPS; the height is used so that two floors are not merged into
one node, which is right for a label and is the open question (design document, section 9)
for the agent's own graph.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np

NODE_SPACING = 1.5      # metres between nodes
REVISIT_RADIUS = 1.0    # metres: how close the nearest old node must be
REVISIT_EXCLUDE = 6     # the most recent nodes, which being near is not a revisit
FLOOR_GAP = 1.0         # metres of height that separate two floors
MAX_ROOMS = 3           # slot S1 lists at most this many rooms

ROOM_QUESTION = "What room is this? Answer in one word."
# Canonical name -> the answers mapped onto it (design document, 4.3). Anything else is `other`.
ROOMS = {
    "bathroom": ("bathroom", "restroom", "washroom", "toilet"),
    "bedroom": ("bedroom", "guest room"),
    "kitchen": ("kitchen", "pantry"),
    "living room": ("living room", "lounge", "family room", "den"),
    "dining room": ("dining room", "dining"),
    "hallway": ("hallway", "hall", "corridor", "entryway", "foyer"),
    "stairs": ("stairs", "staircase", "stairway", "stairwell"),
    "office": ("office", "study"),
    "laundry room": ("laundry", "laundry room", "utility room"),
}
ROOM_NAMES = tuple(ROOMS)                     # the nine named rooms; `other` is not listed
# Index order follows `quadrant`: 0 ahead, 1 left (+90 deg), 2 behind, 3 right (-90 deg).
DIRECTIONS = ("ahead", "to your left", "behind you", "to your right")


@dataclass
class PlaceGraph:
    """Per-step view of the graph as it stood at that step."""

    node_position: np.ndarray   # [N, 3] where each node was made
    node_step: np.ndarray       # [N] the step that made it
    current: np.ndarray         # [T] index of the nearest node at each step
    distance: np.ndarray        # [T] metres to that node
    count: np.ndarray           # [T] nodes existing at each step, after this step's own
    revisit: np.ndarray         # [T] bool, slot S2


def build(position: np.ndarray, spacing: float = NODE_SPACING,
          radius: float = REVISIT_RADIUS, exclude: int = REVISIT_EXCLUDE,
          floor_gap: float = FLOOR_GAP) -> PlaceGraph:
    """Walk the path once and record the graph at every step. Causal by construction."""
    position = np.asarray(position, dtype=np.float64)
    steps = len(position)
    nodes: list[np.ndarray] = []
    made: list[int] = []
    current = np.zeros(steps, dtype=np.int64)
    distance = np.zeros(steps)
    count = np.zeros(steps, dtype=np.int64)
    revisit = np.zeros(steps, dtype=bool)

    for t in range(steps):
        here = position[t]
        if nodes:
            stack = np.asarray(nodes)
            flat = np.hypot(stack[:, 0] - here[0], stack[:, 2] - here[2])
            same_floor = np.abs(stack[:, 1] - here[1]) < floor_gap
            flat = np.where(same_floor, flat, np.inf)
        else:
            flat = np.empty(0)
        if not len(flat) or flat.min() >= spacing:
            nodes.append(here.copy())
            made.append(t)
            flat = np.append(flat, 0.0)
        nearest = int(np.argmin(flat))
        current[t], distance[t], count[t] = nearest, flat[nearest], len(nodes)
        revisit[t] = distance[t] <= radius and nearest < len(nodes) - exclude

    return PlaceGraph(np.asarray(nodes), np.asarray(made), current, distance, count, revisit)


def relative_bearing(position: np.ndarray, yaw: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Angle of `target` seen from each `position`: 0 straight ahead, positive to the left.

    Habitat's agent faces (-sin yaw, -cos yaw) in the (x, z) plane and TURN_LEFT adds +30 deg
    of yaw -- both measured on the demonstrations (every forward step within 0 deg of that
    direction, every left turn +30 deg) -- so its left is (-cos yaw, sin yaw).

    An earlier version measured from the agent's right instead (it borrowed the rotation in
    `dataset.episodic_pose`, whose `forward` is the local x axis). Its four quadrants were the
    same four sectors under other indices, so the LSTM probe's `past30` result does not change;
    a sentence saying "to your left" does.
    """
    delta = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)
    ahead = -delta[..., 0] * np.sin(yaw) - delta[..., 2] * np.cos(yaw)
    left = -delta[..., 0] * np.cos(yaw) + delta[..., 2] * np.sin(yaw)
    return np.arctan2(left, ahead)


def quadrant(angle: np.ndarray) -> np.ndarray:
    """0 ahead (|a| <= 45 deg), 1 left, 2 behind, 3 right -- the order of `DIRECTIONS`."""
    shifted = (np.asarray(angle) + np.pi / 4) % (2 * np.pi)
    return (shifted // (np.pi / 2)).astype(np.int64)


def normalize_room(answer: str) -> str:
    """The VLM's free answer to `ROOM_QUESTION` -> one of `ROOMS` or `other`.

    Longer synonyms are tried first so that "laundry room" is not read as a bare "room" match
    and "dining room" not as "dining" -- harmless here, but it keeps the table order-free.
    """
    text = " " + re.sub(r"[^a-z ]+", " ", answer.lower()) + " "
    text = re.sub(r" +", " ", text)
    pairs = sorted(((syn, name) for name, syns in ROOMS.items() for syn in syns),
                   key=lambda pair: -len(pair[0]))
    for synonym, name in pairs:
        if f" {synonym} " in text:
            return name
    return "other"


@dataclass(frozen=True)
class Prefix:
    """What the prefix says at one step: slot S1's (room, direction) pairs and slot S2."""

    rooms: tuple[tuple[str, str], ...]
    revisit: bool

    @property
    def text(self) -> str:
        listed = ", ".join(f"{room} {side}" for room, side in self.rooms) or "nothing yet"
        return f"You passed: {listed}." + (" You have been here before." if self.revisit else "")

    def slots(self) -> np.ndarray:
        """Condition B's input: S1 as 3 x (10 room + 5 direction) one-hots, then S2. [46]

        Index 0 of each one-hot is "empty slot", so a step with fewer rooms is not all zeros.
        """
        out = np.zeros(MAX_ROOMS * 15 + 1, dtype=np.float32)
        for k in range(MAX_ROOMS):
            room, side = self.rooms[k] if k < len(self.rooms) else (None, None)
            out[k * 15 + (0 if room is None else 1 + ROOM_NAMES.index(room))] = 1.0
            out[k * 15 + 10 + (0 if side is None else 1 + DIRECTIONS.index(side))] = 1.0
        out[-1] = float(self.revisit)
        return out


def describe(pose: np.ndarray, node_rooms: list[str],
             graph: PlaceGraph | None = None) -> list[Prefix]:
    """The prefix at every step of a walk. `pose` is [T, 4] (x, height, z, yaw).

    `node_rooms[i]` is the normalised room name of node i, i.e. of frame `graph.node_step[i]`;
    build the graph first, ask the VLM about those frames, then call this. Causal: step t uses
    only nodes made by t and the room names of their frames.
    """
    pose = np.asarray(pose, dtype=np.float64)
    graph = graph if graph is not None else build(pose[:, :3])
    if len(node_rooms) != len(graph.node_position):
        raise ValueError(f"{len(node_rooms)} room names for {len(graph.node_position)} nodes")
    segments: list[list] = []        # [room, last node] per run of one room along the visits
    previous = None
    out = []
    for t, node in enumerate(graph.current):
        if node != previous:
            if segments and segments[-1][0] == node_rooms[node]:
                segments[-1][1] = node
            else:
                segments.append([node_rooms[node], node])
            previous = node
        here = node_rooms[node]
        rooms: list[tuple[str, str]] = []
        for room, last in reversed(segments[:-1]):
            if room in (here, "other") or any(room == r for r, _ in rooms):
                continue
            angle = relative_bearing(pose[t, :3], pose[t, 3], graph.node_position[last])
            rooms.append((room, DIRECTIONS[int(quadrant(angle))]))
            if len(rooms) == MAX_ROOMS:
                break
        out.append(Prefix(tuple(rooms), bool(graph.revisit[t])))
    return out


def shuffled(prefix: Prefix, episode_id: str, frame_index: int, revisit_rate: float,
             keep_revisit: bool = False) -> Prefix:
    """Condition D (and D' with `keep_revisit`): the same number of S1 slots, wrong contents.

    Room names are drawn without replacement from the nine, directions from the four, and S2 at
    `revisit_rate` (its frequency over the demonstrations), all from a seed fixed by the frame
    -- blake2b("{episode_id}:{frame_index}:shuffle") -- so training and evaluation agree.
    """
    key = f"{episode_id}:{frame_index}:shuffle".encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(),
                                               "big"))
    names = rng.choice(len(ROOM_NAMES), size=len(prefix.rooms), replace=False)
    sides = rng.integers(0, len(DIRECTIONS), size=len(prefix.rooms))
    rooms = tuple((ROOM_NAMES[n], DIRECTIONS[d]) for n, d in zip(names, sides))
    revisit = prefix.revisit if keep_revisit else bool(rng.random() < revisit_rate)
    return Prefix(rooms, revisit)


if __name__ == "__main__":
    # A corridor walked out and back: the way back must be flagged, the way out must not.
    out = np.stack([np.linspace(0, 12, 49), np.zeros(49), np.zeros(49)], axis=1)
    path = np.concatenate([out, out[::-1]])
    graph = build(path)
    print(f"nodes {len(graph.node_position)}, revisit steps {graph.revisit.sum()} of {len(path)}")
    assert not graph.revisit[:49].any(), "walking out is never a revisit"
    assert graph.revisit[-10:].all(), "arriving back at the start is"

    # Directions, in Habitat's convention: yaw 0 faces -z, and +x is then on the right.
    sides = {name: DIRECTIONS[int(quadrant(relative_bearing(np.zeros(3), 0.0, target)))]
             for name, target in {"ahead": (0, 0, -2), "left": (-2, 0, 0),
                                  "behind": (0, 0, 2), "right": (2, 0, 0)}.items()}
    assert sides == {"ahead": "ahead", "left": "to your left", "behind": "behind you",
                     "right": "to your right"}, sides
    # After a quarter turn left the agent faces -x, and +z (what was behind) is on its left.
    assert DIRECTIONS[int(quadrant(relative_bearing(np.zeros(3), np.pi / 2, (0, 0, 2))))] \
        == "to your left"

    # Kitchen, then hallway, then a bedroom, walking along -z; then turn left and stand still.
    walk = np.zeros((40, 4))
    walk[:, 2] = -np.minimum(np.arange(40), 32) * 0.25     # 8 m ahead, then stop
    walk[33:, 3] = np.pi / 2                                # turned left at the end
    graph = build(walk[:, :3])
    rooms = ["kitchen"] * 2 + ["hallway"] * 2 + ["bedroom"] * (len(graph.node_position) - 4)
    steps = describe(walk, rooms, graph)
    print(f"step 0:  {steps[0].text}")
    print(f"step 32: {steps[32].text}")
    print(f"step 39: {steps[39].text}")
    assert steps[0].text == "You passed: nothing yet."
    assert steps[32].rooms == (("hallway", "behind you"), ("kitchen", "behind you"))
    assert steps[39].rooms == (("hallway", "to your left"), ("kitchen", "to your left"))
    assert steps[39].slots().sum() == 6 and steps[39].slots()[-1] == 0

    fake = shuffled(steps[39], "ep", 39, revisit_rate=0.0)
    assert len(fake.rooms) == 2 and fake == shuffled(steps[39], "ep", 39, revisit_rate=0.0)
    assert shuffled(Prefix((), True), "ep", 1, 0.0, keep_revisit=True).revisit
    assert [normalize_room(a) for a in ("Kitchen.", "a laundry room", "Dining", "garage")] \
        == ["kitchen", "laundry room", "dining room", "other"]
    longest = Prefix((("laundry room", "to your right"), ("living room", "to your left"),
                      ("dining room", "behind you")), True).text
    print(f"longest: {longest}  ({len(longest.split())} words)")
    print("ok")
