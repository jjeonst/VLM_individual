"""TopoPrompt step 1-d (a): how often the VLM names the room a place-graph node stands in.

The prefix's slot S1 lists room names the VLM gave (design document, 4.3). A wrong name makes
S1 noise for conditions B and C alike, so the rate is measured before either is trained (T1 d).
The names are already cached: T0's encoding asked "What room is this?" of the frame that made
every node (`probe_prefix.py`, `node_answers`). Only the ground truth is new.

**Ground truth.** HM3D's semantic annotation gives every object a region number but no region a
name. The region a node stands in is read the way Appendix G reads a path point: a semantic
camera at the node looks straight down, and the region of the floor-like object most common in
the middle of the view is the node's. The region is then named from the objects in it -- a
toilet makes a bathroom -- by `ROOM_RULES`. A region can earn several names (an open-plan
kitchen with a dining table and a sofa), and the VLM's answer counts as right when it is any of
them. A region that earns none is left out rather than guessed, and how many are left out is
reported, since the rate is only meaningful over what could be judged.

    python probe_rooms.py label [--limit N]   # ground truth for every T0 trajectory, CPU + GL
    python probe_rooms.py score               # accuracy tables -> rooms_report.json

Habitat quirk found on the way: an agent with only a semantic camera renders nothing (all
zeros); a colour camera has to be attached as well, even though it is never read.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import place_graph  # noqa: E402

HABITAT_ROOT = Path("/data/topovlm/habitat")
SCENE_ROOT = HABITAT_ROOT / "scene_datasets"
SCENE_DATASET_CONFIG = SCENE_ROOT / "hm3d" / "hm3d_annotated_basis.scene_dataset_config.json"
MANIFEST = HABITAT_ROOT / "episodes" / "pr2l_habitat_web_hd" / "train" / "manifest.jsonl"
FRAMES = Path("/data/topovlm/probes/prefix/frames")
OUT = Path("/data/topovlm/probes/rooms")

# A region earns a room name when it holds any object whose category is in the set. Exact
# category names from the HM3D annotation (checked against the 1,542 names it uses).
ROOM_RULES = {
    "bathroom": {"toilet", "shower", "bathtub", "shower wall", "shower door", "shower curtain",
                 "bathroom cabinet", "shower hose/head", "toilet paper", "bath", "bath tub", "bathroom counter",
                 "washbasin", "bath mat", "bathroom mat", "bath sink"},
    "bedroom": {"bed", "bunk bed", "crib", "nightstand"},
    "kitchen": {"stove", "oven", "refrigerator", "fridge", "kitchen cabinet", "microwave",
                "dishwasher", "kitchen countertop item", "kitchen island", "kitchen counter",
                "range hood", "oven and stove", "stovetop", "kitchen sink cabinet", "mini fridge",
                "refrigerator cabinet"},
    "dining room": {"dining table", "dining chair"},
    "living room": {"couch", "sofa", "armchair", "tv", "coffee table", "fireplace",
                    "sofa chair", "l-shaped sofa", "sofa set", "circular sofa", "sofa seat"},
    "office": {"desk", "office chair", "computer", "monitor", "printer"},
    "laundry room": {"washing machine", "clothes dryer", "washer-dryer", "washing machine and dryer"},
    "stairs": {"stairs", "staircase", "stair", "step"},
}
# A region with none of the furniture above but several doors is a passage between rooms.
HALLWAY_MIN_DOORS = 2
DOORS = {"door", "door frame", "sliding door"}
# The six rooms the design's pass mark (T1 d: >= 50 %) is judged on.
MAIN_ROOMS = ("bathroom", "bedroom", "kitchen", "living room", "dining room", "hallway")
FLOORISH = {"floor", "rug", "carpet", "mat", "doormat", "stairs", "step", "floor mat"}
CENTRE = slice(24, 40)          # middle 16 x 16 of the 64 x 64 downward view


def room_names(categories: Counter) -> set[str]:
    """The room names a region's objects support; empty when nothing decides it."""
    names = {room for room, cue in ROOM_RULES.items() if cue & set(categories)}
    if not names and sum(categories[d] for d in DOORS) >= HALLWAY_MIN_DOORS:
        names.add("hallway")
    return names


def make_sim(glb: Path):
    import habitat_sim
    import magnum as mn

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(glb)
    backend.scene_dataset_config_file = str(SCENE_DATASET_CONFIG)
    backend.enable_physics = False
    down = mn.Vector3(-np.pi / 2, 0.0, 0.0)
    sensors = []
    for uuid, kind in (("sem", habitat_sim.SensorType.SEMANTIC),
                       ("rgb", habitat_sim.SensorType.COLOR)):   # see the module docstring
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid, spec.sensor_type, spec.resolution = uuid, kind, [64, 64]
        spec.position, spec.orientation = mn.Vector3(0.0, 1.0, 0.0), down
        sensors.append(spec)
    agent = habitat_sim.AgentConfiguration()
    agent.sensor_specifications = sensors
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))


def region_under(sim, position: np.ndarray, objects) -> str | None:
    """Region id of the floor-like object most seen straight below `position`."""
    import habitat_sim

    state = habitat_sim.AgentState()
    state.position = np.asarray(position, dtype=np.float32)
    sim.get_agent(0).set_state(state)
    ids = np.asarray(sim.get_sensor_observations()["sem"])[CENTRE, CENTRE].ravel()
    counts = Counter(int(i) for i in ids if 0 < i < len(objects))
    floorish = [(n, i) for i, n in counts.items() if objects[i][1] in FLOORISH]
    pool = floorish or [(n, i) for i, n in counts.items() if objects[i][1] != "ceiling"]
    if not pool:
        return None
    return objects[max(pool)[1]][0]


def label(args) -> int:
    records = {json.loads(line)["episode_id"]: json.loads(line) for line in MANIFEST.open()}
    files = sorted(FRAMES.glob("*.npz"))
    if args.limit:
        files = files[:: max(1, len(files) // args.limit)][: args.limit]
    by_scene = defaultdict(list)
    for path in files:
        by_scene[records[path.stem]["scene_id"]].append(path)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, (scene, paths) in enumerate(sorted(by_scene.items()), start=1):
        sim = make_sim(SCENE_ROOT / scene)
        try:
            objects = [None] * (max(int(o.id.split("_")[-1]) for o in sim.semantic_scene.objects
                                    if o is not None) + 1)
            regions = defaultdict(Counter)
            for o in sim.semantic_scene.objects:
                if o is None:
                    continue
                region = o.region.id if o.region is not None else None
                name = o.category.name() if o.category is not None else "unknown"
                objects[int(o.id.split("_")[-1])] = (region, name)
                regions[region][name] += 1
            objects = [o if o is not None else (None, "unknown") for o in objects]
            for path in paths:
                item = np.load(path, allow_pickle=True)
                record = records[path.stem]
                pose = np.load(HABITAT_ROOT / record["pose_path"])
                graph = place_graph.build(pose[:, :3])
                if len(graph.node_position) != len(item["node_rooms"]):
                    raise RuntimeError(f"{path.stem}: graph does not match the cached answers")
                for node, position in enumerate(graph.node_position):
                    region = region_under(sim, position, objects)
                    truth = sorted(room_names(regions[region])) if region not in (None, "_-1") else []
                    rows.append({"episode": path.stem, "scene": record["scene_key"],
                                 "node": node, "step": int(graph.node_step[node]),
                                 "region": region, "truth": truth,
                                 "answer": str(item["node_answers"][node]),
                                 "vlm": str(item["node_rooms"][node])})
        finally:
            sim.close()
        print(f"[label] 장면 {number}/{len(by_scene)} {scene.split('/')[-2]}: 누적 노드 {len(rows)}",
              flush=True)
    target = OUT / ("labels_smoke.jsonl" if args.limit else "labels.jsonl")
    target.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"[label] {target} 저장 ({len(rows)} 노드)")
    return 0


def score(args) -> int:
    rows = [json.loads(line) for line in (OUT / args.labels).open()]
    judged = [r for r in rows if r["truth"]]
    right = [r["vlm"] in r["truth"] for r in judged]
    print(f"[score] 노드 {len(rows)}, 판정 가능 {len(judged)} ({100 * len(judged) / len(rows):.0f}%), "
          f"전체 정확도 {100 * np.mean(right):.1f}%")
    report = {"nodes": len(rows), "judged": len(judged), "accuracy": float(np.mean(right))}

    # Recall on regions that earn exactly one name: of nodes standing in a plain bathroom, how
    # many did the VLM call a bathroom. Multi-name regions would let any of several answers
    # through and so flatter the rate; they are scored in the overall figure only.
    single = defaultdict(list)
    for r in judged:
        if len(r["truth"]) == 1:
            single[r["truth"][0]].append(r["vlm"])
    report["single_region_recall"] = {}
    print("[score] 단일 이름 구역에서 VLM이 맞힌 비율 (주요 6개 방 기준 50 %)")
    for room in list(MAIN_ROOMS) + [r for r in ROOM_RULES if r not in MAIN_ROOMS]:
        answers = single.get(room, [])
        if not answers:
            continue
        rate = float(np.mean([a == room for a in answers]))
        common = Counter(answers).most_common(3)
        flag = (" 통과" if rate >= 0.5 else " 미달") if room in MAIN_ROOMS else ""
        print(f"  {room:12s} n={len(answers):5d}  {100 * rate:5.1f}%{flag}   VLM 답 상위: {common}")
        report["single_region_recall"][room] = {"n": len(answers), "rate": rate,
                                                "answers": dict(Counter(answers))}

    # Precision of each name the VLM gives: when it says "kitchen", is the node in a region a
    # kitchen could be. This is what matters for S1 -- a name in the prefix being trustworthy.
    said = defaultdict(list)
    for r in judged:
        said[r["vlm"]].append(r["vlm"] in r["truth"])
    report["answer_precision"] = {k: {"n": len(v), "rate": float(np.mean(v))} for k, v in said.items()}
    print("[score] VLM이 그 이름을 댔을 때 맞은 비율")
    for k, v in sorted(said.items(), key=lambda kv: -len(kv[1])):
        print(f"  {k:12s} n={len(v):5d}  {100 * np.mean(v):5.1f}%")
    (OUT / "rooms_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=["label", "score"])
    parser.add_argument("--limit", type=int, default=None, help="trajectories, for a smoke run")
    parser.add_argument("--labels", default="labels.jsonl")
    args = parser.parse_args()
    return label(args) if args.mode == "label" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())
