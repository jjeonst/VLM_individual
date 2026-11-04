"""TopoPrompt step 1-d (b): does the VLM see that a doorway is blocked?

Heejun's direction 4 (design document, 9/29 tab) changes the environment at test time -- a
doorway that was open in training is shut -- and asks whether memory helps the agent recover.
Before any of that, the VLM has to be able to see the change. This measures it, without
training, in the two ways T0 measured the prefix: what the VLM *says*, and whether the hidden
state the policy reads *carries* it.

**Pairs.** A doorway is where a demonstration walks from one region of HM3D's semantic
annotation into another (the region under the agent is read as in `probe_rooms.py`), at a
narrow passage. From a navigable point `VIEW_BACK_M` before it, facing it, the scene is rendered
three times: as it is (open), and with a panel standing in the doorway in each of `LOOKS` (blocked).
Viewpoint, lighting and everything else are the same, so the images differ only in the doorway
being shut. The panel is the same kind of object a test-time blocking experiment would insert.

**Two looks.** `dark` is the bare cube primitive, a glossy dark slab that can pass for the mouth of
an unlit room. `door` is the same slab wearing a white two-panel door (`write_door_asset`). A probe
trained on one look alone could learn the look -- "a dark rectangle in the middle" -- rather than
the blockage, so each is also tested on the other, and the transfer is what carries the verdict.

**Measures.**
    answer   "Is the doorway in front of you open or blocked?" -- greedy, one word, scored
             against the truth (blocked accepts blocked / closed / shut), per look.
    latent   the policy's own question ("Would a {goal} be found here?", no-CoT, minimum
             length 0 -- condition A exactly), and a linear probe for open vs blocked on its
             language positions (`last` and `mean`, raw and after nocot_min0's PCA), split by
             scene, bootstrap over pairs, for every train look x test look. Chance is 0.5.

**Pass criterion.** After nocot_min0's PCA, what the policy reads, a probe trained on one look and
tested on the other beats chance by at least 2 sigma of balanced accuracy: both directions, for
`last` and `mean`. The answers are reported but do not decide, since the policy reads the hidden
state and not the answer.

    python probe_doors.py render [--limit N]    # pairs -> /data/topovlm/probes/doors/pairs
    python probe_doors.py encode [--split-gpus] # VLM answers and latents
    python probe_doors.py probe                 # -> doors_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_rooms  # noqa: E402

HABITAT_ROOT = Path("/data/topovlm/habitat")
MANIFEST = probe_rooms.MANIFEST
OUT = Path(__import__("os").environ.get("DOORS_OUT", "/data/topovlm/probes/doors"))
CONDITION = "nocot_min0"

VIEW_BACK_M = 2.0            # viewpoint distance before the doorway
MIN_SIDE_STEPS = 3           # both regions held for this many steps around the crossing
MAX_FREE_WIDTH_M = 1.3       # navigable width across the crossing; wider is open plan, not a door
PANEL_HEIGHT_M = 2.2
PANEL_THICK_M = 0.16
PANEL_MARGIN_M = 0.5         # added to the free width: the agent radius on each side, and some
MIN_DOOR_GAP_M = 1.0         # two crossings closer than this are the same doorway
FLANK_PX = 40                # image columns read on each side of the panel (`site_measures`)
MIN_OPENING = 0.6            # thresholds on `site_measures`, set by eye on the first 21 doorways
MIN_FLANK = 0.5
MIN_WIDENING_M = 0.3
PER_SCENE = 6
TRAJECTORIES_PER_SCENE = 6
QUESTION = "Is the doorway in front of you open or blocked? Answer in one word."
ANSWER_TOKENS = 4
BLOCKED_WORDS = ("blocked", "closed", "shut", "obstructed")
PASS_SIGMA = 2.0
LOOKS = ("dark", "door")     # image index = 1 + position here; 0 is the open view
DOOR_ASSET = "door_panel.glb"


def make_sim(glb: Path):
    import habitat_sim
    import magnum as mn
    import render

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(glb)
    backend.scene_dataset_config_file = str(probe_rooms.SCENE_DATASET_CONFIG)
    backend.enable_physics = False

    def camera(uuid, kind, resolution, height, orientation=None):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid, spec.sensor_type, spec.resolution = uuid, kind, resolution
        spec.hfov = render.HFOV_DEG
        spec.position = mn.Vector3(0.0, height, 0.0)
        if orientation is not None:
            spec.orientation = orientation
        return spec

    full = [render.RGB_HEIGHT, render.RGB_WIDTH]
    agent = habitat_sim.AgentConfiguration()
    agent.height, agent.radius = render.AGENT_HEIGHT, render.AGENT_RADIUS
    agent.sensor_specifications = [
        camera("rgb", habitat_sim.SensorType.COLOR, full, render.CAMERA_HEIGHT),
        camera("depth", habitat_sim.SensorType.DEPTH, full, render.CAMERA_HEIGHT),
        camera("sem", habitat_sim.SensorType.SEMANTIC, [64, 64], 1.0,
               mn.Vector3(-np.pi / 2, 0.0, 0.0))]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))

    return sim


def write_door_asset(path: Path) -> None:
    """A box spanning 2 units like the cube primitive, its two broad faces showing a white
    two-panel door. Written by hand as binary glTF, since the environment has no mesh library."""
    from io import BytesIO

    from PIL import Image

    height, width = 1024, 512
    shade = np.linspace(1.0, 0.9, height)[:, None, None]
    door = np.broadcast_to(np.array([236.0, 232.0, 224.0]) * shade, (height, width, 3)).copy()
    for top, bottom in ((0.06, 0.46), (0.53, 0.95)):
        a, b, left, right = int(top * height), int(bottom * height), int(0.14 * width), int(0.86 * width)
        door[a:b, left:right] *= 0.93
        door[a:a + 6, left:right] *= 0.8
        door[b - 6:b, left:right] *= 1.05
        door[a:b, left:left + 6] *= 0.8
        door[a:b, right - 6:right] *= 1.05
    y, x = int(0.5 * height), int(0.87 * width)
    door[y - 8:y + 8, x - 8:x + 44] = 45.0
    image = BytesIO()
    Image.fromarray(door.clip(0, 255).astype(np.uint8)).save(image, format="PNG")

    # Four vertices a face. The broad faces (z = +-1) take the whole picture, the edges one texel.
    faces = [((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1), (0, 0, 1), True),
             ((1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1), (0, 0, -1), True),
             ((1, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1), (1, 0, 0), False),
             ((-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1), (-1, 0, 0), False),
             ((-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1), (0, 1, 0), False),
             ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1), (0, -1, 0), False)]
    position, normal, uv, index = [], [], [], []
    for number, (*corners, facing, pictured) in enumerate(faces):
        position += corners
        normal += [facing] * 4
        uv += [(0, 1), (1, 1), (1, 0), (0, 0)] if pictured else [(0.02, 0.5)] * 4
        base = 4 * number
        index += [base, base + 1, base + 2, base, base + 2, base + 3]
    position, normal = np.asarray(position, "<f4"), np.asarray(normal, "<f4")
    uv, index = np.asarray(uv, "<f4"), np.asarray(index, "<u2")

    blob, views = b"", []
    for data, target in ((position.tobytes(), 34962), (normal.tobytes(), 34962), (uv.tobytes(), 34962),
                         (index.tobytes(), 34963), (image.getvalue(), None)):
        view = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)}
        if target:
            view["target"] = target
        views.append(view)
        blob += data + b"\0" * (-len(data) % 4)
    gltf = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                                        "indices": 3, "material": 0}]}],
            "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0},
                                                    "metallicFactor": 0.0, "roughnessFactor": 1.0}}],
            "textures": [{"source": 0}],
            "images": [{"bufferView": 4, "mimeType": "image/png"}],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(position), "type": "VEC3",
                           "min": [-1, -1, -1], "max": [1, 1, 1]},
                          {"bufferView": 1, "componentType": 5126, "count": len(normal), "type": "VEC3"},
                          {"bufferView": 2, "componentType": 5126, "count": len(uv), "type": "VEC2"},
                          {"bufferView": 3, "componentType": 5123, "count": len(index), "type": "SCALAR"}],
            "bufferViews": views, "buffers": [{"byteLength": len(blob)}]}
    text = json.dumps(gltf, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, 28 + len(text) + len(blob))
                     + struct.pack("<II", len(text), 0x4E4F534A) + text
                     + struct.pack("<II", len(blob), 0x004E4942) + blob)


def add_panel(sim, width: float, look: str):
    """A flat panel `width` wide, as a kinematic object. Both looks span 2 units; the size has to
    be set on the template, since an added object refuses a new scale. The door is flat shaded and
    unlit so it shows its picture as painted, as the scene's own baked textures do; under the
    default lights, whatever the shader, it comes out as dark as the primitive (about 80 of 255)."""
    import habitat_sim
    import magnum as mn

    templates = sim.get_object_template_manager()
    if look == "door":
        panel = templates.create_new_template("door_panel_door", False)
        panel.render_asset_handle = str(OUT / DOOR_ASSET)
        panel.force_flat_shading = True
        panel.is_collidable = False
    else:
        panel = templates.get_template_by_handle(
            next(h for h in templates.get_template_handles() if "cubeSolid" in h))
    panel.scale = mn.Vector3(width / 2, PANEL_HEIGHT_M / 2, PANEL_THICK_M / 2)
    handle = f"door_panel_{look}"
    templates.register_template(panel, handle)
    light = habitat_sim.gfx.NO_LIGHT_KEY if look == "door" else habitat_sim.gfx.DEFAULT_LIGHTING_KEY
    added = sim.get_rigid_object_manager().add_object_by_template_handle(handle, light_setup_key=light)
    added.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    return added


def place(sim, position, yaw: float):
    import habitat_sim

    state = habitat_sim.AgentState()
    state.position = np.asarray(position, dtype=np.float32)
    state.rotation = habitat_sim.utils.common.quat_from_angle_axis(float(yaw),
                                                                  np.array([0.0, 1.0, 0.0]))
    sim.get_agent(0).set_state(state)
    return sim.get_sensor_observations()


def blocked_views(sim, door: dict, yaw: float, width: float) -> dict:
    """The doorway from the same viewpoint with a panel in it, once per look."""
    import magnum as mn

    views = {}
    for look in LOOKS:
        panel = add_panel(sim, width, look)
        try:
            panel.rotate_y(mn.Rad(float(yaw)))
            c = door["centre"]
            panel.translation = mn.Vector3(c[0], c[1] + PANEL_HEIGHT_M / 2, c[2])
            views[look] = place(sim, door["view"], yaw)
        finally:
            sim.get_rigid_object_manager().remove_object_by_id(panel.object_id)
    return views


def panel_columns(width: float, reach: float) -> tuple[int, int]:
    """The image columns a panel `width` wide spans, seen square-on from `reach`."""
    import render

    focal = render.RGB_WIDTH / 2 / np.tan(np.deg2rad(render.HFOV_DEG) / 2)
    half = focal * width / 2 / max(reach, 1e-3)
    return (max(0, int(render.RGB_WIDTH / 2 - half)),
            min(render.RGB_WIDTH, int(render.RGB_WIDTH / 2 + half)))


def site_measures(sim, view: dict, door: dict, reach: float, width: float) -> dict:
    """How much the place looks like a doorway with the panel fitted in it, from the open view.

    opening  of the panel's width at eye level, the share that sees past it; low when the panel
             stands mostly against walls or furniture
    flank    the share of `FLANK_PX` columns just outside the panel that stops at or before it, on
             whichever side has less; low when the panel stands free in an open room
    widening how much wider the passage gets 0.75 m to either side, in metres; zero in a corridor
    """
    left, right = panel_columns(width, reach)
    depth = np.asarray(view["depth"])[200:280]
    opening = float((depth[:, left:right] > reach + 0.3).mean())
    solid = depth <= reach + 0.3
    sides = [solid[:, max(0, left - FLANK_PX):left], solid[:, right:right + FLANK_PX]]
    flank = min(float(side.mean()) if side.size else 1.0 for side in sides)
    heading = door["heading"]
    across = np.array([-heading[2], 0.0, heading[0]])
    beside = [free_width(sim, np.asarray(sim.pathfinder.snap_point(door["centre"] + sign * 0.75 * heading)), across)
              for sign in (1.0, -1.0)]
    return {"opening": opening, "flank": flank, "widening": max(beside) - door["width"]}


def free_width(sim, centre: np.ndarray, across: np.ndarray) -> float:
    """Navigable width through `centre` along `across`, in 5 cm steps, up to 3 m."""
    width = 0.0
    for sign in (1.0, -1.0):
        for k in range(1, 31):
            if not sim.pathfinder.is_navigable(centre + sign * 0.05 * k * across, 0.5):
                width += 0.05 * (k - 1)
                break
        else:
            width += 1.5
    return width


def doorways(sim, pose: np.ndarray, objects) -> list[dict]:
    """Crossings between two regions at a narrow passage, with a clear view from behind."""
    regions = [probe_rooms.region_under(sim, p, objects) for p in pose[:, :3]]
    found = []
    for t in range(MIN_SIDE_STEPS, len(pose) - MIN_SIDE_STEPS):
        before, after = regions[t - MIN_SIDE_STEPS:t], regions[t:t + MIN_SIDE_STEPS]
        if (len(set(before)) != 1 or len(set(after)) != 1 or before[0] == after[0]
                or None in (before[0], after[0]) or "_-1" in (before[0], after[0])):
            continue
        heading = pose[t, :3] - pose[t - 1, :3]
        heading[1] = 0.0
        if np.linalg.norm(heading) < 1e-3:
            continue
        heading /= np.linalg.norm(heading)
        centre = np.asarray(sim.pathfinder.snap_point((pose[t - 1, :3] + pose[t, :3]) / 2))
        across = np.array([-heading[2], 0.0, heading[0]])
        width = free_width(sim, centre, across)
        if not 0.05 < width <= MAX_FREE_WIDTH_M:
            continue
        view = np.asarray(sim.pathfinder.snap_point(centre - VIEW_BACK_M * heading))
        if not np.isfinite(view).all() or abs(view[1] - centre[1]) > 0.3:
            continue
        found.append({"step": t, "centre": centre, "view": view, "width": width, "heading": heading,
                      "from": before[0], "to": after[0]})
    return found


def render_pairs(args) -> int:
    records = [json.loads(line) for line in MANIFEST.open()]
    by_scene = defaultdict(list)
    for record in records:
        by_scene[record["scene_id"]].append(record)
    scenes = sorted(by_scene)
    if args.limit:
        scenes = scenes[: args.limit]
    out = OUT / "pairs"
    out.mkdir(parents=True, exist_ok=True)
    write_door_asset(OUT / DOOR_ASSET)
    total, started = 0, time.time()
    for number, scene in enumerate(scenes, start=1):
        key = by_scene[scene][0]["scene_key"]
        if (out / f"{key}.done").exists():
            continue
        sim = make_sim(probe_rooms.SCENE_ROOT / scene)
        kept = []
        try:
            objects = [(None, "unknown")] * (max(o.semantic_id for o in sim.semantic_scene.objects
                                                 if o is not None) + 1)
            for o in sim.semantic_scene.objects:
                if o is not None:
                    objects[o.semantic_id] = (o.region.id if o.region is not None else None,
                                              o.category.name() if o.category else "unknown")
            rng = np.random.default_rng(int(hashlib.blake2b(key.encode(), digest_size=4).hexdigest(), 16))
            pool = by_scene[scene]
            for index in rng.permutation(len(pool))[:TRAJECTORIES_PER_SCENE]:
                record = pool[index]
                pose = np.load(HABITAT_ROOT / record["pose_path"])
                for door in doorways(sim, pose, objects):
                    if len(kept) >= PER_SCENE:
                        break
                    if any(np.linalg.norm(door["centre"] - k["centre"]) < MIN_DOOR_GAP_M for k in kept):
                        continue
                    aim = door["centre"] - door["view"]
                    yaw = np.arctan2(-aim[0], -aim[2])         # forward is (-sin yaw, -cos yaw)
                    reach = float(np.linalg.norm(aim[[0, 2]]))
                    opened = place(sim, door["view"], yaw)
                    middle = np.asarray(opened["depth"])[200:280, 280:360]
                    if np.median(middle) < reach - 0.3:        # something stands in front of it
                        continue
                    width = min(door["width"] + PANEL_MARGIN_M, 2.0)
                    measures = site_measures(sim, opened, door, reach, width)
                    if (measures["opening"] < MIN_OPENING or measures["flank"] < MIN_FLANK
                            or measures["widening"] < MIN_WIDENING_M):
                        continue
                    shut = blocked_views(sim, door, yaw, width)
                    after = {look: float(np.median(np.asarray(v["depth"])[200:280, 280:360]))
                             for look, v in shut.items()}
                    if max(after.values()) > reach + 0.3:      # the panel is not where we look
                        continue
                    kept.append(door)
                    name = f"{key}_{len(kept) - 1}"
                    np.savez_compressed(
                        out / f"{name}.npz",
                        open=np.asarray(opened["rgb"])[..., :3],
                        **{look: np.asarray(v["rgb"])[..., :3] for look, v in shut.items()},
                        scene=key, episode=record["episode_id"], goal=record["object_category"],
                        step=door["step"], centre=door["centre"], view=door["view"], yaw=yaw,
                        width=door["width"], region_from=door["from"], region_to=door["to"],
                        depth_open=float(np.median(middle)), depth_blocked=after["dark"], panel_width=width,
                        **measures)
                if len(kept) >= PER_SCENE:
                    break
        finally:
            sim.close()
        (out / f"{key}.done").write_text(str(len(kept)))
        total += len(kept)
        print(f"[render] 장면 {number}/{len(scenes)} {key}: 출입구 {len(kept)}개, 누적 {total}, "
              f"{time.time() - started:.0f}s", flush=True)
    return 0


def encode(args) -> int:
    from probe_answer_length import load_split_vlm
    from probe_prefix import summarise
    from vlm_features import encode_batch, load_vlm, min_new_tokens_for

    pairs = sorted((OUT / "pairs").glob("*.npz"))[args.shard::args.shards]
    vlm = load_split_vlm(args.layers_on_first) if args.split_gpus else load_vlm()
    minimum = min_new_tokens_for(CONDITION)
    out = OUT / "encoded"
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for number, path in enumerate(pairs, start=1):
        target = out / path.name
        if target.exists():
            continue
        item = np.load(path)
        frames = np.stack([item["open"]] + [item[look] for look in LOOKS])
        goal, name, count = str(item["goal"]), path.stem, len(frames)
        # All images of a doorway get the same seeds, so the decoding differs only by the image.
        answers = [x.generated_text for x in encode_batch(
            vlm, frames, goal, [name] * count, [0] * count, with_cot=False, template=QUESTION,
            min_new_tokens=0, greedy=True, max_new_tokens=ANSWER_TOKENS)]
        encoded = encode_batch(vlm, frames, goal, [name] * count, [0] * count, with_cot=False,
                               min_new_tokens=minimum)
        summaries = [summarise(x) for x in encoded]
        np.savez(target, scene=item["scene"], goal=goal, answers=np.asarray(answers),
                 replies=np.asarray([x.generated_text for x in encoded]),
                 last=np.stack([s[0] for s in summaries]).astype(np.float16),
                 mean=np.stack([s[1] for s in summaries]).astype(np.float16))
        if number % 20 == 0 or number == len(pairs):
            print(f"[encode] {number}/{len(pairs)}  {time.time() - started:.0f}s  예: 열림 "
                  f"{answers[0]!r} / 어두운 판자 {answers[1]!r} / 문 {answers[2]!r}", flush=True)
    return 0


def said_blocked(answer: str) -> bool:
    words = answer.lower()
    return any(w in words for w in BLOCKED_WORDS)


def probe(args) -> int:
    import pca
    import torch
    from probe_lstm import BOOTSTRAP, LAMBDAS, balanced_accuracy, fit, split_scenes

    items = [np.load(p) for p in sorted((OUT / "encoded").glob("*.npz"))]
    scenes = [str(i["scene"]) for i in items]
    role = split_scenes([{"scene_key": s} for s in sorted(set(scenes))])
    report = {"pairs": len(items)}

    # What the VLM says. Each doorway has one open image and one blocked image per look.
    said = np.array([[said_blocked(str(a)) for a in i["answers"]] for i in items])
    report["answer"] = {"open_correct": float(np.mean(~said[:, 0])),
                        "answers_open": dict(Counter(str(i["answers"][0]).strip().lower() for i in items).most_common(8))}
    print(f"[probe] 출입구 {len(items)}쌍 | 답변: 열림에 열림 {100 * report['answer']['open_correct']:.1f}%")
    print(f"        열림 답 상위 {report['answer']['answers_open']}")
    for number, look in enumerate(LOOKS, start=1):
        report["answer"][look] = {"blocked_correct": float(np.mean(said[:, number])),
                                  "balanced": float((np.mean(~said[:, 0]) + np.mean(said[:, number])) / 2),
                                  "answers": dict(Counter(str(i["answers"][number]).strip().lower() for i in items).most_common(8))}
        print(f"        {look}: 막힘에 막힘 {100 * report['answer'][look]['blocked_correct']:.1f}%, "
              f"답 상위 {report['answer'][look]['answers']}")

    # Whether the policy's hidden state carries it, on each look and across them.
    basis = pca.load(HABITAT_ROOT / "embeddings" / f"pr2l_habitat_web_hd_{CONDITION}" / "pca.npz")
    split = {r: [k for k, s in enumerate(scenes) if role[s] == r] for r in ("train", "val", "test")}
    print("[probe] 장면 분할 쌍 수: " + ", ".join(f"{r} {len(v)}" for r, v in split.items()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report["latent"] = {}
    for summary in ("last", "mean"):
        for reduced in (False, True):
            def features(index, look):
                raw = np.concatenate([items[k][summary][[0, 1 + LOOKS.index(look)]].astype(np.float32)
                                      for k in index])
                if reduced:
                    return np.concatenate([basis.apply(raw[:, layer])
                                           for layer in range(raw.shape[1])], axis=1)
                return raw.reshape(len(raw), -1)
            labels = {r: np.tile([0, 1], len(v)) for r, v in split.items()}
            for trained in LOOKS:
                train_x = features(split["train"], trained)
                mean, std = train_x.mean(0), train_x.std(0)
                std = np.where(std < 1e-3, 1.0, std)
                to = lambda a: torch.from_numpy((a - mean) / std).float().to(device)
                tx, ty = to(train_x), torch.from_numpy(labels["train"]).to(device)
                best = None
                val_x = to(features(split["val"], trained))
                for lam in LAMBDAS:
                    model = fit(tx, ty, 2, lam)
                    with torch.no_grad():
                        pred = model(val_x).argmax(1).cpu().numpy()
                    score = balanced_accuracy(pred, labels["val"], 2)
                    if best is None or (np.isfinite(score) and not score <= best[0]):
                        best = (score, lam, model)
                _, lam, model = best
                for tested in LOOKS:
                    with torch.no_grad():
                        pred = model(to(features(split["test"], tested))).argmax(1).cpu().numpy()
                    true, pairs = labels["test"], len(split["test"])
                    # Resampled by doorway, keeping each open/blocked couple together.
                    draws = np.random.default_rng(1).integers(0, pairs, (BOOTSTRAP, pairs))
                    boots = np.asarray([balanced_accuracy(pred[np.concatenate([[2 * i, 2 * i + 1] for i in d])],
                                                          true[np.concatenate([[2 * i, 2 * i + 1] for i in d])], 2)
                                        for d in draws])
                    score, sigma = balanced_accuracy(pred, true, 2), float(np.nanstd(boots))
                    key = f"{summary}/{'pca' if reduced else 'raw'}/{trained}->{tested}"
                    report["latent"][key] = {"bacc": score, "sigma": sigma, "lambda": lam, "test_pairs": pairs}
                    print(f"[probe] 잠재 {key:22s} bAcc {score:.3f} ± {sigma:.3f}  "
                          + (f"(우연 0.5 대비 {(score - 0.5) / sigma:+.1f}σ, λ {lam})" if sigma > 0
                             else f"(재표본 분산 0, λ {lam})"))
    transfer = [v for k, v in report["latent"].items()
                if "/pca/" in k and k.split("/")[-1] in ("dark->door", "door->dark")]
    report["passed"] = all(v["sigma"] > 0 and (v["bacc"] - 0.5) / v["sigma"] >= PASS_SIGMA for v in transfer)
    print(f"[probe] 판정 (PCA 교차 {len(transfer)}칸 모두 우연 + {PASS_SIGMA:g}σ 이상): "
          f"{'통과' if report['passed'] else '미달'}")
    (OUT / "doors_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[probe] {OUT / 'doors_report.json'} 저장")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=["render", "encode", "probe"])
    parser.add_argument("--limit", type=int, default=None, help="scenes, for a smoke run")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--split-gpus", action="store_true")
    parser.add_argument("--layers-on-first", type=int, default=14)
    args = parser.parse_args()
    return {"render": render_pairs, "encode": encode, "probe": probe}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
