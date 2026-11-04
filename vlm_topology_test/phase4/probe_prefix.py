"""Does the topology sentence reach the positions the policy reads? (TopoPrompt T0, step 1-c)

Condition C reads the place-graph sentence before the question but keeps only the question and
answer positions (design document, 4.5), so whatever the sentence says can reach the policy only
if the language model carries it into those positions. This measures that before anything is
trained: the same demonstration frames are encoded three ways and a linear probe asks how much
of the *true* topology each encoding holds.

    A  no sentence                                    the baseline
    C  the true sentence (`place_graph.describe`)     what condition C trains on
    D  same slots, shuffled contents (`shuffled`)     controls for a longer, different prompt

Pass criterion (design document, 7, T0): C beats D by at least 2 sigma of balanced accuracy.

**encode** (GPU, sharded). Picks the frames, asks the VLM the room of every node (the room-name
cache conditions B-D need anyway), writes the sentences, and encodes each frame under A, C and
D with the no-CoT min0 recipe -- same question, same per-frame decoding seed as the stored
training embeddings -- keeping two summaries of the language positions per frame:

* `last` -- the last prompt token, which has read the whole prompt and is where the answer starts.
* `mean` -- the mean over question and answer positions (BOS excluded).

Both are kept before PCA, [2 layers, 4096]; PCA is affine, so it is applied to the summaries at
probe time with the condition's stored basis and commutes with the mean.

Frames are drawn to make revisits common enough to probe: every training trajectory with a
revisit (about 5 %) gives up to `REVISIT_FRAMES` revisit steps and as many others, and the same
number of trajectories without one give `PLAIN_FRAMES` random steps each.

**probe** (GPU or CPU). Labels are the true slots of each frame:

    revisit     2-way   slot S2
    first_room  10-way  slot S1's most recent room, or none
    first_dir   4-way   its direction, on frames that list one
    rooms       4-way   how many rooms S1 lists -- D keeps this by design, so C and D should tie
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pca  # noqa: E402
import place_graph  # noqa: E402
from probe_lstm import BOOTSTRAP, LAMBDAS, balanced_accuracy, fit, split_scenes  # noqa: E402

HABITAT_ROOT = Path("/data/topovlm/habitat")
MANIFEST = HABITAT_ROOT / "episodes" / "pr2l_habitat_web_hd" / "train" / "manifest.jsonl"
CONDITION = "nocot_min0"
REVISIT_FRAMES = 4
PLAIN_FRAMES = 8
ROOM_BATCH = 8
ROOM_TOKENS = 4
CONDITIONS = ("A", "C", "D")
SUMMARIES = ("last", "mean")


def select(records: list[dict], seed: int = 0) -> tuple[list[tuple[dict, list[int]]], float]:
    """Trajectories and the steps to encode in each, fixed by `seed` alone, and slot S2's
    frequency over every demonstration step -- the rate condition D draws S2 at."""
    rng = np.random.default_rng(seed)
    with_revisit, without = [], []
    hits = total = 0
    for record in records:
        pose = np.load(HABITAT_ROOT / record["pose_path"])
        revisit = place_graph.build(pose[:, :3]).revisit
        hits, total = hits + int(revisit.sum()), total + len(revisit)
        (with_revisit if revisit.any() else without).append((record, np.flatnonzero(revisit)))
    chosen = []
    for record, revisits in with_revisit:
        rest = np.setdiff1d(np.arange(1, record["steps"]), revisits)
        take = rng.choice(revisits, min(REVISIT_FRAMES, len(revisits)), replace=False)
        other = rng.choice(rest, min(len(take), len(rest)), replace=False)
        chosen.append((record, sorted(int(t) for t in np.concatenate([take, other]))))
    for index in rng.choice(len(without), min(len(with_revisit), len(without)), replace=False):
        record, _ = without[index]
        steps = rng.choice(record["steps"], min(PLAIN_FRAMES, record["steps"]), replace=False)
        chosen.append((record, sorted(int(t) for t in steps)))
    return chosen, hits / max(total, 1)


def summarise(item) -> tuple[np.ndarray, np.ndarray]:
    """`last` and `mean` of one frame's language positions, each [layers, 4096]."""
    from vlm_features import POOLED_VISUAL_TOKENS
    language = item.tokens[POOLED_VISUAL_TOKENS:]           # [BOS + prompt][answer]
    last = language[item.prompt_count - 1]
    return last, language[1:].mean(axis=0)


def encode(args) -> int:
    from probe_answer_length import load_split_vlm
    from vlm_features import encode_batch, load_vlm, min_new_tokens_for

    records = [json.loads(line) for line in MANIFEST.open()]
    role = split_scenes(records)
    chosen, revisit_rate = select(records)
    if args.limit:
        chosen = chosen[:: max(1, len(chosen) // args.limit)][: args.limit]
    mine = chosen[args.shard::args.shards]
    print(f"[encode] 궤적 {len(chosen)}개 중 샤드 {args.shard}/{args.shards}: {len(mine)}개, "
          f"프레임 {sum(len(s) for _, s in mine)}, 시연 S2 비율 {100 * revisit_rate:.2f}%",
          flush=True)

    vlm = load_split_vlm(args.layers_on_first) if args.split_gpus else load_vlm()
    minimum = min_new_tokens_for(CONDITION)
    out = args.out / "frames"
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for number, (record, steps) in enumerate(mine, start=1):
        target = out / f"{record['episode_id']}.npz"
        if target.exists():
            continue
        episode, goal = record["episode_id"], record["object_category"]
        frames = np.load(HABITAT_ROOT / record["rgb_path"], mmap_mode="r")
        pose = np.load(HABITAT_ROOT / record["pose_path"])
        graph = place_graph.build(pose[:, :3])

        # The room of every node, asked of the frame that made it (design document, 4.3).
        answers = []
        for start in range(0, len(graph.node_step), ROOM_BATCH):
            made = [int(t) for t in graph.node_step[start:start + ROOM_BATCH]]
            answers += [item.generated_text for item in encode_batch(
                vlm, np.asarray(frames[made]), goal, episode, made, with_cot=False,
                template=place_graph.ROOM_QUESTION, min_new_tokens=0, greedy=True,
                max_new_tokens=ROOM_TOKENS)]
        rooms = [place_graph.normalize_room(a) for a in answers]
        prefixes = place_graph.describe(pose, rooms, graph)

        texts = {"A": [None] * len(steps),
                 "C": [prefixes[t].text for t in steps],
                 "D": [place_graph.shuffled(prefixes[t], episode, t, revisit_rate).text
                       for t in steps]}
        feats = {(c, s): [None] * len(steps) for c in CONDITIONS for s in SUMMARIES}
        replies = {c: [None] * len(steps) for c in CONDITIONS}
        for condition in CONDITIONS:
            # Frames sharing a prompt go together; A's all do, C's and D's mostly do not.
            groups = defaultdict(list)
            for k, text in enumerate(texts[condition]):
                groups[text].append(k)
            for text, members in groups.items():
                for start in range(0, len(members), args.batch):
                    part = members[start:start + args.batch]
                    index = [steps[k] for k in part]
                    encoded = encode_batch(vlm, np.asarray(frames[index]), goal, episode, index,
                                           with_cot=False, min_new_tokens=minimum,
                                           prefix_text=text)
                    for k, item in zip(part, encoded):
                        last, mean = summarise(item)
                        feats[(condition, "last")][k] = last
                        feats[(condition, "mean")][k] = mean
                        replies[condition][k] = item.generated_text

        chosen_prefix = [prefixes[t] for t in steps]
        np.savez(target, steps=np.asarray(steps), scene=record["scene_key"],
                 role=role[record["scene_key"]], goal=goal,
                 node_answers=np.asarray(answers), node_rooms=np.asarray(rooms),
                 revisit=np.asarray([p.revisit for p in chosen_prefix]),
                 rooms=np.asarray([len(p.rooms) for p in chosen_prefix]),
                 first_room=np.asarray([0 if not p.rooms else
                                        1 + place_graph.ROOM_NAMES.index(p.rooms[0][0])
                                        for p in chosen_prefix]),
                 first_dir=np.asarray([-1 if not p.rooms else
                                       place_graph.DIRECTIONS.index(p.rooms[0][1])
                                       for p in chosen_prefix]),
                 text_C=np.asarray(texts["C"]), text_D=np.asarray(texts["D"]),
                 **{f"reply_{c}": np.asarray(replies[c]) for c in CONDITIONS},
                 **{f"{c}_{s}": np.stack(v).astype(np.float16) for (c, s), v in feats.items()})
        if number % 10 == 0 or number == len(mine):
            print(f"[encode] {number}/{len(mine)} 궤적  {time.time() - started:.0f}s  "
                  f"예: {texts['C'][-1]!r} → A {replies['A'][-1]!r} / C {replies['C'][-1]!r}",
                  flush=True)
    print(f"[encode] 끝: {time.time() - started:.0f}s", flush=True)
    return 0


def probe(args) -> int:
    files = sorted((args.out / "frames").glob("*.npz"))
    data = defaultdict(lambda: defaultdict(list))
    answers = []
    for path in files:
        item = np.load(path)
        role = str(item["role"])
        n = len(item["steps"])
        for key in ("revisit", "rooms", "first_room", "first_dir"):
            data[role][key].append(item[key].astype(np.int64))
        for c in CONDITIONS:
            for s in SUMMARIES:
                data[role][f"{c}_{s}"].append(item[f"{c}_{s}"])
        data[role]["traj"].append(np.full(n, len(data[role]["traj"])))
        answers += list(item["node_rooms"])
    cat = {r: {k: np.concatenate(v) for k, v in d.items()} for r, d in data.items()}
    for r in ("train", "val", "test"):
        print(f"[probe] {r}: 프레임 {len(cat[r]['traj'])}, 재방문 {cat[r]['revisit'].sum()}")
    names, counts = np.unique(answers, return_counts=True)
    print("[probe] 노드 방 이름: " + ", ".join(f"{n} {c}" for n, c in
                                          sorted(zip(names, counts), key=lambda x: -x[1])))

    basis = pca.load(HABITAT_ROOT / "embeddings" / f"pr2l_habitat_web_hd_{CONDITION}" / "pca.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)

    def features(d, name, reduced):
        raw = d[name].astype(np.float32)                         # [n, 2, 4096]
        if reduced:
            return np.concatenate([basis.apply(raw[:, layer]) for layer in range(raw.shape[1])],
                                  axis=1)
        return raw.reshape(len(raw), -1)

    tasks = {"revisit": 2, "first_room": 10, "first_dir": 4, "rooms": 4}
    report = {}
    for task, classes in tasks.items():
        for reduced in (False, True):
            for summary in SUMMARIES:
                key = f"{task}/{'pca' if reduced else 'raw'}/{summary}"
                report[key] = {}
                for condition in CONDITIONS:
                    sets = {}
                    for r in ("train", "val", "test"):
                        keep = np.flatnonzero(cat[r][task] >= 0)
                        part = {k: v[keep] for k, v in cat[r].items()}
                        sets[r] = (features(part, f"{condition}_{summary}", reduced),
                                   part[task], part["traj"])
                    tx_np, ty_np, _ = sets["train"]
                    mean, std = tx_np.mean(0), tx_np.std(0)
                    std = np.where(std < 1e-3, 1.0, std)
                    to = lambda a: torch.from_numpy((a - mean) / std).float().to(device)
                    tx, ty = to(tx_np), torch.from_numpy(ty_np).to(device)
                    best = None
                    for lam in LAMBDAS:
                        model = fit(tx, ty, classes, lam)
                        with torch.no_grad():
                            pred = model(to(sets["val"][0])).argmax(1).cpu().numpy()
                        score = balanced_accuracy(pred, sets["val"][1], classes)
                        if best is None or (np.isfinite(score) and not score <= best[0]):
                            best = (score, lam, model)
                    _, lam, model = best
                    test_x, test_y, test_traj = sets["test"]
                    with torch.no_grad():
                        pred = model(to(test_x)).argmax(1).cpu().numpy()
                    groups = [np.flatnonzero(test_traj == k) for k in np.unique(test_traj)]
                    # The same resamples for every condition, so their differences are paired.
                    draws = np.random.default_rng(1).integers(0, len(groups),
                                                              (BOOTSTRAP, len(groups)))
                    boots = np.asarray([balanced_accuracy(
                        pred[np.concatenate([groups[i] for i in draw])],
                        test_y[np.concatenate([groups[i] for i in draw])], classes)
                        for draw in draws])
                    report[key][condition] = {
                        "bacc": balanced_accuracy(pred, test_y, classes),
                        "sigma": float(np.nanstd(boots)), "lambda": lam,
                        "boots": boots, "test": int(len(test_y))}
                c, d, a = (report[key][x] for x in ("C", "D", "A"))
                gap = c["bacc"] - d["bacc"]
                paired = float(np.nanstd(c["boots"] - d["boots"]))
                report[key]["C_minus_D_sigma"] = gap / max(paired, 1e-9)
                print(f"[probe] {key:26s} A {a['bacc']:.3f}  C {c['bacc']:.3f}  "
                      f"D {d['bacc']:.3f}  |  C−D {gap:+.3f} ({gap / max(paired, 1e-9):+.1f}σ)  "
                      f"시험 {c['test']}", flush=True)
    for value in report.values():
        for condition in CONDITIONS:
            value[condition].pop("boots")
    target = args.out / "report.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"[probe] {target} 저장")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["encode", "probe"])
    parser.add_argument("--out", type=Path, default=Path("/data/topovlm/probes/prefix"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="trajectories, for a smoke run")
    parser.add_argument("--split-gpus", action="store_true",
                        help="float16 over two cards, for rtx2080 (no bfloat16, 11 GB each)")
    parser.add_argument("--layers-on-first", type=int, default=14)
    args = parser.parse_args()
    return encode(args) if args.mode == "encode" else probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
