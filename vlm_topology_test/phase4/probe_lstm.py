"""Does the trained policy's LSTM already remember the path? (TopoPrompt step 1-a)

The design (TopoPrompt document, H0) assumes that path memory is missing from the system: the
VLM sees one frame at a time, and if the policy's LSTM had the path well in hand there would be
little for a prefix sentence to add. This measures how much of the path a linear readout can
recover from the LSTM state, against what it can recover from the step's own input.

**extract** (GPU). Feeds every demonstration through the checkpoint with teacher forcing --
the human's actions, the stored embeddings -- and keeps two vectors per step:

* `h` -- the top LSTM layer's output, 2048 wide. This is what the action head reads.
* `x` -- the LSTM's input at that step, 1152 wide: the frame summary and the four side
  channels. Everything the policy knows *without* memory is in here.

**probe** (GPU or CPU). Labels every step from the demonstration's poses and fits a
class-balanced logistic regression from each vector to each label, on training scenes, scored
by balanced accuracy on scenes it never saw. A label counts as *remembered* when `h` beats the
memoryless baseline `x + t` (input plus elapsed-step features) by more than 2 sigma, sigma from
a bootstrap over held-out trajectories. Both sides get `t`: revisits and explored area grow with
time, and a log-time one-hot is an easier linear read of elapsed time than whatever `h` keeps of
it, so `h` alone against `x + t` mostly measured that (first full run: revisit h 0.67 < x+t 0.84).
The criterion is therefore `h + t` against `x + t`; `t` alone is reported as the floor.

    revisit      bool    back at a place left long ago (slot S2, `place_graph.build`)
    past30       4-way   which quadrant, relative to the heading, the agent stood in 30 steps ago
    nodes        3-way   how much has been explored: places made so far, <=3 / 4-8 / >=9
    prev_action  3-way   sanity check: it is an input, so `x` must read it nearly perfectly.
                         Forward / left / right only: STOP ends a demonstration, so it is never
                         the previous action and a 4-way label left one class empty (NaN).
    shuffled     bool    revisit labels permuted across steps: every feature must sit at 0.5

**--test-on FOLDER** keeps the probes trained (and their regularisation chosen) on the
demonstrations but scores them on the policy's own rollouts, saved by `evaluate.py
--save-states`. Human demonstrations almost never double back -- 2.4 % of steps are revisits --
so "the LSTM does not hold revisits" was a statement about a label the demos barely exercise;
a policy that loops exercises it far more. Every rollout is a test trajectory.
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
import dataset  # noqa: E402
import place_graph  # noqa: E402
from policy import NavigationPolicy  # noqa: E402

HABITAT_ROOT = Path("/data/topovlm/habitat")
CHECKPOINT_ROOT = Path("/data/topovlm/checkpoints/pr2l_phase4")
PAST = 30                    # steps back for the past30 label
PAST_MIN_M = 1.0             # ignore steps that have barely moved since then
NODE_BINS = (3, 8)           # nodes <=3 / 4-8 / >=9
TEST_FRACTION = 0.25         # of scenes
VAL_FRACTION = 0.15          # of training scenes, for choosing the regularisation
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1)
MAX_TRAIN_STEPS = 200_000
BOOTSTRAP = 1000


def split_scenes(records: list[dict], seed: int = 0) -> dict[str, str]:
    scenes = sorted({r["scene_key"] for r in records})
    order = np.random.default_rng(seed).permutation(len(scenes))
    test = int(round(len(scenes) * TEST_FRACTION))
    val = int(round((len(scenes) - test) * VAL_FRACTION))
    role = {}
    for rank, index in enumerate(order):
        role[scenes[index]] = "test" if rank < test else "val" if rank < test + val else "train"
    return role


@torch.inference_mode()
def extract(args) -> int:
    embedding_root = HABITAT_ROOT / "embeddings" / f"pr2l_habitat_web_hd_{args.condition}"
    records = dataset.read_manifest(embedding_root)
    if args.limit:
        records = records[:: max(1, len(records) // args.limit)][: args.limit]
    role = split_scenes(dataset.read_manifest(embedding_root))

    checkpoint = torch.load(CHECKPOINT_ROOT / f"{args.run_name}.pt", map_location="cpu")
    policy = NavigationPolicy(heading_encoding=checkpoint.get("heading_encoding", "angle"),
                              num_heads=checkpoint.get("num_heads", 1),
                              token_dim=checkpoint.get("token_dim", 2048),
                              lstm_hidden=checkpoint.get("lstm_hidden", 2048),
                              lstm_layers=checkpoint.get("lstm_layers", 2),
                              side_dim=checkpoint.get("side_dim", 32))
    policy.load_state_dict(checkpoint["policy"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device).eval()

    out = args.out / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for number, record in enumerate(records, start=1):
        target = out / f"{record['episode_id']}.npz"
        if target.exists():
            continue
        traj = dataset.load_trajectory(record, embedding_root)
        one = lambda a: torch.from_numpy(np.asarray(a))[None].to(device)
        tokens = one(traj.tokens).float()
        summary = policy.summary(tokens, one(traj.token_padding))
        x = torch.cat([summary, policy.side(one(traj.gps), one(traj.compass),
                                            one(traj.previous_action), one(traj.goal))], dim=-1)
        h, _ = policy.memory(x)
        pose = np.load(dataset.trajectory_paths(record, embedding_root)[2])
        np.savez(target, h=h[0].half().cpu().numpy(), x=x[0].half().cpu().numpy(),
                 pose=pose.astype(np.float32), actions=traj.actions,
                 scene=record["scene_key"], role=role[record["scene_key"]])
        if number % 200 == 0:
            print(f"[extract] {number}/{len(records)}  {time.time() - started:.0f}s", flush=True)
    print(f"[extract] 끝: {len(records)} 궤적, {time.time() - started:.0f}s", flush=True)
    return 0


def labels_for(pose: np.ndarray, actions: np.ndarray) -> dict[str, np.ndarray]:
    """Per-step labels; -1 where a label is undefined for that step."""
    steps = len(pose)
    position, yaw = pose[:, :3], pose[:, 3]
    graph = place_graph.build(position)
    out = {"revisit": graph.revisit.astype(np.int64)}

    past = np.full(steps, -1, dtype=np.int64)
    if steps > PAST:
        now, then = position[PAST:], position[:-PAST]
        moved = np.hypot(now[:, 0] - then[:, 0], now[:, 2] - then[:, 2]) >= PAST_MIN_M
        angle = place_graph.relative_bearing(now, yaw[PAST:], then)
        past[PAST:] = np.where(moved, place_graph.quadrant(angle), -1)
    out["past30"] = past

    out["nodes"] = np.digitize(graph.count, np.array(NODE_BINS) + 0.5).astype(np.int64)
    prev = np.full(steps, -1, dtype=np.int64)
    prev[1:] = actions[:-1] - 1
    out["prev_action"] = prev
    return out


def time_features(steps: int) -> np.ndarray:
    """Elapsed steps as one-hot log2 bins: what a memoryless model could still know."""
    bins = np.minimum(np.floor(np.log2(np.arange(steps) + 1)).astype(int), 11)
    return np.eye(12, dtype=np.float32)[bins]


def fit(features: torch.Tensor, labels: torch.Tensor, classes: int, lam: float) -> torch.nn.Linear:
    """Class-balanced multinomial logistic regression, L2 penalty, full-batch L-BFGS."""
    counts = torch.bincount(labels, minlength=classes).float().clamp(min=1)
    weight = (len(labels) / (classes * counts)).to(features.device)
    model = torch.nn.Linear(features.shape[1], classes).to(features.device)
    optimiser = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=200,
                                  history_size=20, line_search_fn="strong_wolfe")

    def closure():
        optimiser.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(features), labels, weight=weight)
        loss = loss + lam * model.weight.pow(2).sum()
        loss.backward()
        return loss

    optimiser.step(closure)
    return model


def balanced_accuracy(pred: np.ndarray, true: np.ndarray, classes: int) -> float:
    """Mean per-class recall over *all* classes; NaN when one is absent from `true`.

    Averaging only over the classes present would score "always no" at 1.0 on a test set with
    no revisits in it -- which is how the smoke run's shuffled control came out at 0.994.
    """
    if any(not np.any(true == c) for c in range(classes)):
        return float("nan")
    return float(np.mean([np.mean(pred[true == c] == c) for c in range(classes)]))


def probe(args) -> int:
    root = args.out / args.run_name
    files = [(path, None) for path in sorted(root.glob("*.npz"))]
    if args.test_on is not None:
        files += [(path, "test") for path in sorted(args.test_on.glob("*.npz"))]
    data = defaultdict(lambda: defaultdict(list))   # role -> key -> list
    for path, forced in files:
        item = np.load(path)
        role = forced or str(item["role"])
        if forced is None and args.test_on is not None and role == "test":
            continue
        lab = labels_for(item["pose"], item["actions"])
        steps = len(item["actions"])
        data[role]["h"].append(item["h"])
        data[role]["x"].append(item["x"])
        data[role]["t"].append(time_features(steps))
        data[role]["traj"].append(np.full(steps, len(data[role]["traj"])))
        for name, value in lab.items():
            data[role][name].append(value)
    cat = {role: {k: np.concatenate(v) for k, v in parts.items()} for role, parts in data.items()}
    for role in ("train", "val", "test"):
        print(f"[probe] {role}: 궤적 {len(data[role]['traj'])}, 스텝 {len(cat[role]['h'])}, "
              f"재방문 스텝 {100 * np.mean(cat[role]['revisit'] > 0):.1f}%")

    rng = np.random.default_rng(0)
    # Negative control: revisit labels shuffled within each split.
    for role in cat:
        cat[role]["shuffled"] = rng.permutation(cat[role]["revisit"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_sets = {
        "x":   lambda d: d["x"].astype(np.float32),
        "x+t": lambda d: np.concatenate([d["x"].astype(np.float32), d["t"]], axis=1),
        "t":   lambda d: d["t"],
        "h":   lambda d: d["h"].astype(np.float32),
        "h+t": lambda d: np.concatenate([d["h"].astype(np.float32), d["t"]], axis=1),
    }
    tasks = {"revisit": 2, "past30": 4, "nodes": 3, "prev_action": 3, "shuffled": 2}
    report = {}
    for task, classes in tasks.items():
        report[task] = {}
        for name, make in feature_sets.items():
            sets = {}
            for role in ("train", "val", "test"):
                keep = np.flatnonzero(cat[role][task] >= 0)
                if role == "train" and len(keep) > MAX_TRAIN_STEPS:
                    keep = np.sort(rng.choice(keep, MAX_TRAIN_STEPS, replace=False))
                # Index before widening to float32: the full split would not fit.
                part = {k: cat[role][k][keep] for k in ("h", "x", "t")}
                sets[role] = (make(part), cat[role][task][keep], cat[role]["traj"][keep])
            train_x, train_y, _ = sets["train"]
            # Columns that never vary in training (a time bin no training step reached) are
            # left unscaled: dividing by ~0 sent the held-out values to 1e6 in the smoke run.
            mean, std = train_x.mean(0), train_x.std(0)
            std = np.where(std < 1e-3, 1.0, std)
            to = lambda a: torch.from_numpy((a - mean) / std).to(device)
            tx, ty = to(train_x), torch.from_numpy(train_y).to(device)

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
            score = balanced_accuracy(pred, test_y, classes)
            # Bootstrap over held-out trajectories: steps of one trajectory are not independent.
            groups = [np.flatnonzero(test_traj == k) for k in np.unique(test_traj)]
            boots = []
            for _ in range(BOOTSTRAP):
                take = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                boots.append(balanced_accuracy(pred[take], test_y[take], classes))
            boots = np.asarray(boots)
            sigma = float(np.nanstd(boots)) if np.isfinite(boots).any() else float("nan")
            report[task][name] = {"bacc": score, "sigma": sigma, "lambda": lam,
                                  "test_steps": int(len(test_y)),
                                  "test_positive": int(np.sum(test_y > 0)) if classes == 2 else None}
            print(f"[probe] {task:11s} {name:4s} bAcc {score:.3f} ± {sigma:.3f}  (λ {lam:g}, "
                  f"시험 스텝 {len(test_y)})", flush=True)
        h, base = report[task]["h+t"], report[task]["x+t"]
        gap = h["bacc"] - base["bacc"]
        report[task]["ht_minus_xt_sigma"] = gap / max(np.hypot(h["sigma"], base["sigma"]), 1e-9)
        print(f"[probe] {task:11s} (h+t) − (x+t) = {gap:+.3f}  "
              f"({report[task]['ht_minus_xt_sigma']:+.1f}σ)\n", flush=True)

    suffix = f"_on_{args.test_on.name}" if args.test_on is not None else ""
    target = args.out / f"{args.run_name}{suffix}_report.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"[probe] {target} 저장")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["extract", "probe"])
    parser.add_argument("--run-name", default="stageB_nocotMin0C3")
    parser.add_argument("--condition", default="nocot_min0")
    parser.add_argument("--out", type=Path, default=Path("/data/topovlm/probes/lstm"))
    parser.add_argument("--limit", type=int, default=None, help="trajectories, for a smoke run")
    parser.add_argument("--test-on", type=Path, default=None,
                        help="score on these rollouts (evaluate.py --save-states) instead of "
                             "the held-out demonstrations")
    args = parser.parse_args()
    return extract(args) if args.mode == "extract" else probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
