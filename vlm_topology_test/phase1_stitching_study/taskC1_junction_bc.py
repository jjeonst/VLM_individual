"""Task C-1 (simple) — Junction-stratified BC action accuracy.

Does the trained PR2L behavior-cloning policy fail specifically at DECISION
points? We reuse the graph cache the policy was trained on (each node already
carries the PR2L representation AND the expert action `node_actions`), run the
frozen policy per node, and stratify per-node accuracy by whether the node is
an *inflection* point.

Decision point = "inflection" node, exactly as the trainer defines it
(training/runner.py): node_action[i] != node_action[i-1] — the expert changes
action (e.g. FORWARD -> TURN), i.e. a heading re-decision at a junction.
Non-decision node = action continues (corridor forward, or mid-rotation).

We compare, per stratum:
  - policy action accuracy
  - majority-class baseline (accuracy of always predicting that stratum's most
    common action) -- controls for the corridor/forward class imbalance
so a policy that is competent only where topology does NOT matter shows up as
"inflection accuracy ~ majority baseline, corridor accuracy >> baseline".

No VLM, no simulator, no frame alignment: everything needed is in the cache.
Runs on CPU in a couple minutes.

Usage:
  python vlm_topology_test/taskC1_junction_bc.py --limit 0   # all graphs
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

CKPT = "/data/topovlm/checkpoints/pr2l_hm3d_bc/seed_42/model.pt"
GRAPH_DIR = "/data/topovlm/habitat/graphs/pr2l_hm3d_bc/train"
OUT = Path(__file__).resolve().parents[0] / "results" / "taskC1_junction_bc.json"
ACTION_NAMES = {0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT"}


def load_policy(device, ckpt=CKPT):
    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from configs.schema import PolicyConfig
    from policies.graph_policy import GraphTransformerPolicy
    state = torch.load(ckpt, map_location=device)
    pcfg = state["config"]["model"]["policy"]
    cfg = PolicyConfig(
        type=pcfg["type"], input_dim=pcfg["input_dim"], hidden_dim=pcfg["hidden_dim"],
        transformer_heads=pcfg["transformer_heads"], transformer_layers=pcfg["transformer_layers"],
        dropout=pcfg["dropout"], num_actions=pcfg["num_actions"],
        prediction_target=pcfg["prediction_target"], max_positions=pcfg["max_positions"],
        causal=pcfg.get("causal", False),
    )
    policy = GraphTransformerPolicy(cfg).to(device)
    policy.load_state_dict(state["model"])
    policy.eval()
    print(f"[policy] causal={cfg.causal}", flush=True)
    return policy, state.get("metrics", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all graphs")
    ap.add_argument("--causal", action="store_true",
                    help="predict action[i] from PAST-only nodes[0..i] (rollout condition), "
                         "instead of the full bidirectional trajectory")
    ap.add_argument("--ckpt", default=CKPT, help="policy checkpoint (default = bidirectional baseline)")
    ap.add_argument("--tag", default="", help="output filename suffix, e.g. _causalmodel")
    args = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, train_metrics = load_policy(device, ckpt=args.ckpt)
    print(f"[policy] loaded on {device}; train metrics: {train_metrics}", flush=True)

    files = sorted(f for f in glob.glob(GRAPH_DIR + "/*.npz"))
    if args.limit:
        files = files[: args.limit]
    print(f"[data] {len(files)} graph episodes", flush=True)

    # per-node records: (correct, expert_action, is_inflection, pred_action)
    correct_infl = tot_infl = 0
    correct_non = tot_non = 0
    infl_actions, non_actions = [], []          # for per-stratum majority baseline
    infl_pred_forward = 0                         # inflection nodes the policy calls FORWARD
    confusion = np.zeros((4, 4), dtype=np.int64)  # [expert, pred] over inflection nodes
    n_nodes = 0

    with torch.inference_mode():
        for k, f in enumerate(files):
            g = np.load(f)
            nodes = g["nodes"].astype("float32")          # (T,16,1024)
            acts = g["node_actions"].astype("int64")      # (T,)
            mask = g["action_mask"].astype(bool)          # (T,)
            T = nodes.shape[0]
            if T == 0:
                continue
            gn_full = torch.from_numpy(nodes)[None].to(device)     # [1,T,16,1024]
            if args.causal:
                # predict action[i] from PAST-only prefix nodes[0..i] (rollout condition)
                pred = np.empty(T, dtype=np.int64)
                for i in range(T):
                    gp = gn_full[:, : i + 1]
                    gm = torch.ones(1, i + 1, dtype=torch.bool, device=device)
                    li = policy(gp, gm)                            # [1,i+1,4]
                    pred[i] = int(li[0, -1].argmax(-1).item())     # last (current) node
            else:
                gm = torch.ones(1, T, dtype=torch.bool, device=device)
                logits = policy(gn_full, gm)                       # [1,T,4] bidirectional
                pred = logits.argmax(-1)[0].cpu().numpy()          # (T,)

            infl = np.zeros(T, dtype=bool)
            infl[1:] = acts[1:] != acts[:-1]                       # trainer's inflection def
            for i in range(1, T):   # i>=1: inflection undefined at node 0 (no previous action)
                if not mask[i]:
                    continue
                n_nodes += 1
                ok = bool(pred[i] == acts[i])
                if infl[i]:
                    tot_infl += 1; correct_infl += int(ok)
                    infl_actions.append(int(acts[i]))
                    confusion[acts[i], pred[i]] += 1
                    if pred[i] == 1:
                        infl_pred_forward += 1
                else:
                    tot_non += 1; correct_non += int(ok)
                    non_actions.append(int(acts[i]))
            if (k + 1) % 1000 == 0:
                print(f"  {k+1}/{len(files)} episodes", flush=True)

    def majority_baseline(actions):
        if not actions:
            return 0.0, None
        vals, counts = np.unique(actions, return_counts=True)
        top = int(vals[counts.argmax()])
        return float(counts.max() / len(actions)), top

    infl_maj, infl_maj_act = majority_baseline(infl_actions)
    non_maj, non_maj_act = majority_baseline(non_actions)
    acc_infl = correct_infl / tot_infl if tot_infl else 0.0
    acc_non = correct_non / tot_non if tot_non else 0.0

    result = {
        "checkpoint": CKPT,
        "inference_mode": "causal_past_only" if args.causal else "bidirectional_full_trajectory",
        "episodes": len(files),
        "nodes_evaluated": n_nodes,
        "decision_point_definition": "inflection: node_action[i] != node_action[i-1] (trainer def)",
        "inflection_decision_nodes": {
            "count": tot_infl,
            "policy_accuracy": round(acc_infl, 3),
            "majority_baseline": round(infl_maj, 3),
            "majority_action": ACTION_NAMES.get(infl_maj_act),
            "policy_minus_baseline": round(acc_infl - infl_maj, 3),
            "pct_predicted_FORWARD": round(100 * infl_pred_forward / tot_infl, 1) if tot_infl else 0.0,
        },
        "non_inflection_corridor_nodes": {
            "count": tot_non,
            "policy_accuracy": round(acc_non, 3),
            "majority_baseline": round(non_maj, 3),
            "majority_action": ACTION_NAMES.get(non_maj_act),
            "policy_minus_baseline": round(acc_non - non_maj, 3),
        },
        "accuracy_gap_corridor_minus_decision": round(acc_non - acc_infl, 3),
        "inflection_confusion_expert_x_pred": {
            ACTION_NAMES[e]: {ACTION_NAMES[p]: int(confusion[e, p]) for p in range(4)}
            for e in range(4)
        },
    }
    suffix = ("_causal" if args.causal else "") + args.tag
    out_path = OUT.with_name(f"taskC1_junction_bc{suffix}.json") if suffix else OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print("\n=== C-1 RESULT (junction-stratified BC accuracy) ===")
    print(json.dumps({k: v for k, v in result.items()
                      if k != "inflection_confusion_expert_x_pred"}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
