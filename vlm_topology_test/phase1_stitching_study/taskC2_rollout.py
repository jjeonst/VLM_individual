"""Task C-2 — closed-loop policy rollout in Habitat (stitching harness).

Runs the trained PR2L BC policy CLOSED-LOOP: at each step the agent's RGB is
encoded online with the goal prompt (PR2L representation + PCA), compressed into
the same sequential-similarity graph the policy was trained on, and the policy
picks the next action -- until STOP or max_steps. Success = the agent STOPs
within `success_distance` (geodesic) of a goal viewpoint.

This is the machinery for C-2. With the current single-object data it measures
the IN-DISTRIBUTION rollout (start -> object): does the policy, whose causal
turn-accuracy we saw drop in C-1, actually reach the goal? A shortest-path
follower gives the achievable upper bound. True cross-object stitching
(start_A -> goal_B) plugs in once multi-object data (taskC C0-C4) exists: just
feed a start_A pose with goal_B's viewpoints + goal_text.

Modes:
  --smoke     : habitat only (sim + RGB render + geodesic + follower), no VLM
  --mode both : policy rollout AND follower upper bound (default)

Heavy (7B VLM in the loop + Habitat render) -> slurm rtx4090.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DATA_ROOT = Path("/data/topovlm")
HABITAT_ROOT = DATA_ROOT / "habitat"
SCENE_ROOT = HABITAT_ROOT / "scene_datasets" / "hm3d_v0.2"
SCENE_DATASET_CONFIG = SCENE_ROOT / "hm3d_annotated_basis.scene_dataset_config.json"
ORIG_OBJECTNAV = HABITAT_ROOT / "datasets" / "objectnav" / "hm3d" / "v2" / "objectnav_hm3d_v2"
CKPT = "/data/topovlm/checkpoints/pr2l_hm3d_bc_causal/seed_42/model.pt"  # causal-trained (rollout-matched)
CKPT_BIDIR = "/data/topovlm/checkpoints/pr2l_hm3d_bc/seed_42/model.pt"
WEIGHTS = "/data/topovlm/vlm_weights/prismatic/prism-dinosiglip+7b"
PCA_PATH = "/data/topovlm/habitat/embeddings/pr2l_hm3d_bc/projection_pca.npz"
OUT = Path(__file__).resolve().parents[0] / "results" / "taskC2_rollout.json"

FORWARD_M = 0.25
TURN_DEG = 30.0
AGENT_HEIGHT = 0.88
AGENT_RADIUS = 0.18
ACTION_NAME = {1: "move_forward", 2: "turn_left", 3: "turn_right"}   # 0 = STOP


# ------------------------- Habitat sim (RGB) -------------------------
def make_sim(scene_glb, width=640, height=480):
    import habitat_sim
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene_glb
    backend.scene_dataset_config_file = str(SCENE_DATASET_CONFIG)
    backend.enable_physics = False

    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [height, width]
    rgb.position = [0.0, AGENT_HEIGHT, 0.0]

    agent = habitat_sim.AgentConfiguration()
    agent.height = AGENT_HEIGHT
    agent.radius = AGENT_RADIUS
    agent.sensor_specifications = [rgb]
    agent.action_space = {
        "move_forward": habitat_sim.ActionSpec("move_forward", habitat_sim.ActuationSpec(amount=FORWARD_M)),
        "turn_left": habitat_sim.ActionSpec("turn_left", habitat_sim.ActuationSpec(amount=TURN_DEG)),
        "turn_right": habitat_sim.ActionSpec("turn_right", habitat_sim.ActuationSpec(amount=TURN_DEG)),
    }
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))


def scene_glb_path(scene_key):  # scene_key e.g. "00744-1S7LAXRdDqK"
    name = scene_key.split("-", 1)[-1]
    return str(SCENE_ROOT / "train" / scene_key / f"{name}.basis.glb")


def load_scene_episodes(scene_key, limit):
    """Return list of {start_position, start_rotation, object_category, goal_positions}."""
    name = scene_key.split("-", 1)[-1]
    content = ORIG_OBJECTNAV / "train" / "content" / f"{name}.json.gz"
    if not content.exists():
        return []
    with gzip.open(content, "rt") as fh:
        d = json.load(fh)
    gbc = d.get("goals_by_category", {})
    goal_positions_by_cat = {}
    for key, goals in gbc.items():
        cat = key.split(".basis.glb_")[-1]
        pts = []
        for g in goals:
            for vp in g.get("view_points", []) or []:
                p = vp.get("agent_state", {}).get("position")
                if p is not None:
                    pts.append(np.asarray(p, dtype=np.float32))
        goal_positions_by_cat[cat] = pts
    eps = []
    for e in d["episodes"][:limit] if limit else d["episodes"]:
        cat = e["object_category"]
        eps.append({
            "episode_id": e["episode_id"],
            "start_position": np.asarray(e["start_position"], dtype=np.float32),
            "start_rotation": np.asarray(e["start_rotation"], dtype=np.float32),
            "object_category": cat,
            "goal_positions": goal_positions_by_cat.get(cat, []),
        })
    return eps


def geodesic_min(pathfinder, start, goals):
    import habitat_sim
    best = float("inf")
    for g in goals:
        sp = habitat_sim.ShortestPath()
        sp.requested_start = np.asarray(start, dtype=np.float32)
        sp.requested_end = np.asarray(g, dtype=np.float32)
        if pathfinder.find_path(sp) and np.isfinite(sp.geodesic_distance):
            best = min(best, float(sp.geodesic_distance))
    return best


def set_start(agent, ep):
    import habitat_sim
    from habitat_sim.utils.common import quat_from_coeffs
    st = agent.get_state()
    st.position = ep["start_position"]
    st.rotation = quat_from_coeffs(ep["start_rotation"])
    agent.set_state(st)
    return st


# ------------------------- PR2L online encoder -------------------------
class PR2LOnlineEncoder:
    """RGB -> (16,1024) PR2L node: 2-layer visual tokens, 4x4 pool, PCA. Same as cache."""

    def __init__(self, include_generated_text=False):
        import os
        from configs.schema import VLMConfig
        from encoders.prismatic import PrismaticEncoder
        from PIL import Image
        self.Image = Image
        cfg = VLMConfig(
            backend="prismatic", model_id="prism-dinosiglip+7b", weights_path=WEIGHTS,
            device="cuda", dtype="bfloat16", frozen=True,
            representation="pr2l_visual_tokens_last_two_layers",
            hidden_layer_indices=[-2, -1], visual_pool_grid=4, visual_bank_reduction="mean",
            projection="pca", projection_path=PCA_PATH, projection_dim=1024,
            include_generated_text=include_generated_text,
            generation_seed=0, generation_temperature=0.4, max_new_tokens=48,
            output_dim=1024, prompt_template="Would a {goal_text} be found here? Why or why not?",
        )
        _ = os
        self.enc = PrismaticEncoder(cfg)
        z = np.load(PCA_PATH)
        self.mean = z["mean"].astype("float32")             # (8192,)
        self.components = z["components"].astype("float32")  # (1024,8192)

    def encode_node(self, rgb, goal_text):
        img = self.Image.fromarray(rgb[..., :3].astype("uint8"), mode="RGB")
        tokens = np.asarray(self.enc.encode_image_goal_tokens(img, goal_text)["tokens"], dtype="float32")
        return (tokens - self.mean) @ self.components.T      # (16,1024)


# ------------------------- online policy runner -------------------------
class OnlinePolicyRunner:
    """Online sequential-similarity graph + causal policy, matching training."""

    def __init__(self, device, ckpt=CKPT, thr=0.9, max_nodes=128, min_seg=1, frame_stride=4):
        import torch
        from configs.schema import PolicyConfig
        from policies.graph_policy import GraphTransformerPolicy
        self.torch = torch
        self.device = device
        state = torch.load(ckpt, map_location=device)
        p = state["config"]["model"]["policy"]
        cfg = PolicyConfig(type=p["type"], input_dim=p["input_dim"], hidden_dim=p["hidden_dim"],
                           transformer_heads=p["transformer_heads"], transformer_layers=p["transformer_layers"],
                           dropout=p["dropout"], num_actions=p["num_actions"],
                           prediction_target=p["prediction_target"], max_positions=p["max_positions"],
                           causal=p.get("causal", False))
        print(f"[runner] policy causal={cfg.causal} frame_stride={frame_stride} from {ckpt.split('/')[-3]}", flush=True)
        self.policy = GraphTransformerPolicy(cfg).to(device).eval()
        self.policy.load_state_dict(state["model"])
        self.max_token_positions = cfg.max_positions
        self.thr, self.max_nodes, self.min_seg = thr, max_nodes, min_seg
        self.frame_stride = frame_stride  # match training: graph built over every-Nth frame
        self.reset()

    def reset(self):
        self.strided = []   # committed frames sampled every frame_stride steps (matches training input)
        self.step = 0
        self.n_nodes = 0    # last graph node count (for logging)

    @staticmethod
    def _metric(node):
        v = node.mean(axis=0)      # mean over 16 tokens -> (1024,)
        return v / max(float(np.linalg.norm(v)), 1e-8)

    def _seqsim_nodes(self, frames):
        """Sequential-similarity graph node features over a frame list, mirroring
        topology.graph_builder.build_sequential_similarity_graph (thr, max_nodes, min_seg)."""
        if not frames:
            return []
        node_feats = [frames[0].copy()]
        seg = [frames[0]]
        seg_metric = [self._metric(frames[0])]
        for f in frames[1:]:
            m = self._metric(f)
            cur = np.mean(seg_metric, axis=0)
            cur = cur / max(float(np.linalg.norm(cur)), 1e-8)
            sim = float(np.dot(m, cur))
            if sim >= self.thr or len(node_feats) >= self.max_nodes:
                seg.append(f); seg_metric.append(m); node_feats[-1] = np.mean(seg, axis=0)
            elif len(seg) < self.min_seg and len(node_feats) > 1:
                seg.append(f); seg_metric.append(m); node_feats[-1] = np.mean(seg, axis=0)
            else:
                seg = [f]; seg_metric = [m]; node_feats.append(f.copy())
        return node_feats

    def act(self, node, allow_stop=True):
        torch = self.torch
        # graph = committed strided frames + current frame (fresh each step); seq-sim compressed
        feats = self._seqsim_nodes(self.strided + [node])
        max_nodes_fit = self.max_token_positions // 16
        if len(feats) > max_nodes_fit:
            feats = feats[-max_nodes_fit:]
        self.n_nodes = len(feats)
        gn = torch.from_numpy(np.stack(feats, axis=0)[None].astype("float32")).to(self.device)  # [1,N,16,1024]
        gm = torch.ones(1, gn.shape[1], dtype=torch.bool, device=self.device)
        with torch.inference_mode():
            logits = self.policy(gn, gm)[0, -1]   # [4]
        if not allow_stop:
            logits = logits.clone()
            logits[0] = float("-inf")             # forbid STOP (early-step navigation bootstrap)
        # commit current frame into the strided graph every frame_stride steps (training-matched density)
        if self.step % self.frame_stride == 0:
            self.strided.append(node)
        self.step += 1
        return int(logits.argmax(-1).item())


# ------------------------- rollouts -------------------------
def rollout_policy(sim, agent, encoder, runner, ep, max_steps, success_dist, min_steps=10, verbose=False):
    runner.reset()
    set_start(agent, ep)
    goals = ep["goal_positions"]
    if verbose:
        print(f"    [policy] ep start, {len(goals)} goal viewpoints, computing start_geo...", flush=True)
    start_geo = geodesic_min(sim.pathfinder, agent.get_state().position, goals)
    if verbose:
        print(f"    [policy] start_geo={_f(start_geo)}, stepping (min_steps={min_steps})...", flush=True)
    fwd = 0
    stopped = False
    for step in range(max_steps):
        obs = sim.get_sensor_observations()
        node = encoder.encode_node(np.asarray(obs["rgb"]), ep["object_category"])
        a = runner.act(node, allow_stop=(step >= min_steps))   # forbid immediate STOP (1-node -> STOP artifact)
        if verbose and (step < 3 or step % 25 == 0):
            print(f"    [policy] step {step}: action={a} nodes={runner.n_nodes} strided={len(runner.strided)}", flush=True)
        if a == 0:
            stopped = True
            break
        sim.step(ACTION_NAME[a])
        if a == 1:
            fwd += 1
    final_geo = geodesic_min(sim.pathfinder, agent.get_state().position, goals)
    success = bool(stopped and final_geo <= success_dist)
    path_len = fwd * FORWARD_M
    spl = (success * start_geo / max(start_geo, path_len, 1e-6)) if np.isfinite(start_geo) else 0.0
    return {"success": success, "stopped": stopped, "start_geo": _f(start_geo),
            "final_geo": _f(final_geo), "steps": fwd, "spl": round(float(spl), 3)}


def rollout_follower(sim, agent, ep, max_steps, success_dist):
    import habitat_sim
    set_start(agent, ep)
    goals = ep["goal_positions"]
    start = agent.get_state().position
    # nearest goal viewpoint as the follower target
    best_g, best_d = None, float("inf")
    for g in goals:
        sp = habitat_sim.ShortestPath(); sp.requested_start = np.asarray(start, np.float32); sp.requested_end = np.asarray(g, np.float32)
        if sim.pathfinder.find_path(sp) and sp.geodesic_distance < best_d:
            best_d, best_g = float(sp.geodesic_distance), np.asarray(g, np.float32)
    if best_g is None:
        return {"success": False, "reason": "no_path", "start_geo": _f(best_d)}
    follower = habitat_sim.GreedyGeodesicFollower(
        sim.pathfinder, agent, goal_radius=success_dist,
        forward_key="move_forward", left_key="turn_left", right_key="turn_right", stop_key="stop")
    fwd = 0
    stopped = False
    for _ in range(max_steps):
        try:
            act = follower.next_action_along(best_g)
        except Exception:
            act = None
        if act is None or act == "stop":
            stopped = True
            break
        sim.step(act)
        if act == "move_forward":
            fwd += 1
    final_geo = geodesic_min(sim.pathfinder, agent.get_state().position, goals)
    success = bool(stopped and final_geo <= success_dist)
    spl = (success * best_d / max(best_d, fwd * FORWARD_M, 1e-6)) if np.isfinite(best_d) else 0.0
    return {"success": success, "stopped": stopped, "start_geo": _f(best_d),
            "final_geo": _f(final_geo), "steps": fwd, "spl": round(float(spl), 3)}


def _f(x):
    return None if (x is None or not np.isfinite(x)) else round(float(x), 3)


def summarize(rollouts):
    ok = [r for r in rollouts if "success" in r]
    n = len(ok)
    if n == 0:
        return {"n": 0}
    return {"n": n,
            "success_rate": round(sum(r["success"] for r in ok) / n, 3),
            "spl": round(sum(r.get("spl", 0.0) for r in ok) / n, 3),
            "stop_rate": round(sum(r.get("stopped", False) for r in ok) / n, 3),
            "mean_final_geo": round(float(np.mean([r["final_geo"] for r in ok if r.get("final_geo") is not None])), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--episodes-per-scene", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--min-steps", type=int, default=10, help="forbid STOP before this step (1-node->STOP artifact)")
    ap.add_argument("--success-dist", type=float, default=1.0)
    ap.add_argument("--mode", choices=["both", "policy", "follower"], default="both")
    ap.add_argument("--gen-text", action="store_true", help="include VLM generated answer (slower, matches cache)")
    ap.add_argument("--smoke", action="store_true", help="habitat only, no VLM/policy")
    ap.add_argument("--verbose", action="store_true", help="per-step progress logging")
    ap.add_argument("--ckpt", default=CKPT, help="policy checkpoint (default = causal-trained)")
    ap.add_argument("--frame-stride", type=int, default=4,
                    help="build the online graph over every-Nth frame (match training frame_stride=4)")
    args = ap.parse_args()

    # only scenes that actually have ObjectNav episodes (145 of 800 scene dirs)
    content_names = {p.name.split(".")[0] for p in (ORIG_OBJECTNAV / "train" / "content").glob("*.json.gz")}
    scene_keys = [p.name for p in sorted((SCENE_ROOT / "train").iterdir())
                  if p.is_dir() and p.name.split("-", 1)[-1] in content_names][: args.scenes]
    print(f"[c2] {len(scene_keys)} scenes, {args.episodes_per_scene} eps/scene, "
          f"success_dist={args.success_dist}m, mode={'SMOKE' if args.smoke else args.mode}", flush=True)

    encoder = runner = None
    if not args.smoke and args.mode in ("both", "policy"):
        import torch
        device = torch.device("cuda")
        encoder = PR2LOnlineEncoder(include_generated_text=args.gen_text)
        runner = OnlinePolicyRunner(device, ckpt=args.ckpt, frame_stride=args.frame_stride)
        print("[c2] policy + PR2L online encoder ready", flush=True)

    policy_rollouts, follower_rollouts, smoke_rows = [], [], []
    for sk in scene_keys:
        glb = scene_glb_path(sk)
        if not Path(glb).exists():
            print(f"  [skip] {sk} no glb", flush=True); continue
        eps = load_scene_episodes(sk, args.episodes_per_scene)
        if not eps:
            continue
        print(f"  [scene] {sk}: loading sim ({len(eps)} eps)...", flush=True)
        sim = make_sim(glb)
        agent = sim.get_agent(0)
        print(f"  [scene] {sk}: sim ready, pathfinder loaded={sim.pathfinder.is_loaded}", flush=True)
        try:
            for ei, ep in enumerate(eps):
                if not ep["goal_positions"]:
                    print(f"    ep {ei} {ep['object_category']}: no goal viewpoints, skip", flush=True)
                    continue
                if args.smoke:
                    set_start(agent, ep)
                    obs = sim.get_sensor_observations()
                    geo = geodesic_min(sim.pathfinder, agent.get_state().position, ep["goal_positions"])
                    fr = rollout_follower(sim, agent, ep, args.max_steps, args.success_dist)
                    smoke_rows.append({"scene": sk, "cat": ep["object_category"],
                                       "rgb_shape": list(np.asarray(obs["rgb"]).shape),
                                       "start_geo": _f(geo), "n_goals": len(ep["goal_positions"]),
                                       "follower": fr})
                    print(f"  [smoke] {sk} {ep['object_category']} rgb={np.asarray(obs['rgb']).shape} "
                          f"geo={_f(geo)} follower={fr['success']} steps={fr.get('steps')}", flush=True)
                    continue
                if args.mode in ("both", "policy"):
                    policy_rollouts.append({"scene": sk, **rollout_policy(sim, agent, encoder, runner, ep, args.max_steps, args.success_dist, min_steps=args.min_steps, verbose=args.verbose)})
                if args.mode in ("both", "follower"):
                    follower_rollouts.append({"scene": sk, **rollout_follower(sim, agent, ep, args.max_steps, args.success_dist)})
                if args.mode != "follower":
                    print(f"  {sk} {ep['object_category']}: policy={policy_rollouts[-1]['success'] if policy_rollouts else '-'} "
                          f"steps={policy_rollouts[-1].get('steps') if policy_rollouts else '-'}", flush=True)
        finally:
            sim.close()

    result = {
        "config": {"scenes": len(scene_keys), "episodes_per_scene": args.episodes_per_scene,
                   "max_steps": args.max_steps, "min_steps": args.min_steps, "success_dist": args.success_dist,
                   "frame_stride": args.frame_stride, "ckpt": args.ckpt.split("/")[-3],
                   "mode": "smoke" if args.smoke else args.mode, "gen_text": args.gen_text},
        "note": "single-object data -> IN-DISTRIBUTION rollout (start->object), not yet cross-object stitching",
    }
    if args.smoke:
        result["smoke"] = smoke_rows
        result["follower_summary"] = summarize([r["follower"] for r in smoke_rows])
    else:
        if policy_rollouts:
            result["policy_summary"] = summarize(policy_rollouts); result["policy_rollouts"] = policy_rollouts
        if follower_rollouts:
            result["follower_summary"] = summarize(follower_rollouts); result["follower_rollouts"] = follower_rollouts
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print("\n=== C-2 ROLLOUT SUMMARY ===")
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("rollouts") and k != "smoke"}, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
