"""Behavior-cloning training for the nav_baseline navigator.

The policy is trained to reproduce the expert action at every visited state. Training data
are the shards written by ``nav_baseline.collect``; several shard directories can be passed
at once so that a DAgger round is trained on the union of all rounds collected so far.

Because the policy is recurrent, samples are drawn as fixed-length windows of consecutive
steps (truncated backpropagation through time) rather than as independent frames.

Expert navigation trajectories are dominated by MOVE_FORWARD, so the cross-entropy loss is
weighted by inverse action frequency; without this the policy collapses onto always going
straight, which is the failure mode observed in earlier experiments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from nav_baseline.collect import OUT_ROOT
from nav_baseline.env import NUM_ACTIONS
from nav_baseline.policy import NavPolicy

CKPT_ROOT = Path("/data/topovlm/nav_baseline/checkpoints")


class WindowDataset(Dataset):
    """Fixed-length windows of consecutive steps drawn from the collected episodes."""

    def __init__(self, shard_dirs: list[Path], window: int = 64):
        self.window = window
        self.episodes = []
        for shard_dir in shard_dirs:
            for shard in sorted(shard_dir.glob("*.npz")):
                payload = np.load(shard)
                keys = {name.split("|")[0] for name in payload.files}
                for key in sorted(keys):
                    depth = payload[f"{key}|depth"]
                    if len(depth) < 2:
                        continue
                    self.episodes.append({
                        "depth": depth,
                        "goal": payload[f"{key}|goal"],
                        "prev_action": payload[f"{key}|prev_action"],
                        "label": payload[f"{key}|label"],
                    })
        if not self.episodes:
            raise ValueError(f"No episodes found in {[str(d) for d in shard_dirs]}")
        self.index = [(i, start)
                      for i, ep in enumerate(self.episodes)
                      for start in range(0, len(ep["label"]), window)]

    def action_counts(self) -> np.ndarray:
        counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
        for episode in self.episodes:
            counts += np.bincount(episode["label"], minlength=NUM_ACTIONS)
        return counts

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        episode_index, start = self.index[item]
        episode = self.episodes[episode_index]
        stop = min(start + self.window, len(episode["label"]))
        length = stop - start
        depth = np.zeros((self.window, *episode["depth"].shape[1:]), dtype=np.float32)
        goal = np.zeros((self.window, episode["goal"].shape[1]), dtype=np.float32)
        prev_action = np.zeros(self.window, dtype=np.int64)
        label = np.zeros(self.window, dtype=np.int64)
        mask = np.zeros(self.window, dtype=bool)
        depth[:length] = episode["depth"][start:stop].astype(np.float32) / 255.0
        goal[:length] = episode["goal"][start:stop]
        prev_action[:length] = episode["prev_action"][start:stop]
        label[:length] = episode["label"][start:stop]
        mask[:length] = True
        return depth, goal, prev_action, label, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True,
                        help="shard directory names under the data root (multiple = DAgger union)")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--visual-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shard_dirs = [OUT_ROOT / name for name in args.data]
    dataset = WindowDataset(shard_dirs, window=args.window)
    counts = dataset.action_counts()
    print(f"[train] {len(dataset.episodes)} episodes, {int(counts.sum())} steps, "
          f"action counts STOP/FWD/LEFT/RIGHT = {counts.tolist()}", flush=True)

    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"[train] class weights = {np.round(weights, 3).tolist()}", flush=True)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                        drop_last=False)
    model_args = {"visual_dim": args.visual_dim, "hidden_dim": args.hidden_dim}
    policy = NavPolicy(**model_args).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    out_dir = CKPT_ROOT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        policy.train()
        total_loss, total_correct, total_steps = 0.0, 0, 0
        for depth, goal, prev_action, label, mask in loader:
            depth = depth.to(device); goal = goal.to(device)
            prev_action = prev_action.to(device); label = label.to(device)
            mask = mask.to(device)
            logits, _ = policy(depth, goal, prev_action)
            loss_per_step = criterion(logits.reshape(-1, NUM_ACTIONS), label.reshape(-1))
            loss = (loss_per_step * mask.reshape(-1).float()).sum() / mask.sum().clamp(min=1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            steps = int(mask.sum().item())
            total_loss += float(loss.item()) * steps
            total_correct += int(((logits.argmax(-1) == label) & mask).sum().item())
            total_steps += steps
        mean_loss = total_loss / max(total_steps, 1)
        accuracy = total_correct / max(total_steps, 1)
        history.append({"epoch": epoch, "loss": round(mean_loss, 4),
                        "action_accuracy": round(accuracy, 4)})
        print(f"[train] epoch {epoch}/{args.epochs} loss {mean_loss:.4f} "
              f"action_accuracy {accuracy:.4f}", flush=True)
        torch.save({"model": policy.state_dict(), "model_args": model_args,
                    "epoch": epoch, "history": history, "data": args.data},
                   out_dir / "model.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"\n[train] wrote {out_dir / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
