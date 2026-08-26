"""Train the fixed-route policy on either representation.

The same training procedure is applied to both representations so that any difference in the
result comes from the representation itself.

- ``oracle``: one token per frame stating how far the goal is and in which direction it lies.
  A policy that cannot learn from this is not limited by perception, so this arm tells us
  whether the training and deployment procedure works at all.
- ``vlm``: the token set produced by ``encode.py``, covering the image, the question and the
  model's answer.

Training is behavior cloning: the expert's action at each visited state is the label. Because
demonstrations are dominated by moving forward, the loss weights the moments where the action
changes and the moments where the action is a stop or a turn, which are the decisions that
actually matter. Whole demonstrations form the batches, and the policy reads them with causal
masking so that a step never sees the future.

``--no-positions`` removes the step-index information from the policy. On a fixed route a
policy can otherwise reproduce the demonstration by counting steps rather than by looking, so
comparing a run with and without this option shows how much of the performance survives when
counting is impossible.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vlm_topology_test.phase3_fixed_route.collect import DATA_ROOT  # noqa: E402
from vlm_topology_test.phase3_fixed_route.encode import PCA_FILE  # noqa: E402
from vlm_topology_test.phase3_fixed_route.policy import FixedRoutePolicy, NUM_ACTIONS  # noqa: E402

CKPT_ROOT = DATA_ROOT / "policies"
STOP_TURN_ACTIONS = (0, 2, 3)
INFLECTION_WEIGHT = 2.0
STOP_TURN_WEIGHT = 1.5


class DemonstrationSet(Dataset):
    """Whole demonstrations, each a sequence of token sets with expert labels."""

    def __init__(self, directory: Path, representation: str):
        self.items = []
        for shard in sorted(directory.glob("*.npz")):
            if shard.name == PCA_FILE:      # the principal directions, not a demonstration
                continue
            payload = np.load(shard)
            for key in sorted({name.split("|")[0] for name in payload.files}):
                labels = payload[f"{key}|label"]
                if len(labels) < 2:
                    continue
                if representation == "oracle":
                    tokens = payload[f"{key}|oracle"][:, None, :]   # one token per frame
                else:
                    tokens = payload[f"{key}|tokens"]
                self.items.append({"tokens": tokens, "label": labels, "scene": key})
        if not self.items:
            raise ValueError(f"no demonstrations in {directory}")
        self.token_count = int(self.items[0]["tokens"].shape[1])
        self.width = int(self.items[0]["tokens"].shape[2])

    def action_counts(self) -> np.ndarray:
        counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
        for item in self.items:
            counts += np.bincount(item["label"], minlength=NUM_ACTIONS)
        return counts

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        return item["tokens"].astype(np.float32), item["label"].astype(np.int64)


def collate(batch):
    length = max(len(labels) for _, labels in batch)
    count, width = batch[0][0].shape[1], batch[0][0].shape[2]
    tokens = np.zeros((len(batch), length, count, width), dtype=np.float32)
    labels = np.zeros((len(batch), length), dtype=np.int64)
    mask = np.zeros((len(batch), length), dtype=bool)
    for index, (block, label) in enumerate(batch):
        tokens[index, : len(label)] = block
        labels[index, : len(label)] = label
        mask[index, : len(label)] = True
    return torch.from_numpy(tokens), torch.from_numpy(labels), torch.from_numpy(mask)


def normalisation(dataset: DemonstrationSet, limit: int = 20000) -> dict:
    """Per-feature mean and scale, so both representations reach the policy on one scale."""
    rows = []
    for item in dataset.items:
        rows.append(item["tokens"].reshape(-1, dataset.width).astype(np.float32))
        if sum(len(r) for r in rows) >= limit:
            break
    stack = np.concatenate(rows, axis=0)[:limit]
    mean = stack.mean(axis=0)
    scale = stack.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32)}


def step_weights(labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = torch.ones_like(labels, dtype=torch.float32)
    stop_or_turn = torch.zeros_like(mask)
    for action in STOP_TURN_ACTIONS:
        stop_or_turn |= labels == action
    weights = torch.where(stop_or_turn, weights * STOP_TURN_WEIGHT, weights)
    if labels.shape[1] > 1:
        changed = labels[:, 1:] != labels[:, :-1]
        weights[:, 1:] = torch.where(changed, weights[:, 1:] * INFLECTION_WEIGHT,
                                     weights[:, 1:])
    return weights * mask.float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", required=True, choices=["random", "matched", "junction"])
    parser.add_argument("--representation", required=True, choices=["oracle", "vlm"])
    parser.add_argument("--prompt", default="baseline",
                        help="which encoding to read when the representation is 'vlm'")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="matches the rate the prior work settled on for this task")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--no-positions", action="store_true",
                        help="remove step-index information so the route cannot be replayed "
                             "by counting steps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    directory = (DATA_ROOT / f"demos_{args.rule}" if args.representation == "oracle"
                 else DATA_ROOT / f"encoded_{args.rule}__{args.prompt}")
    dataset = DemonstrationSet(directory, args.representation)
    counts = dataset.action_counts()
    print(f"[train] {args.rule}/{args.representation}: {len(dataset)} demonstrations, "
          f"{int(counts.sum())} steps, {dataset.token_count} tokens x {dataset.width} wide",
          flush=True)
    print(f"[train] actions STOP/FORWARD/LEFT/RIGHT = {counts.tolist()} "
          f"(always-forward baseline {counts[1] / counts.sum():.3f})", flush=True)

    stats = normalisation(dataset)
    mean = torch.as_tensor(stats["mean"], device=device)
    scale = torch.as_tensor(stats["scale"], device=device)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
                        collate_fn=collate)
    model_args = {"input_dim": dataset.width, "hidden_dim": args.hidden_dim,
                  "layers": args.layers, "use_positions": not args.no_positions}
    policy = FixedRoutePolicy(**model_args).to(device)
    optimiser = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction="none")

    run = args.run_name or (f"{args.rule}_{args.representation}"
                            + (f"_{args.prompt}" if args.representation == "vlm" else "")
                            + ("_nopos" if args.no_positions else ""))
    out_dir = CKPT_ROOT / run
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        policy.train()
        total_loss, correct, total = 0.0, 0, 0
        for tokens, labels, mask in loader:
            tokens = ((tokens.to(device) - mean) / scale)
            labels, mask = labels.to(device), mask.to(device)
            logits = policy(tokens, mask)
            weights = step_weights(labels, mask)
            losses = criterion(logits.reshape(-1, NUM_ACTIONS), labels.reshape(-1))
            loss = (losses * weights.reshape(-1)).sum() / weights.sum().clamp(min=1.0)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimiser.step()
            valid = int(mask.sum().item())
            total_loss += float(loss.item()) * valid
            correct += int(((logits.argmax(-1) == labels) & mask).sum().item())
            total += valid
        record = {"epoch": epoch, "loss": round(total_loss / max(total, 1), 4),
                  "action_accuracy": round(correct / max(total, 1), 4)}
        history.append(record)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[train] epoch {epoch}/{args.epochs} loss {record['loss']} "
                  f"causal action accuracy {record['action_accuracy']}", flush=True)
        torch.save({"model": policy.state_dict(), "model_args": model_args,
                    "normalisation": stats, "representation": args.representation,
                    "rule": args.rule, "prompt": args.prompt, "epoch": epoch,
                    "max_train_length": max(len(i["label"]) for i in dataset.items),
                    "history": history}, out_dir / "model.pt")

    (out_dir / "history.json").write_text(json.dumps(
        {"run": run, "rule": args.rule, "representation": args.representation,
         "prompt": args.prompt, "use_positions": not args.no_positions,
         "demonstrations": len(dataset), "steps": int(counts.sum()),
         "action_counts": counts.tolist(), "history": history}, indent=2) + "\n")
    print(f"\n[train] wrote {out_dir / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
