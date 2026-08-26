"""Train the navigation policy on vision-language model representations (step 3).

The policy follows the recipe described for PR2L's Habitat experiments (see the reference
list in ``experiment_design_2x2.md``): the representation produced by the frozen
vision-language model is compressed with a fitted projection, a learned Transformer
processes the compressed sequence, and an action head predicts one of the four navigation
actions. Training is behavior cloning with the paper's two reweightings, and batches are
formed from whole trajectories with gradient accumulation.

    representation (4096) -> PCA projection (1024) -> linear (hidden)
        -> Transformer with causal masking -> action logits (4)

**One deliberate departure from the earlier implementation of this recipe.** The policy is
deployed step by step, so at training time each timestep must only see timesteps before it.
The earlier implementation used an unmasked Transformer, which let every timestep read the
rest of the trajectory; it scored 0.994 on held-out actions when given the whole trajectory
but 0.776 when restricted to the past, and it could not navigate at all in closed loop.
This script therefore applies a causal mask, which makes the training condition identical
to the deployment condition.

The projection is fitted on the training representations of this run and stored in the
checkpoint, so that evaluation compresses observations exactly the same way.

Conditions are expressed as data mixes. ``--data a`` uses all of collection ``a``;
``--data a:0.5 b`` uses half of ``a`` and all of ``b``, which is how the experiment keeps
the total number of episodes equal across conditions.

Run:
  python -m vlm_topology_test.phase2_representation_redesign.train_policy \\
      --data exp_shortest_800__structure --run-name cond3_structure_shortest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ENCODED_ROOT = Path("/data/topovlm/nav_baseline/encoded")
CKPT_ROOT = Path("/data/topovlm/nav_baseline/policies")
NUM_ACTIONS = 4
STOP_TURN_ACTIONS = (0, 2, 3)   # STOP, TURN_LEFT, TURN_RIGHT
INFLECTION_WEIGHT = 2.0
STOP_TURN_WEIGHT = 1.5


# --------------------------------------------------------------------------- data
def parse_mix(specs: list[str]) -> list[tuple[str, float]]:
    """Turn ``["a:0.5", "b"]`` into ``[("a", 0.5), ("b", 1.0)]``."""
    mix = []
    for spec in specs:
        name, _, fraction = spec.partition(":")
        mix.append((name, float(fraction) if fraction else 1.0))
    return mix


class TrajectoryDataset(Dataset):
    """Whole encoded trajectories, so that batches are trajectory-aware as in the paper."""

    def __init__(self, mix: list[tuple[str, float]], seed: int = 0):
        self.episodes = []
        rng = np.random.default_rng(seed)
        for name, fraction in mix:
            directory = ENCODED_ROOT / name
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            collected = []
            for shard in sorted(directory.glob("*.npz")):
                payload = np.load(shard)
                for key in sorted({field.split("|")[0] for field in payload.files}):
                    labels = payload[f"{key}|label"]
                    if len(labels) < 2:
                        continue
                    collected.append({"repr": payload[f"{key}|repr"], "label": labels,
                                      "source": name})
            if fraction < 1.0:
                keep = max(1, int(round(len(collected) * fraction)))
                chosen = rng.choice(len(collected), size=keep, replace=False)
                collected = [collected[i] for i in sorted(chosen)]
            print(f"[data] {name}: {len(collected)} episodes (fraction {fraction})", flush=True)
            self.episodes.extend(collected)
        if not self.episodes:
            raise ValueError("no episodes in the requested mix")
        self.dim = int(self.episodes[0]["repr"].shape[-1])

    def action_counts(self) -> np.ndarray:
        counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
        for episode in self.episodes:
            counts += np.bincount(episode["label"], minlength=NUM_ACTIONS)
        return counts

    def sample_representations(self, limit: int, seed: int = 0) -> np.ndarray:
        """A random subset of frames, used to fit the compression projection."""
        rng = np.random.default_rng(seed)
        rows = []
        budget = limit
        order = rng.permutation(len(self.episodes))
        for index in order:
            block = self.episodes[index]["repr"].astype(np.float32)
            if len(block) > budget:
                block = block[rng.choice(len(block), size=budget, replace=False)]
            rows.append(block)
            budget -= len(block)
            if budget <= 0:
                break
        return np.concatenate(rows, axis=0)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, item):
        episode = self.episodes[item]
        return episode["repr"].astype(np.float32), episode["label"].astype(np.int64)


def collate(batch):
    """Pad trajectories in a batch to the same length and return a validity mask."""
    length = max(len(labels) for _, labels in batch)
    dim = batch[0][0].shape[-1]
    features = np.zeros((len(batch), length, dim), dtype=np.float32)
    labels = np.zeros((len(batch), length), dtype=np.int64)
    mask = np.zeros((len(batch), length), dtype=bool)
    for index, (feature, label) in enumerate(batch):
        features[index, : len(label)] = feature
        labels[index, : len(label)] = label
        mask[index, : len(label)] = True
    return (torch.from_numpy(features), torch.from_numpy(labels), torch.from_numpy(mask))


# --------------------------------------------------------------------------- projection
def fit_projection(samples: np.ndarray, dim: int) -> dict:
    """Principal-component projection of the representation, as PR2L compresses its tokens."""
    samples = samples.astype(np.float32, copy=False)
    mean = samples.mean(axis=0)
    centred = samples - mean
    _, singular, right = np.linalg.svd(centred, full_matrices=False)
    components = right[:dim].astype(np.float32)
    # Fix each component's sign so that repeated fits on the same data agree.
    peak = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(components.shape[0]), peak])
    signs[signs == 0] = 1.0
    components *= signs[:, None]
    explained = float((singular[:dim] ** 2).sum() / max((singular ** 2).sum(), 1e-9))
    return {"mean": mean.astype(np.float32), "components": components,
            "explained_variance_ratio": explained}


def apply_projection(features: torch.Tensor, projection: dict, device) -> torch.Tensor:
    mean = torch.as_tensor(projection["mean"], device=device)
    components = torch.as_tensor(projection["components"], device=device)
    return (features - mean) @ components.T


# --------------------------------------------------------------------------- policy
class PR2LPolicy(nn.Module):
    """Compressed representation -> causal Transformer -> action logits."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, layers: int = 2,
                 heads: int = 8, dropout: float = 0.1, max_positions: int = 512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.position = nn.Embedding(max_positions, hidden_dim)
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=heads,
                                           dim_feedforward=hidden_dim * 4, dropout=dropout,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, NUM_ACTIONS)
        self.max_positions = max_positions

    def forward(self, features, mask):
        steps = features.shape[1]
        if steps > self.max_positions:
            raise ValueError(f"trajectory of {steps} steps exceeds max_positions")
        hidden = self.input_proj(features)
        positions = torch.arange(steps, device=features.device)
        hidden = hidden + self.position(positions)[None]
        causal = torch.triu(torch.full((steps, steps), float("-inf"),
                                       device=features.device), diagonal=1)
        encoded = self.encoder(hidden, mask=causal, src_key_padding_mask=~mask)
        return self.head(self.norm(encoded))


def step_weights(labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Upweight decision points and non-forward actions, as the paper's training does."""
    weights = torch.ones_like(labels, dtype=torch.float32)
    stop_or_turn = torch.zeros_like(mask)
    for action in STOP_TURN_ACTIONS:
        stop_or_turn |= labels == action
    weights = torch.where(stop_or_turn, weights * STOP_TURN_WEIGHT, weights)
    if labels.shape[1] > 1:
        inflection = labels[:, 1:] != labels[:, :-1]
        weights[:, 1:] = torch.where(inflection, weights[:, 1:] * INFLECTION_WEIGHT,
                                     weights[:, 1:])
    return weights * mask.float()


# --------------------------------------------------------------------------- training
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True,
                        help="encoded directories, optionally as name:fraction")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulate", type=int, default=8,
                        help="gradient accumulation steps, as in the paper's training")
    parser.add_argument("--projection-dim", type=int, default=1024)
    parser.add_argument("--projection-fit-frames", type=int, default=20000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TrajectoryDataset(parse_mix(args.data), seed=args.seed)
    counts = dataset.action_counts()
    max_train_length = max(len(e["label"]) for e in dataset.episodes)
    print(f"[train] longest training trajectory: {max_train_length} steps", flush=True)
    print(f"[train] {len(dataset)} trajectories, {int(counts.sum())} steps, "
          f"actions STOP/FORWARD/LEFT/RIGHT = {counts.tolist()}", flush=True)

    projection_dim = min(args.projection_dim, dataset.dim)
    samples = dataset.sample_representations(args.projection_fit_frames, seed=args.seed)
    projection = fit_projection(samples, projection_dim)
    print(f"[train] projection {dataset.dim} -> {projection_dim} fitted on {len(samples)} "
          f"frames, explains {projection['explained_variance_ratio']:.3f} of variance",
          flush=True)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
                        collate_fn=collate)
    policy = PR2LPolicy(projection_dim, hidden_dim=args.hidden_dim,
                        layers=args.layers).to(device)
    optimiser = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction="none")

    out_dir = CKPT_ROOT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        policy.train()
        total_loss, correct, total = 0.0, 0, 0
        optimiser.zero_grad(set_to_none=True)
        for step, (features, labels, mask) in enumerate(loader, start=1):
            features = apply_projection(features.to(device), projection, device)
            labels, mask = labels.to(device), mask.to(device)
            logits = policy(features, mask)
            weights = step_weights(labels, mask)
            losses = criterion(logits.reshape(-1, NUM_ACTIONS), labels.reshape(-1))
            loss = (losses * weights.reshape(-1)).sum() / weights.sum().clamp(min=1.0)
            (loss / args.accumulate).backward()
            if step % args.accumulate == 0:
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
            valid = int(mask.sum().item())
            total_loss += float(loss.item()) * valid
            correct += int(((logits.argmax(-1) == labels) & mask).sum().item())
            total += valid
        if len(loader) % args.accumulate != 0:
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
        mean_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        history.append({"epoch": epoch, "loss": round(mean_loss, 4),
                        "action_accuracy": round(accuracy, 4)})
        print(f"[train] epoch {epoch}/{args.epochs} loss {mean_loss:.4f} "
              f"causal action accuracy {accuracy:.4f}", flush=True)
        torch.save({"model": policy.state_dict(),
                    "model_args": {"input_dim": projection_dim, "hidden_dim": args.hidden_dim,
                                   "layers": args.layers},
                    "projection": projection, "data": args.data, "epoch": epoch,
                    # Position embeddings are only trained up to the longest training
                    # trajectory; evaluation uses this to keep its context inside that range.
                    "max_train_length": max_train_length,
                    "history": history}, out_dir / "model.pt")

    (out_dir / "history.json").write_text(json.dumps(
        {"run_name": args.run_name, "data": args.data,
         "trajectories": len(dataset), "steps": int(counts.sum()),
         "action_counts": counts.tolist(),
         "projection_dim": projection_dim,
         "max_train_length": max_train_length,
         "explained_variance_ratio": projection["explained_variance_ratio"],
         "history": history}, indent=2) + "\n")
    print(f"\n[train] wrote {out_dir / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
