"""Recurrent navigation policy for the nav_baseline navigator.

The policy is causal by construction: it is a GRU that consumes one observation at a time
and carries a hidden state forward, so a prediction at step t can only depend on steps
0..t. This removes the future-information leak that a bidirectional sequence encoder has
when it is trained offline but deployed step by step.

One step of input is the concatenation of
  - a depth embedding produced by a small convolutional encoder,
  - the goal-relative vector (distance, sin bearing, cos bearing) from ``env.goal_vector``,
  - a one-hot encoding of the previous action.

The visual encoder is deliberately a plug-in point: swapping the depth CNN for a frozen
representation (for example the PR2L VLM embedding) keeps the rest of the recipe fixed,
which is what turns this navigator into a controlled comparison between representations.
"""
from __future__ import annotations

import torch
from torch import nn

from nav_baseline.env import DEPTH_SIZE, NUM_ACTIONS

GOAL_DIM = 3


class DepthEncoder(nn.Module):
    """Small convolutional encoder mapping a DEPTH_SIZE x DEPTH_SIZE depth map to a vector."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2), nn.ReLU(),   # 64 -> 32
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 32 -> 16
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),  # 16 -> 8
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),  # 8 -> 4
            nn.Flatten(),
            nn.Linear(64 * (DEPTH_SIZE // 16) ** 2, out_dim), nn.ReLU(),
        )

    def forward(self, depth):
        return self.net(depth.unsqueeze(-3))  # add channel dim: [..., 1, H, W]


class NavPolicy(nn.Module):
    """GRU policy predicting one of STOP / FORWARD / LEFT / RIGHT per step."""

    def __init__(self, visual_dim: int = 128, hidden_dim: int = 256, num_layers: int = 1):
        super().__init__()
        self.visual = DepthEncoder(visual_dim)
        self.input_dim = visual_dim + GOAL_DIM + NUM_ACTIONS
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(self.input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, NUM_ACTIONS))

    def step_features(self, depth, goal, prev_action):
        """Build the per-step input vector for a batch of single timesteps."""
        visual = self.visual(depth)
        prev_onehot = torch.nn.functional.one_hot(prev_action, NUM_ACTIONS).float()
        return torch.cat([visual, goal, prev_onehot], dim=-1)

    def forward(self, depth, goal, prev_action, hidden=None):
        """Run the policy over a sequence.

        Shapes: depth [B, T, H, W], goal [B, T, 3], prev_action [B, T] (long).
        Returns action logits [B, T, NUM_ACTIONS] and the final hidden state.
        """
        batch, steps = depth.shape[0], depth.shape[1]
        flat_visual = self.visual(depth.reshape(batch * steps, *depth.shape[2:]))
        visual = flat_visual.reshape(batch, steps, -1)
        prev_onehot = torch.nn.functional.one_hot(prev_action, NUM_ACTIONS).float()
        sequence = torch.cat([visual, goal, prev_onehot], dim=-1)
        output, hidden = self.gru(sequence, hidden)
        return self.head(output), hidden

    @torch.inference_mode()
    def act(self, depth, goal, prev_action, hidden=None, allow_stop: bool = True):
        """Single-step inference used during closed-loop rollout.

        Shapes: depth [1, H, W], goal [1, 3], prev_action [1]. Returns (action, hidden).
        """
        logits, hidden = self.forward(depth.unsqueeze(1), goal.unsqueeze(1),
                                      prev_action.unsqueeze(1), hidden)
        logits = logits[:, -1]
        if not allow_stop:
            logits = logits.clone()
            logits[:, 0] = float("-inf")
        return int(logits.argmax(dim=-1).item()), hidden
