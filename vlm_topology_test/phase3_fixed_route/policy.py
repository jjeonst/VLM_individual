"""The policy used for the fixed-route check, shared by both representations.

One policy definition serves both the oracle representation and the vision-language model
representation so that any difference between them comes from the representation and not
from the network around it. A frame arrives as a **set of tokens**: the oracle contributes a
single token holding direction and distance, while the vision-language model contributes
several tokens covering different parts of the input. The policy therefore has two stages.

1. **Within a frame**, a learned query attends over that frame's tokens and summarises them
   into one vector. With a single token this reduces to a linear map, so the oracle is not
   handicapped; with many tokens it lets the policy attend to whichever part of the view
   matters, which is the reason for keeping several tokens in the first place.
2. **Across time**, a Transformer with causal masking reads the summaries of the frames seen
   so far and predicts the next action. The mask is what keeps training honest: the policy is
   deployed one step at a time, so at training time a step must not see later steps.

An option is provided to drop the positional information across time. With a fixed route the
policy could otherwise succeed by memorising the action sequence and replaying it by step
number, ignoring what it sees. Removing positions is one way to make that impossible; the
observation ablation in ``evaluate.py`` is the other, and the two are complementary.
"""
from __future__ import annotations

import torch
from torch import nn

NUM_ACTIONS = 4


class FixedRoutePolicy(nn.Module):
    """Token set per frame -> attention pooling -> causal Transformer -> action."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, layers: int = 2,
                 heads: int = 8, dropout: float = 0.1, max_positions: int = 512,
                 use_positions: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_positions = max_positions
        self.use_positions = use_positions

        self.token_proj = nn.Linear(input_dim, hidden_dim)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pool_attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout,
                                                    batch_first=True)
        self.position = nn.Embedding(max_positions, hidden_dim)
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=heads,
                                           dim_feedforward=hidden_dim * 4, dropout=dropout,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                  nn.Linear(hidden_dim, NUM_ACTIONS))
        nn.init.normal_(self.pool_query, std=0.02)

    def summarise_frames(self, tokens: torch.Tensor) -> torch.Tensor:
        """[B, T, K, D] -> [B, T, hidden]: one vector per frame."""
        batch, steps, count, _ = tokens.shape
        projected = self.token_proj(tokens).reshape(batch * steps, count, self.hidden_dim)
        query = self.pool_query.expand(batch * steps, 1, self.hidden_dim)
        pooled, _ = self.pool_attention(query, projected, projected, need_weights=False)
        return pooled.reshape(batch, steps, self.hidden_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """tokens [B, T, K, D], mask [B, T] -> action logits [B, T, NUM_ACTIONS]."""
        steps = tokens.shape[1]
        if steps > self.max_positions:
            raise ValueError(f"sequence of {steps} steps exceeds max_positions")
        hidden = self.summarise_frames(tokens)
        if self.use_positions:
            hidden = hidden + self.position(torch.arange(steps, device=tokens.device))[None]
        causal = torch.triu(torch.full((steps, steps), float("-inf"), device=tokens.device),
                            diagonal=1)
        encoded = self.encoder(hidden, mask=causal, src_key_padding_mask=~mask)
        return self.head(self.norm(encoded))
