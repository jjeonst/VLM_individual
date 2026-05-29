"""Graph-conditioned transformer behavior-cloning policy."""

from __future__ import annotations

import torch

from configs.schema import PolicyConfig


class GraphTransformerPolicy(torch.nn.Module):
    """Predict a discrete action from cached topological graph nodes."""

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        self.input_proj = torch.nn.Linear(config.input_dim, config.hidden_dim)
        self.goal_token = torch.nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        layer = torch.nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=config.transformer_layers)
        self.norm = torch.nn.LayerNorm(config.hidden_dim)
        self.action_head = torch.nn.Linear(config.hidden_dim, config.num_actions)
        torch.nn.init.normal_(self.goal_token, std=0.02)

    def forward(self, graph_nodes, graph_mask):
        if graph_nodes.ndim != 3:
            raise ValueError("graph_nodes must have shape [batch, nodes, dim]")
        if graph_mask.ndim != 2:
            raise ValueError("graph_mask must have shape [batch, nodes]")
        node_features = self.input_proj(graph_nodes)
        batch_size = graph_nodes.shape[0]
        goal = self.goal_token.expand(batch_size, -1, -1)
        sequence = torch.cat([goal, node_features], dim=1)
        goal_mask = torch.ones(batch_size, 1, dtype=graph_mask.dtype, device=graph_mask.device)
        full_mask = torch.cat([goal_mask, graph_mask], dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=~full_mask.bool())
        goal_features = self.norm(encoded[:, 0])
        return self.action_head(goal_features)
