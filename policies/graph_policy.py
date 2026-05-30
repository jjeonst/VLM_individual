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
        self.position_embed = torch.nn.Embedding(config.max_positions, config.hidden_dim)
        torch.nn.init.normal_(self.goal_token, std=0.02)

    def forward(self, graph_nodes, graph_mask):
        if graph_nodes.ndim not in {3, 4}:
            raise ValueError(
                "graph_nodes must have shape [batch, nodes, dim] or "
                "[batch, nodes, tokens, dim]"
            )
        if graph_mask.ndim != 2:
            raise ValueError("graph_mask must have shape [batch, nodes]")
        if graph_nodes.ndim == 4:
            return self._forward_token_nodes(graph_nodes, graph_mask)
        node_features = self._add_positions(self.input_proj(graph_nodes))
        if self.config.prediction_target == "nodes":
            encoded = self.encoder(node_features, src_key_padding_mask=~graph_mask.bool())
            return self.action_head(self.norm(encoded))
        if self.config.prediction_target != "graph":
            raise ValueError(f"Unsupported prediction target: {self.config.prediction_target}")
        batch_size = graph_nodes.shape[0]
        goal = self.goal_token.expand(batch_size, -1, -1)
        sequence = torch.cat([goal, node_features], dim=1)
        sequence = self._add_positions(sequence)
        goal_mask = torch.ones(batch_size, 1, dtype=graph_mask.dtype, device=graph_mask.device)
        full_mask = torch.cat([goal_mask, graph_mask], dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=~full_mask.bool())
        goal_features = self.norm(encoded[:, 0])
        return self.action_head(goal_features)

    def _forward_token_nodes(self, graph_nodes, graph_mask):
        if self.config.prediction_target != "nodes":
            raise ValueError("Token-node inputs require policy.prediction_target=nodes")
        batch_size, node_count, token_count, _ = graph_nodes.shape
        token_features = self.input_proj(graph_nodes).reshape(
            batch_size, node_count * token_count, self.config.hidden_dim
        )
        token_features = self._add_positions(token_features)
        token_mask = graph_mask[:, :, None].expand(batch_size, node_count, token_count).reshape(
            batch_size, node_count * token_count
        )
        encoded = self.encoder(token_features, src_key_padding_mask=~token_mask.bool())
        encoded = encoded.reshape(batch_size, node_count, token_count, self.config.hidden_dim)
        node_features = self.norm(encoded.mean(dim=2))
        return self.action_head(node_features)

    def _add_positions(self, sequence):
        sequence_len = sequence.shape[1]
        if sequence_len > self.config.max_positions:
            raise ValueError(
                f"Sequence length {sequence_len} exceeds "
                f"policy.max_positions {self.config.max_positions}"
            )
        positions = torch.arange(sequence_len, device=sequence.device)
        return sequence + self.position_embed(positions)[None, :, :]
