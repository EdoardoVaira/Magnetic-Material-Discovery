"""Single multitask crystal model for magnetic materials screening."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import AttentionalAggregation, global_max_pool, global_mean_pool

from dataset import NODE_GEOMETRY_FEATURE_DIM, NODE_SCALAR_FEATURE_DIM


SITE_STAT_DIM = 8


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 192
    vector_dim: int = 64
    num_layers: int = 5
    dropout: float = 0.05
    max_atomic_number: int = 118
    graph_pooling: str = "mean_max"
    ordering_pooling: str | None = "mean_max_attention"


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    *,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class GraphReadout(nn.Module):
    def __init__(self, hidden_dim: int, pooling: str) -> None:
        super().__init__()
        self.pooling = pooling
        if pooling not in {"mean", "mean_max", "attention", "mean_max_attention"}:
            raise ValueError(f"Unsupported pooling mode: {pooling}")
        self.attention = None
        if "attention" in pooling:
            self.attention = AttentionalAggregation(
                gate_nn=nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1),
                )
            )

    @property
    def output_multiplier(self) -> int:
        if self.pooling == "mean":
            return 1
        if self.pooling == "mean_max":
            return 2
        if self.pooling == "attention":
            return 1
        return 3

    def forward(self, x: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.pooling in {"mean", "mean_max", "mean_max_attention"}:
            parts.append(global_mean_pool(x, batch_index))
        if self.pooling in {"mean_max", "mean_max_attention"}:
            parts.append(global_max_pool(x, batch_index))
        if self.pooling in {"attention", "mean_max_attention"}:
            assert self.attention is not None
            parts.append(self.attention(x, batch_index))
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=-1)


def _augmented_edge_features(batch) -> torch.Tensor:
    edge_distance = batch.edge_distance
    if edge_distance.dim() == 1:
        edge_distance = edge_distance.unsqueeze(-1)
    inv_distance = edge_distance.clamp(min=1e-6).reciprocal()
    return torch.cat([batch.edge_attr, edge_distance, inv_distance], dim=-1)


def _edge_vectors(batch) -> torch.Tensor:
    edge_vector = getattr(batch, "edge_vector", None)
    if edge_vector is None:
        num_edges = batch.edge_index.size(1)
        return torch.zeros((num_edges, 3), dtype=batch.edge_attr.dtype, device=batch.edge_attr.device)
    return edge_vector


def _vector_norm(vector_state: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(vector_state.pow(2).sum(dim=1).clamp(min=1e-8))


def _mean_aggregate(messages: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if messages.dim() < 2:
        raise ValueError("Messages must have at least two dimensions for aggregation.")
    out_shape = (dim_size,) + tuple(messages.shape[1:])
    aggregated = messages.new_zeros(out_shape)
    aggregated.index_add_(0, index, messages)

    counts = messages.new_zeros(dim_size)
    counts.index_add_(0, index, torch.ones_like(index, dtype=messages.dtype))
    broadcast_shape = (dim_size,) + (1,) * (messages.dim() - 1)
    return aggregated / counts.clamp(min=1.0).view(broadcast_shape)


def _site_moment_statistics(
    site_moments: torch.Tensor,
    batch_index: torch.Tensor,
    *,
    threshold: float = 0.1,
) -> torch.Tensor:
    site_values = site_moments.unsqueeze(-1)
    signed_mean = global_mean_pool(site_values, batch_index)
    net_abs = signed_mean.abs()
    abs_mean = global_mean_pool(site_values.abs(), batch_index)
    sq_mean = global_mean_pool(site_values.square(), batch_index)
    std = (sq_mean - signed_mean.square()).clamp(min=0.0).add(1e-8).sqrt()
    max_abs = global_max_pool(site_values.abs(), batch_index)
    pos_frac = global_mean_pool((site_values > threshold).to(torch.float32), batch_index)
    neg_frac = global_mean_pool((site_values < -threshold).to(torch.float32), batch_index)
    cancellation = (abs_mean - net_abs).clamp(min=0.0)
    return torch.cat(
        [
            signed_mean,
            net_abs,
            abs_mean,
            std,
            max_abs,
            pos_frac,
            neg_frac,
            cancellation,
        ],
        dim=-1,
    )


class EquivariantResidualLayer(nn.Module):
    """Lightweight scalar/vector interaction layer using edge directions."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        vector_dim: int,
        edge_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vector_dim = vector_dim
        self.scalar_norm = nn.LayerNorm(hidden_dim)
        self.post_norm = nn.LayerNorm(hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear((2 * hidden_dim) + edge_dim, 2 * hidden_dim + (2 * vector_dim)),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim + (2 * vector_dim), hidden_dim + hidden_dim + vector_dim + vector_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + vector_dim, 2 * hidden_dim + vector_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim + vector_dim, hidden_dim + vector_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        scalar_state: torch.Tensor,
        vector_state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = edge_index[0]
        source = edge_index[1]

        scalar_target = self.scalar_norm(scalar_state[target])
        scalar_source = self.scalar_norm(scalar_state[source])
        edge_input = torch.cat([scalar_target, scalar_source, edge_attr], dim=-1)
        edge_update = self.edge_mlp(edge_input)
        scalar_message, scalar_gate, vector_gate, direction_scale = torch.split(
            edge_update,
            [self.hidden_dim, self.hidden_dim, self.vector_dim, self.vector_dim],
            dim=-1,
        )

        scalar_message = scalar_message * torch.sigmoid(scalar_gate)
        unit_vector = F.normalize(edge_vector, dim=-1, eps=1e-8)
        source_vector = vector_state[source] * torch.sigmoid(vector_gate).unsqueeze(1)
        directional_vector = unit_vector.unsqueeze(-1) * direction_scale.unsqueeze(1)
        vector_message = source_vector + directional_vector

        scalar_update = _mean_aggregate(scalar_message, target, scalar_state.size(0))
        vector_update = _mean_aggregate(vector_message, target, scalar_state.size(0))

        scalar_state = scalar_state + self.dropout(scalar_update)
        vector_state = vector_state + self.dropout(vector_update)

        node_update = self.node_mlp(torch.cat([self.post_norm(scalar_state), _vector_norm(vector_state)], dim=-1))
        scalar_delta, vector_scale = torch.split(
            node_update,
            [self.hidden_dim, self.vector_dim],
            dim=-1,
        )
        scalar_state = scalar_state + self.dropout(scalar_delta)
        vector_state = vector_state * (1.0 + 0.1 * torch.tanh(vector_scale).unsqueeze(1))
        return scalar_state, vector_state


class MagneticBackbone(nn.Module):
    def __init__(self, edge_dim: int, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.embedding = nn.Embedding(self.config.max_atomic_number + 1, self.config.hidden_dim)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(NODE_SCALAR_FEATURE_DIM + NODE_GEOMETRY_FEATURE_DIM, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.input_norm = nn.LayerNorm(self.config.hidden_dim)
        self.edge_dim = edge_dim + 2
        self.layers = nn.ModuleList(
            [
                EquivariantResidualLayer(
                    hidden_dim=self.config.hidden_dim,
                    vector_dim=self.config.vector_dim,
                    edge_dim=self.edge_dim,
                    dropout=self.config.dropout,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.output_proj = nn.Sequential(
            nn.Linear(self.config.hidden_dim + self.config.vector_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )

    def encode_nodes(self, batch) -> torch.Tensor:
        local_features = torch.cat([batch.node_scalar_features, batch.node_geometry_features], dim=-1)
        scalar_state = self.embedding(batch.z) + self.scalar_encoder(local_features)
        scalar_state = self.input_norm(scalar_state)
        vector_state = scalar_state.new_zeros((scalar_state.size(0), 3, self.config.vector_dim))

        edge_attr = _augmented_edge_features(batch)
        edge_vector = _edge_vectors(batch)
        for layer in self.layers:
            scalar_state, vector_state = layer(
                scalar_state=scalar_state,
                vector_state=vector_state,
                edge_index=batch.edge_index,
                edge_attr=edge_attr,
                edge_vector=edge_vector,
            )

        return self.output_proj(torch.cat([scalar_state, _vector_norm(vector_state)], dim=-1))


class MagneticModel(nn.Module):
    def __init__(
        self,
        edge_dim: int,
        num_classes: int,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.backbone = MagneticBackbone(edge_dim=edge_dim, config=config)
        self.config = self.backbone.config

        ordering_pooling = self.config.ordering_pooling or self.config.graph_pooling
        self.structural_readout = GraphReadout(self.config.hidden_dim, self.config.graph_pooling)
        self.magnetic_readout = GraphReadout(self.config.hidden_dim, ordering_pooling)
        self.ordering_readout = GraphReadout(self.config.hidden_dim, ordering_pooling)

        self.magnetic_node_proj = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.ordering_node_proj = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.site_moment_head = build_mlp(
            self.config.hidden_dim,
            self.config.hidden_dim,
            1,
            dropout=self.config.dropout,
        )

        structural_dim = self.structural_readout.output_multiplier * self.config.hidden_dim
        magnetic_dim = self.magnetic_readout.output_multiplier * self.config.hidden_dim
        ordering_dim = self.ordering_readout.output_multiplier * self.config.hidden_dim

        self.energy_head = build_mlp(structural_dim, self.config.hidden_dim, 1, dropout=self.config.dropout)
        self.formation_energy_head = build_mlp(
            structural_dim,
            self.config.hidden_dim,
            1,
            dropout=self.config.dropout,
        )
        self.band_gap_head = build_mlp(structural_dim, self.config.hidden_dim, 1, dropout=self.config.dropout)
        self.transition_temperature_head = build_mlp(
            structural_dim + magnetic_dim + ordering_dim + SITE_STAT_DIM,
            self.config.hidden_dim,
            1,
            dropout=self.config.dropout,
        )
        self.magnetization_residual_head = build_mlp(
            structural_dim + magnetic_dim + SITE_STAT_DIM,
            self.config.hidden_dim,
            1,
            dropout=self.config.dropout,
        )
        self.magnetic_head = build_mlp(
            magnetic_dim + SITE_STAT_DIM,
            self.config.hidden_dim,
            1,
            dropout=self.config.dropout,
        )
        self.ordering_head = build_mlp(
            ordering_dim + magnetic_dim + SITE_STAT_DIM + 1,
            self.config.hidden_dim,
            num_classes,
            dropout=self.config.dropout,
        )

    def forward(self, batch) -> dict[str, torch.Tensor]:
        node_features = self.backbone.encode_nodes(batch)
        magnetic_node_features = self.magnetic_node_proj(node_features)
        ordering_node_features = self.ordering_node_proj(node_features)

        site_moments = self.site_moment_head(magnetic_node_features).squeeze(-1)
        site_stats = _site_moment_statistics(site_moments, batch.batch)

        structural_pooled = self.structural_readout(node_features, batch.batch)
        magnetic_pooled = self.magnetic_readout(magnetic_node_features, batch.batch)
        ordering_pooled = self.ordering_readout(ordering_node_features, batch.batch)

        magnetization_from_sites = site_stats[:, 1]
        magnetic_logits = self.magnetic_head(torch.cat([magnetic_pooled, site_stats], dim=-1)).squeeze(-1)
        magnetization_residual = self.magnetization_residual_head(
            torch.cat([structural_pooled, magnetic_pooled, site_stats], dim=-1)
        ).squeeze(-1)
        magnetization = F.softplus(magnetization_from_sites + magnetization_residual)

        ordering_logits = self.ordering_head(
            torch.cat([ordering_pooled, magnetic_pooled, site_stats, magnetic_logits.unsqueeze(-1)], dim=-1)
        )

        transition_temperature = F.softplus(
            self.transition_temperature_head(
                torch.cat([structural_pooled, magnetic_pooled, ordering_pooled, site_stats], dim=-1)
            ).squeeze(-1)
        )

        return {
            "energy": F.softplus(self.energy_head(structural_pooled).squeeze(-1)),
            "formation_energy": self.formation_energy_head(structural_pooled).squeeze(-1),
            "band_gap": F.softplus(self.band_gap_head(structural_pooled).squeeze(-1)),
            "transition_temperature": transition_temperature,
            "magnetization": magnetization,
            "magnetization_from_sites": magnetization_from_sites,
            "magnetic_logits": magnetic_logits,
            "site_moments": site_moments,
            "ordering_logits": ordering_logits,
        }


MagNet = MagneticModel
