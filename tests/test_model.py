from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Batch, Data

from model import MagNet, ModelConfig


def test_magnet_returns_all_heads() -> None:
    graph_a = Data(
        z=torch.tensor([8, 8], dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.randn(2, 8),
        edge_distance=torch.rand(2),
        edge_vector=torch.randn(2, 3),
        node_scalar_features=torch.randn(2, 13),
        node_geometry_features=torch.randn(2, 10),
        batch=torch.tensor([0, 0], dtype=torch.long),
        num_nodes=2,
    )
    graph_b = Data(
        z=torch.tensor([14, 14, 14], dtype=torch.long),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        edge_attr=torch.randn(4, 8),
        edge_distance=torch.rand(4),
        edge_vector=torch.randn(4, 3),
        node_scalar_features=torch.randn(3, 13),
        node_geometry_features=torch.randn(3, 10),
        batch=torch.tensor([0, 0, 0], dtype=torch.long),
        num_nodes=3,
    )
    batch = Batch.from_data_list([graph_a, graph_b])

    model = MagNet(
        edge_dim=8,
        num_classes=4,
        config=ModelConfig(
            hidden_dim=16,
            vector_dim=8,
            num_layers=2,
            dropout=0.0,
            graph_pooling="mean_max",
            ordering_pooling="mean_max_attention",
        ),
    )
    outputs = model(batch)

    assert set(outputs) == {
        "energy",
        "formation_energy",
        "band_gap",
        "transition_temperature",
        "magnetization",
        "magnetization_from_sites",
        "magnetic_logits",
        "site_moments",
        "ordering_logits",
    }
    assert outputs["energy"].shape == (2,)
    assert outputs["formation_energy"].shape == (2,)
    assert outputs["band_gap"].shape == (2,)
    assert outputs["transition_temperature"].shape == (2,)
    assert outputs["magnetization"].shape == (2,)
    assert outputs["magnetization_from_sites"].shape == (2,)
    assert outputs["magnetic_logits"].shape == (2,)
    assert outputs["site_moments"].shape == (5,)
    assert outputs["ordering_logits"].shape == (2, 4)
