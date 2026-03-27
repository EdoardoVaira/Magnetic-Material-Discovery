from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pymatgen")

from pymatgen.core import Lattice, Structure

from dataset import CrystalGraphBuilder, GraphConfig


def test_periodic_graph_builds_expected_neighbors() -> None:
    structure = Structure(Lattice.cubic(4.0), ["Li"], [[0.0, 0.0, 0.0]])
    builder = CrystalGraphBuilder(GraphConfig(cutoff=5.0, num_radial=8, radial_width=0.5))

    graph = builder.build(structure, target=0.0, material_id="mp-test")

    assert graph.num_nodes == 1
    assert graph.edge_index.shape == (2, 6)
    assert graph.edge_attr.shape == (6, 8)
    assert torch.allclose(graph.edge_distance, torch.full((6, 1), 4.0))
