from __future__ import annotations

import pytest

pytest.importorskip("pymatgen")
pytest.importorskip("plotly")

from pymatgen.core import Lattice, Structure

from screen import create_crystal_figure


def test_create_crystal_figure_includes_atoms_and_unit_cell() -> None:
    structure = Structure(Lattice.cubic(4.0), ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    figure = create_crystal_figure(
        atomic_numbers=structure.atomic_numbers,
        positions=structure.cart_coords,
        lattice=structure.lattice.matrix,
        title="LiO",
    )

    assert len(figure.data) >= 2
    assert figure.layout.title.text == "LiO"
    marker_traces = [trace for trace in figure.data if getattr(trace, "mode", None) == "markers"]
    cartesian_axis_traces = [
        trace for trace in figure.data if str(getattr(trace, "name", "") or "").endswith("-axis")
    ]

    assert sum(len(trace.x) for trace in marker_traces) >= 9
    assert len(cartesian_axis_traces) == 3
    annotation_text = figure.layout.annotations[0].text
    assert "space group" in annotation_text
    assert "cell" in annotation_text
