"""GNoME magnetic screening and crystal structure visualization."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import warnings
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from pymatgen.core import Composition, Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from torch_geometric.data import Batch
from tqdm.auto import tqdm

from dataset import (
    CrystalGraphBuilder,
    GraphConfig,
    MAGNETIC_ORDERING_CLASSES,
    ensure_parent_dir,
    resolve_device,
)
from model import MagNet, ModelConfig


# ---------------------------------------------------------------------------
# Crystal structure visualization
# ---------------------------------------------------------------------------

CPK_COLORS: dict[str, str] = {
    "H": "#FFFFFF", "He": "#D9FFFF",
    "Li": "#CC80FF", "Be": "#C2FF00", "B": "#FFB5B5", "C": "#909090",
    "N": "#3050F8", "O": "#FF0D0D", "F": "#90E050", "Ne": "#B3E3F5",
    "Na": "#AB5CF2", "Mg": "#8AFF00", "Al": "#BFA6A6", "Si": "#F0C8A0",
    "P": "#FF8000", "S": "#FFFF30", "Cl": "#1FF01F", "Ar": "#80D1E3",
    "K": "#8F40D4", "Ca": "#3DFF00", "Sc": "#E6E6E6", "Ti": "#BFC2C7",
    "V": "#A6A6AB", "Cr": "#8A99C7", "Mn": "#9C7AC7", "Fe": "#E06633",
    "Co": "#F090A0", "Ni": "#50D050", "Cu": "#C88033", "Zn": "#7D80B0",
    "Ga": "#C28F8F", "Ge": "#668F8F", "As": "#BD80E3", "Se": "#FFA100",
    "Br": "#A62929", "Kr": "#5CB8D1",
    "Rb": "#702EB0", "Sr": "#00FF00", "Y": "#94FFFF", "Zr": "#94E0E0",
    "Nb": "#73C2C9", "Mo": "#54B5B5", "Tc": "#3B9E9E", "Ru": "#248F8F",
    "Rh": "#0A7D8C", "Pd": "#006985", "Ag": "#C0C0C0", "Cd": "#FFD98F",
    "In": "#A67573", "Sn": "#668080", "Sb": "#9E63B5", "Te": "#D47A00",
    "I": "#940094", "Xe": "#429EB0",
    "Cs": "#57178F", "Ba": "#00C900", "La": "#70D4FF",
    "Ce": "#FFFFC7", "Pr": "#D9FFC7", "Nd": "#C7FFC7", "Pm": "#A3FFC7",
    "Sm": "#8FFFC7", "Eu": "#61FFC7", "Gd": "#45FFC7", "Tb": "#30FFC7",
    "Dy": "#1FFFC7", "Ho": "#00FF9C", "Er": "#00E675", "Tm": "#00D452",
    "Yb": "#00BF38", "Lu": "#00AB24",
    "Hf": "#4DC2FF", "Ta": "#4DA6FF", "W": "#2194D6", "Re": "#267DAB",
    "Os": "#266696", "Ir": "#175487", "Pt": "#D0D0E0", "Au": "#FFD123",
    "Hg": "#B8B8D0", "Tl": "#A6544D", "Pb": "#575961", "Bi": "#9E4FB5",
    "Ac": "#70ABFA", "Th": "#00BAFF", "Pa": "#00A1FF", "U": "#008FFF",
}

COVALENT_RADII: dict[str, float] = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11,
    "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76,
    "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39, "Fe": 1.32,
    "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20,
    "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16, "Rb": 2.20, "Sr": 1.95,
    "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54, "Ru": 1.46, "Rh": 1.42,
    "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39, "Sb": 1.39,
    "Te": 1.38, "I": 1.39, "Xe": 1.40, "Cs": 2.44, "Ba": 2.15, "La": 2.07,
    "Ce": 2.04, "Pr": 2.03, "Nd": 2.01, "Sm": 1.98, "Eu": 1.98, "Gd": 1.96,
    "Tb": 1.94, "Dy": 1.92, "Ho": 1.92, "Er": 1.89, "Tm": 1.90, "Yb": 1.87,
    "Lu": 1.87, "Hf": 1.75, "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44,
    "Ir": 1.41, "Pt": 1.36, "Au": 1.36, "Hg": 1.32, "Tl": 1.45, "Pb": 1.46,
    "Bi": 1.48, "Th": 2.06, "U": 1.96,
}

_FALLBACK_PALETTE = [
    "#0B6E4F", "#C84C09", "#355070", "#6D597A", "#D1495B",
    "#3A86FF", "#FFB703", "#2A9D8F", "#5C677D", "#B56576",
]

RARE_EARTH_ELEMENTS: frozenset[str] = frozenset(
    {
        "Sc", "Y",
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    }
)
RADIOACTIVE_ELEMENTS: frozenset[str] = frozenset(
    {"Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf"}
)
TOXICITY_FLAG_ELEMENTS: frozenset[str] = frozenset({"Hg", "Cd", "Pb"})
NOBLE_METAL_ELEMENTS: frozenset[str] = frozenset({"Ru", "Rh", "Pd", "Os", "Ir", "Pt", "Au"})


@dataclass(frozen=True)
class DisplayStructure:
    symbols: list[str]
    positions: np.ndarray
    cell_lattice: np.ndarray
    repeat_factors: tuple[int, int, int]
    space_group: str | None
    bond_pairs: list[tuple[int, int]]


def create_crystal_figure(
    *,
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    lattice: np.ndarray,
    title: str,
) -> go.Figure:
    display = _prepare_display_structure(
        atomic_numbers=np.asarray(atomic_numbers, dtype=int),
        positions=np.asarray(positions, dtype=float),
        lattice=np.asarray(lattice, dtype=float),
    )
    figure = go.Figure()
    unique_symbols = sorted(set(display.symbols))

    cx, cy, cz = _unit_cell_lines(display.cell_lattice)
    figure.add_trace(
        go.Scatter3d(
            x=cx, y=cy, z=cz,
            mode="lines",
            name="Unit cell",
            hoverinfo="skip",
            showlegend=False,
            line={"color": "rgba(248, 241, 231, 0.16)", "width": 1},
        )
    )

    if display.bond_pairs:
        bx, by, bz = _bond_segments(display.positions, display.bond_pairs)
        figure.add_trace(
            go.Scatter3d(
                x=bx, y=by, z=bz,
                mode="lines",
                name="Bonds",
                hoverinfo="skip",
                line={"color": "rgba(125, 189, 255, 0.58)", "width": 4},
                showlegend=False,
            )
        )

    for symbol in unique_symbols:
        mask = np.array([s == symbol for s in display.symbols])
        pts = display.positions[mask]
        hover_text = [
            f"{symbol}<br>x={c[0]:.2f} A<br>y={c[1]:.2f} A<br>z={c[2]:.2f} A"
            for c in pts
        ]
        figure.add_trace(
            go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                name=symbol,
                hovertemplate="%{text}<extra></extra>",
                text=hover_text,
                marker={
                    "size": _element_size(symbol),
                    "color": _element_color(symbol),
                    "line": {"color": "rgba(248, 241, 231, 0.35)", "width": 0.5},
                    "opacity": 0.95,
                },
            )
        )
    for axis_trace in _cartesian_axes_traces(display.cell_lattice):
        figure.add_trace(axis_trace)

    subtitle_parts = []
    if display.repeat_factors != (1, 1, 1):
        view_label = "x".join(str(v) for v in display.repeat_factors)
        subtitle_parts.append(f"{view_label} supercell")
        subtitle_parts.append("pale frame = one cell")
    else:
        subtitle_parts.append("single standardized cell")
    if display.space_group:
        subtitle_parts.append(f"space group {display.space_group}")
    subtitle = " · ".join(subtitle_parts)

    figure.update_layout(
        title={"text": title, "x": 0.02, "font": {"size": 16, "color": "#F8F1E7"}},
        margin={"l": 0, "r": 0, "t": 36, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        autosize=True,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.02,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(12, 22, 34, 0.55)",
            "font": {"color": "#F8F1E7", "size": 11},
        },
        scene={
            "aspectmode": "data",
            "bgcolor": "rgba(0,0,0,0)",
            "xaxis": _axis_style(),
            "yaxis": _axis_style(),
            "zaxis": _axis_style(),
            "camera": {"eye": {"x": 1.6, "y": 1.25, "z": 0.9}},
            "domain": {"x": [0, 1], "y": [0.05, 1]},
        },
        annotations=[
            {
                "xref": "paper", "yref": "paper",
                "x": 0.02, "y": 0.02,
                "showarrow": False, "align": "left",
                "font": {"size": 11, "color": "#C8C0B5"},
                "text": subtitle,
            }
        ],
    )
    return figure


def _axis_style() -> dict[str, object]:
    return {
        "showbackground": True,
        "backgroundcolor": "rgba(248, 241, 231, 0.04)",
        "gridcolor": "rgba(248, 241, 231, 0.10)",
        "zerolinecolor": "rgba(248, 241, 231, 0.16)",
        "title": "",
        "showticklabels": False,
    }


def _element_color(symbol: str) -> str:
    if symbol in CPK_COLORS:
        return CPK_COLORS[symbol]
    digest = hashlib.md5(symbol.encode("utf-8")).hexdigest()
    return _FALLBACK_PALETTE[int(digest, 16) % len(_FALLBACK_PALETTE)]


def _element_size(symbol: str) -> float:
    radius = COVALENT_RADII.get(symbol, 1.2)
    return max(6.0, min(20.0, 5.0 + 6.5 * radius))


def _prepare_display_structure(
    *,
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    lattice: np.ndarray,
) -> DisplayStructure:
    symbols = [Element.from_Z(int(z)).symbol for z in atomic_numbers]

    structure = Structure(
        lattice=Lattice(lattice),
        species=symbols,
        coords=positions,
        coords_are_cartesian=True,
    )
    for i in range(len(structure)):
        frac = structure[i].frac_coords % 1.0
        structure[i] = structure[i].species, frac

    space_group: str | None = None
    for symprec in [1e-2, 5e-2, 1e-1]:
        try:
            analyzer = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=5.0)
            conv = analyzer.get_conventional_standard_structure()
            space_group = analyzer.get_space_group_symbol()
            structure = conv
            break
        except Exception:
            continue

    for i in range(len(structure)):
        frac = structure[i].frac_coords % 1.0
        structure[i] = structure[i].species, frac

    cell_lattice = np.asarray(structure.lattice.matrix, dtype=float)

    repeat = _repeat_factors(len(structure))
    if repeat != (1, 1, 1):
        supercell = structure.copy()
        supercell.make_supercell(repeat)
        sc_symbols = [site.specie.symbol for site in supercell]
        sc_positions = np.asarray(supercell.cart_coords, dtype=float)
        bond_pairs = _compute_in_cell_bonds(sc_positions)
        return DisplayStructure(
            symbols=sc_symbols,
            positions=sc_positions,
            cell_lattice=cell_lattice,
            repeat_factors=repeat,
            space_group=space_group,
            bond_pairs=bond_pairs,
        )

    cell_symbols, cell_positions = _expand_boundary_images(structure)
    bond_pairs = _compute_in_cell_bonds(cell_positions)

    return DisplayStructure(
        symbols=cell_symbols,
        positions=cell_positions,
        cell_lattice=cell_lattice,
        repeat_factors=(1, 1, 1),
        space_group=space_group,
        bond_pairs=bond_pairs,
    )


def _repeat_factors(num_sites: int) -> tuple[int, int, int]:
    if num_sites <= 4:
        return (3, 3, 3)
    if num_sites <= 8:
        return (2, 2, 2)
    if num_sites <= 16:
        return (2, 2, 1)
    return (1, 1, 1)


def _expand_boundary_images(
    structure: Structure,
    *,
    tolerance: float = 1e-6,
) -> tuple[list[str], np.ndarray]:
    lattice_matrix = np.asarray(structure.lattice.matrix, dtype=float)
    display_symbols: list[str] = []
    display_positions: list[np.ndarray] = []
    seen: set[tuple[str, tuple[float, float, float]]] = set()

    for site in structure:
        frac = np.mod(site.frac_coords, 1.0)
        axis_shifts: list[tuple[float, ...]] = []
        for value in frac:
            if np.isclose(value, 0.0, atol=tolerance):
                axis_shifts.append((0.0, 1.0))
            else:
                axis_shifts.append((0.0,))

        for shift_x in axis_shifts[0]:
            for shift_y in axis_shifts[1]:
                for shift_z in axis_shifts[2]:
                    display_frac = frac + np.array([shift_x, shift_y, shift_z], dtype=float)
                    display_frac = np.clip(display_frac, 0.0, 1.0)
                    key = (
                        site.specie.symbol,
                        tuple(np.round(display_frac, decimals=8)),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    display_symbols.append(site.specie.symbol)
                    display_positions.append(display_frac @ lattice_matrix)

    return display_symbols, np.asarray(display_positions, dtype=float)


def _compute_in_cell_bonds(
    positions: np.ndarray,
) -> list[tuple[int, int]]:
    num_sites = len(positions)
    if num_sites <= 1:
        return []

    deltas = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    np.fill_diagonal(distances, np.inf)

    nearest_neighbor_distance = np.min(distances, axis=1)
    finite_nn = nearest_neighbor_distance[np.isfinite(nearest_neighbor_distance)]
    if finite_nn.size == 0:
        return []

    bond_pairs: list[tuple[int, int]] = []
    for i in range(num_sites):
        for j in range(i + 1, num_sites):
            local_cutoff = 1.12 * max(nearest_neighbor_distance[i], nearest_neighbor_distance[j])
            if distances[i, j] <= local_cutoff:
                bond_pairs.append((i, j))
    return bond_pairs


def _cartesian_axes_traces(lattice: np.ndarray) -> list[go.Scatter3d]:
    axis_length = max(1.0, 0.18 * float(np.max(np.linalg.norm(lattice, axis=1))))
    origin = np.zeros(3, dtype=float)
    axes = [
        ("x", "#E76F51", np.array([axis_length, 0.0, 0.0], dtype=float)),
        ("y", "#2A9D8F", np.array([0.0, axis_length, 0.0], dtype=float)),
        ("z", "#3A86FF", np.array([0.0, 0.0, axis_length], dtype=float)),
    ]

    traces: list[go.Scatter3d] = []
    for label, color, endpoint in axes:
        traces.append(
            go.Scatter3d(
                x=[origin[0], endpoint[0]],
                y=[origin[1], endpoint[1]],
                z=[origin[2], endpoint[2]],
                mode="lines",
                name=f"{label}-axis",
                hoverinfo="skip",
                showlegend=False,
                line={"color": color, "width": 5},
            )
        )
        traces.append(
            go.Scatter3d(
                x=[endpoint[0]],
                y=[endpoint[1]],
                z=[endpoint[2]],
                mode="text",
                text=[label],
                hoverinfo="skip",
                showlegend=False,
                textfont={"color": color, "size": 12},
            )
        )
    return traces


def _bond_segments(
    positions: np.ndarray,
    bond_pairs: list[tuple[int, int]],
) -> tuple[list[float], list[float], list[float]]:
    x_vals: list[float] = []
    y_vals: list[float] = []
    z_vals: list[float] = []
    for i, j in bond_pairs:
        x_vals.extend([positions[i, 0], positions[j, 0], None])
        y_vals.extend([positions[i, 1], positions[j, 1], None])
        z_vals.extend([positions[i, 2], positions[j, 2], None])
    return x_vals, y_vals, z_vals


def _unit_cell_lines(
    lattice: np.ndarray,
) -> tuple[list[float], list[float], list[float]]:
    origin = np.zeros(3, dtype=float)
    a, b, c = np.asarray(lattice, dtype=float)
    vertices = np.array([
        origin,
        origin + a,
        origin + b,
        origin + c,
        origin + a + b,
        origin + a + c,
        origin + b + c,
        origin + a + b + c,
    ])
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7),
    ]
    x_vals: list[float] = []
    y_vals: list[float] = []
    z_vals: list[float] = []
    for s, e in edges:
        x_vals.extend([vertices[s, 0], vertices[e, 0], None])
        y_vals.extend([vertices[s, 1], vertices[e, 1], None])
        z_vals.extend([vertices[s, 2], vertices[e, 2], None])
    return x_vals, y_vals, z_vals


# ---------------------------------------------------------------------------
# GNoME screening
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GNoMEScreeningConfig:
    summary_csv: Path
    cif_zip: Path
    run_dir: Path
    output_dir: Path
    required_elements: tuple[str, ...] = ("Fe", "Co", "Ni", "Mn", "Cr")
    allowed_elements: tuple[str, ...] = ()
    max_sites: int = 12
    max_elements: int = 4
    max_candidates: int | None = None
    shard_index: int = 0
    num_shards: int = 1
    batch_size: int = 64
    device: str = "auto"
    top_k: int = 100
    report_examples: int = 12
    stability_reference: float = 0.1


@dataclass(frozen=True)
class GNoMEPrediction:
    material_id: str
    formula: str
    composition: str
    elements: tuple[str, ...]
    num_sites: int
    crystal_system: str
    space_group: str
    predicted_energy_above_hull: float
    predicted_formation_energy_per_atom: float
    predicted_band_gap: float
    predicted_moment_per_atom: float
    predicted_transition_temperature: float | None
    predicted_site_moments: tuple[float, ...]
    predicted_site_moment_mean: float
    predicted_site_moment_abs_mean: float
    predicted_ordering: str
    ordering_probs: dict[str, float]
    score: float
    predicted_moment_from_sites: float | None = None
    predicted_is_magnetic_probability: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _contains_any(elements: tuple[str, ...], blocked: frozenset[str]) -> bool:
    return any(element in blocked for element in elements)


def _is_rare_earth_free(prediction: GNoMEPrediction) -> bool:
    return not _contains_any(prediction.elements, RARE_EARTH_ELEMENTS)


def _average_atomic_mass_per_atom(prediction: GNoMEPrediction) -> float:
    composition = Composition(prediction.composition)
    return float(composition.weight / composition.num_atoms)


def _promising_score(prediction: GNoMEPrediction) -> float:
    fm_like = (
        prediction.ordering_probs.get("FM", 0.0)
        + 0.9 * prediction.ordering_probs.get("FiM", 0.0)
        + 0.2 * prediction.ordering_probs.get("AFM", 0.0)
    )
    stability = max(0.0, 1.0 - max(prediction.predicted_energy_above_hull, 0.0) / 0.05)
    moment = min(max(prediction.predicted_moment_per_atom, 0.0) / 2.5, 1.5)
    transition = prediction.predicted_transition_temperature
    transition_factor = 0.55 if transition is None else min(max(transition, 0.0) / 400.0, 1.5)
    metallicity = 1.0 / (1.0 + max(prediction.predicted_band_gap, 0.0))
    practicality = 1.0
    if _contains_any(prediction.elements, RADIOACTIVE_ELEMENTS):
        practicality *= 0.15
    if _contains_any(prediction.elements, TOXICITY_FLAG_ELEMENTS):
        practicality *= 0.7
    if _contains_any(prediction.elements, RARE_EARTH_ELEMENTS):
        practicality *= 0.78
    if len(prediction.elements) > 4:
        practicality *= 0.93
    if prediction.num_sites > 20:
        practicality *= 0.92

    return float(moment * (0.2 + 0.8 * fm_like) * stability * (0.75 + 0.25 * metallicity) * transition_factor * practicality)


def _lightweight_score(prediction: GNoMEPrediction) -> float:
    avg_mass = _average_atomic_mass_per_atom(prediction)
    mass_factor = min(2.0, 50.0 / max(avg_mass, 1e-6))
    return float(_promising_score(prediction) * mass_factor)


def _low_cost_score(prediction: GNoMEPrediction) -> float:
    affordability = 1.0
    if _contains_any(prediction.elements, RADIOACTIVE_ELEMENTS):
        affordability *= 0.05
    if _contains_any(prediction.elements, TOXICITY_FLAG_ELEMENTS):
        affordability *= 0.55
    if _contains_any(prediction.elements, RARE_EARTH_ELEMENTS):
        affordability *= 0.45
    if _contains_any(prediction.elements, NOBLE_METAL_ELEMENTS):
        affordability *= 0.4
    avg_mass = _average_atomic_mass_per_atom(prediction)
    affordability *= min(1.6, 55.0 / max(avg_mass, 1e-6))
    if len(prediction.elements) > 4:
        affordability *= 0.93
    if prediction.num_sites > 20:
        affordability *= 0.95
    return float(_promising_score(prediction) * affordability)


def _candidate_row(rank: int, prediction: GNoMEPrediction) -> dict[str, object]:
    return {
        "rank": rank,
        "material_id": prediction.material_id,
        "formula": prediction.formula,
        "num_sites": prediction.num_sites,
        "elements": ",".join(prediction.elements),
        "crystal_system": prediction.crystal_system,
        "space_group": prediction.space_group,
        "rare_earth_free": _is_rare_earth_free(prediction),
        "average_atomic_mass_per_atom": round(_average_atomic_mass_per_atom(prediction), 6),
        "predicted_energy_above_hull": round(prediction.predicted_energy_above_hull, 6),
        "predicted_formation_energy_per_atom": round(prediction.predicted_formation_energy_per_atom, 6),
        "predicted_band_gap": round(prediction.predicted_band_gap, 6),
        "predicted_moment_per_atom": round(prediction.predicted_moment_per_atom, 6),
        "predicted_transition_temperature": (
            None
            if prediction.predicted_transition_temperature is None
            else round(prediction.predicted_transition_temperature, 6)
        ),
        "predicted_moment_from_sites": (
            None
            if prediction.predicted_moment_from_sites is None
            else round(prediction.predicted_moment_from_sites, 6)
        ),
        "predicted_site_moment_mean": round(prediction.predicted_site_moment_mean, 6),
        "predicted_site_moment_abs_mean": round(prediction.predicted_site_moment_abs_mean, 6),
        "predicted_is_magnetic_probability": (
            None
            if prediction.predicted_is_magnetic_probability is None
            else round(prediction.predicted_is_magnetic_probability, 6)
        ),
        "predicted_ordering": prediction.predicted_ordering,
        "score": round(prediction.score, 6),
        "promising_score": round(_promising_score(prediction), 6),
        "lightweight_score": round(_lightweight_score(prediction), 6),
        "low_cost_score": round(_low_cost_score(prediction), 6),
        **{f"p_{label}": round(prediction.ordering_probs[label], 6) for label in MAGNETIC_ORDERING_CLASSES},
    }


def _write_prediction_csv(path: Path, predictions: list[GNoMEPrediction]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "material_id",
                "formula",
                "num_sites",
                "elements",
                "crystal_system",
                "space_group",
                "rare_earth_free",
                "average_atomic_mass_per_atom",
                "predicted_energy_above_hull",
                "predicted_formation_energy_per_atom",
                "predicted_band_gap",
                "predicted_moment_per_atom",
                "predicted_transition_temperature",
                "predicted_moment_from_sites",
                "predicted_site_moment_mean",
                "predicted_site_moment_abs_mean",
                "predicted_is_magnetic_probability",
                "predicted_ordering",
                "score",
                "promising_score",
                "lightweight_score",
                "low_cost_score",
                *[f"p_{label}" for label in MAGNETIC_ORDERING_CLASSES],
            ],
        )
        writer.writeheader()
        for rank, prediction in enumerate(predictions, start=1):
            writer.writerow(_candidate_row(rank, prediction))


def _build_shortlists(
    predictions: list[GNoMEPrediction],
    *,
    shortlist_size: int = 30,
) -> dict[str, list[GNoMEPrediction]]:
    overall = predictions[:shortlist_size]
    no_rare_earth = [item for item in predictions if _is_rare_earth_free(item)][:shortlist_size]
    practical_pool = sorted(predictions, key=lambda item: (_promising_score(item), item.score), reverse=True)
    lightweight_pool = sorted(predictions, key=lambda item: (_lightweight_score(item), _promising_score(item), item.score), reverse=True)
    low_cost_pool = sorted(predictions, key=lambda item: (_low_cost_score(item), _promising_score(item), item.score), reverse=True)
    promising = practical_pool[:shortlist_size]
    return {
        "overall_top30": overall,
        "rare_earth_free_top30": no_rare_earth,
        "most_promising_top30": promising,
        "lightweight_top30": lightweight_pool[:shortlist_size],
        "low_cost_top30": low_cost_pool[:shortlist_size],
    }


def _screening_stats(predictions: list[GNoMEPrediction]) -> dict[str, object]:
    by_ordering = {label: 0 for label in MAGNETIC_ORDERING_CLASSES}
    for item in predictions:
        by_ordering[item.predicted_ordering] = by_ordering.get(item.predicted_ordering, 0) + 1
    rare_earth_free = [item for item in predictions if _is_rare_earth_free(item)]
    room_temp_like = [
        item for item in predictions
        if item.predicted_ordering in {"FM", "FiM"}
        and item.predicted_transition_temperature is not None
        and item.predicted_transition_temperature >= 300.0
        and item.predicted_energy_above_hull <= 0.05
    ]
    return {
        "predicted_ordering_counts": by_ordering,
        "rare_earth_free_candidates": len(rare_earth_free),
        "stable_candidates_below_0p05_ev": sum(item.predicted_energy_above_hull <= 0.05 for item in predictions),
        "fm_or_fim_candidates": sum(item.predicted_ordering in {"FM", "FiM"} for item in predictions),
        "room_temperature_magnet_like_candidates": len(room_temp_like),
    }


def run_gnome_screen(config: GNoMEScreeningConfig) -> tuple[list[GNoMEPrediction], dict[str, object]]:
    metrics = json.loads((config.run_dir / "metrics.json").read_text(encoding="utf-8"))
    graph_config = GraphConfig(**metrics["graph_config"])
    model_config = ModelConfig(**metrics["model_config"])
    class_names = metrics.get("class_names", list(MAGNETIC_ORDERING_CLASSES))

    builder = CrystalGraphBuilder(graph_config)
    device = resolve_device(config.device)
    model = MagNet(
        edge_dim=graph_config.num_radial,
        num_classes=len(class_names),
        config=model_config,
    ).to(device)
    checkpoint = torch.load(config.run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    filtered_total = count_filtered_rows(config)
    predictions: list[GNoMEPrediction] = []
    batch_graphs = []
    batch_rows = []
    skipped = 0

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Issues encountered while parsing CIF:.*",
            category=UserWarning,
        )
        with zipfile.ZipFile(config.cif_zip) as archive:
            iterator = tqdm(
                _iter_filtered_rows(config),
                total=filtered_total,
                desc="Screening GNoME",
                unit="material",
            )
            for row in iterator:
                try:
                    structure = _load_gnome_structure(archive, row["material_id"])
                    graph = builder.build(
                        structure,
                        target=0.0,
                        material_id=row["material_id"],
                        formula=row["formula"],
                    )
                except Exception:
                    skipped += 1
                    iterator.set_postfix(predicted=len(predictions), skipped=skipped)
                    continue

                batch_graphs.append(graph)
                batch_rows.append(row)
                if len(batch_graphs) >= config.batch_size:
                    predictions.extend(_predict_batch(model, batch_graphs, batch_rows, class_names, device, config))
                    batch_graphs = []
                    batch_rows = []
                    iterator.set_postfix(predicted=len(predictions), skipped=skipped)

            if batch_graphs:
                predictions.extend(_predict_batch(model, batch_graphs, batch_rows, class_names, device, config))
                iterator.set_postfix(predicted=len(predictions), skipped=skipped)

    predictions.sort(key=lambda item: item.score, reverse=True)
    summary = {
        "source_run": str(config.run_dir),
        "summary_csv": str(config.summary_csv),
        "cif_zip": str(config.cif_zip),
        "device": str(device),
        "filters": {
            "required_elements": list(config.required_elements),
            "allowed_elements": list(config.allowed_elements),
            "max_sites": config.max_sites,
            "max_elements": config.max_elements,
            "max_candidates": config.max_candidates,
            "shard_index": config.shard_index,
            "num_shards": config.num_shards,
        },
        "screened_rows": filtered_total,
        "successful_predictions": len(predictions),
        "skipped_structures": skipped,
    }
    return predictions, summary


def write_gnome_report(
    *,
    predictions: list[GNoMEPrediction],
    config: GNoMEScreeningConfig,
    summary: dict[str, object],
) -> dict[str, Path]:
    ensure_parent_dir(config.output_dir / "placeholder.txt")
    all_path = config.output_dir / "screened_candidates.json"
    top_path = config.output_dir / "top_candidates.csv"
    overall_top30_path = config.output_dir / "top30_overall.csv"
    no_rare_earth_top30_path = config.output_dir / "top30_no_rare_earth.csv"
    promising_top30_path = config.output_dir / "top30_most_promising.csv"
    lightweight_top30_path = config.output_dir / "top30_lightweight.csv"
    low_cost_top30_path = config.output_dir / "top30_low_cost.csv"
    report_path = config.output_dir / "report.html"
    summary_path = config.output_dir / "summary.json"

    top_predictions = predictions[: config.top_k]
    shortlists = _build_shortlists(predictions)
    stats = _screening_stats(predictions)
    all_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "predictions": [item.to_dict() for item in predictions],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _write_prediction_csv(top_path, top_predictions)
    _write_prediction_csv(overall_top30_path, shortlists["overall_top30"])
    _write_prediction_csv(no_rare_earth_top30_path, shortlists["rare_earth_free_top30"])
    _write_prediction_csv(promising_top30_path, shortlists["most_promising_top30"])
    _write_prediction_csv(lightweight_top30_path, shortlists["lightweight_top30"])
    _write_prediction_csv(low_cost_top30_path, shortlists["low_cost_top30"])

    summary_path.write_text(
        json.dumps(
            {
                **summary,
                "stats": stats,
                "top_candidate": top_predictions[0].to_dict() if top_predictions else None,
                "shortlists": {
                    key: [item.to_dict() for item in value]
                    for key, value in shortlists.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_path.write_text(
        _build_report_html(
            top_predictions[: config.report_examples],
            config,
            stats=stats,
            shortlists=shortlists,
        ),
        encoding="utf-8",
    )
    return {
        "all_predictions": all_path,
        "top_candidates_csv": top_path,
        "top30_overall_csv": overall_top30_path,
        "top30_no_rare_earth_csv": no_rare_earth_top30_path,
        "top30_most_promising_csv": promising_top30_path,
        "top30_lightweight_csv": lightweight_top30_path,
        "top30_low_cost_csv": low_cost_top30_path,
        "summary": summary_path,
        "report": report_path,
    }


def count_filtered_rows(config: GNoMEScreeningConfig) -> int:
    return sum(1 for _ in _iter_filtered_rows(config))


def _iter_filtered_rows(config: GNoMEScreeningConfig):
    required = set(config.required_elements)
    allowed = set(config.allowed_elements)
    with config.summary_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        matched_count = 0
        for row in reader:
            elements = tuple(ast.literal_eval(row["Elements"]))
            if len(elements) > config.max_elements:
                continue
            if required and not required.intersection(elements):
                continue
            if allowed and not set(elements).issubset(allowed):
                continue
            num_sites = int(row["NSites"])
            if num_sites > config.max_sites:
                continue
            if config.max_candidates is not None and matched_count >= config.max_candidates:
                break
            if config.num_shards > 1 and (matched_count % config.num_shards) != config.shard_index:
                matched_count += 1
                continue

            yield {
                "material_id": row["MaterialId"],
                "formula": row["Reduced Formula"],
                "composition": row["Composition"],
                "elements": tuple(sorted(elements)),
                "num_sites": num_sites,
                "crystal_system": row["Crystal System"],
                "space_group": row["Space Group"],
            }
            matched_count += 1


def _load_gnome_structure(archive: zipfile.ZipFile, material_id: str) -> Structure:
    member = f"by_id/{material_id}.CIF"
    with archive.open(member) as handle:
        cif_text = handle.read().decode("utf-8")
    return Structure.from_str(cif_text, fmt="cif")


@torch.no_grad()
def _predict_batch(
    model: torch.nn.Module,
    graphs: list,
    rows: list[dict[str, object]],
    class_names: list[str],
    device: torch.device,
    config: GNoMEScreeningConfig,
) -> list[GNoMEPrediction]:
    batch = Batch.from_data_list(graphs).to(device)
    outputs = model(batch)
    energy = outputs["energy"].detach().cpu().numpy()
    formation_energy = outputs["formation_energy"].detach().cpu().numpy()
    band_gap = outputs["band_gap"].detach().cpu().numpy()
    moment = outputs["magnetization"].detach().cpu().numpy()
    site_moments = outputs["site_moments"].detach().cpu().numpy()
    transition_temperature = None
    if "transition_temperature" in outputs:
        transition_temperature = outputs["transition_temperature"].detach().cpu().numpy()
    moment_from_sites = None
    if "magnetization_from_sites" in outputs:
        moment_from_sites = outputs["magnetization_from_sites"].detach().cpu().numpy()
    magnetic_prob = None
    if "magnetic_logits" in outputs:
        magnetic_prob = torch.sigmoid(outputs["magnetic_logits"]).detach().cpu().numpy()
    if "ordering_log_probs" in outputs:
        probs = outputs["ordering_log_probs"].exp().detach().cpu().numpy()
    else:
        probs = torch.softmax(outputs["ordering_logits"], dim=-1).detach().cpu().numpy()
    preds = probs.argmax(axis=1)
    ptr = batch.ptr.detach().cpu().numpy()

    resolved: list[GNoMEPrediction] = []
    for index, row in enumerate(rows):
        row_probs = {label: float(probs[index, class_idx]) for class_idx, label in enumerate(class_names)}
        fm_prob = row_probs.get("FM", 0.0)
        fim_prob = row_probs.get("FiM", 0.0)
        afm_prob = row_probs.get("AFM", 0.0)
        row_site_moments = tuple(float(value) for value in site_moments[ptr[index] : ptr[index + 1]])
        stable = max(0.0, 1.0 - max(float(energy[index]), 0.0) / max(config.stability_reference, 1e-6))
        magnetic_ordering_score = fm_prob + (0.8 * fim_prob) + (0.35 * afm_prob)
        if magnetic_prob is not None:
            magnetic_ordering_score *= float(magnetic_prob[index])
        transition_score = 1.0
        if transition_temperature is not None:
            transition_score = min(max(float(transition_temperature[index]), 0.0) / 300.0, 1.5)
        metallicity_score = 1.0 / (1.0 + max(float(band_gap[index]), 0.0))
        score = (
            max(float(moment[index]), 0.0)
            * (0.15 + 0.85 * magnetic_ordering_score)
            * stable
            * (0.75 + 0.25 * metallicity_score)
            * (0.6 + 0.4 * transition_score)
        )
        resolved.append(
            GNoMEPrediction(
                material_id=str(row["material_id"]),
                formula=str(row["formula"]),
                composition=str(row["composition"]),
                elements=tuple(row["elements"]),
                num_sites=int(row["num_sites"]),
                crystal_system=str(row["crystal_system"]),
                space_group=str(row["space_group"]),
                predicted_energy_above_hull=float(energy[index]),
                predicted_formation_energy_per_atom=float(formation_energy[index]),
                predicted_band_gap=float(band_gap[index]),
                predicted_moment_per_atom=float(moment[index]),
                predicted_transition_temperature=(
                    None if transition_temperature is None else float(transition_temperature[index])
                ),
                predicted_moment_from_sites=(
                    None if moment_from_sites is None else float(moment_from_sites[index])
                ),
                predicted_site_moments=row_site_moments,
                predicted_site_moment_mean=float(np.mean(row_site_moments)),
                predicted_site_moment_abs_mean=float(np.mean(np.abs(row_site_moments))),
                predicted_is_magnetic_probability=(
                    None if magnetic_prob is None else float(magnetic_prob[index])
                ),
                predicted_ordering=class_names[int(preds[index])],
                ordering_probs=row_probs,
                score=float(score),
            )
        )
    return resolved


def _shortlist_table(
    title: str,
    predictions: list[GNoMEPrediction],
    *,
    extra_label: str,
    extra_value_fn,
) -> str:
    rows = []
    for rank, item in enumerate(predictions, start=1):
        tc_text = "n/a" if item.predicted_transition_temperature is None else f"{item.predicted_transition_temperature:.1f}"
        avg_mass = _average_atomic_mass_per_atom(item)
        rows.append(
            f"""
            <tr>
              <td>{rank}</td>
              <td>{html.escape(item.formula)}</td>
              <td>{html.escape(item.material_id)}</td>
              <td>{html.escape(item.predicted_ordering)}</td>
              <td>{item.predicted_moment_per_atom:.3f}</td>
              <td>{tc_text}</td>
              <td>{item.predicted_energy_above_hull:.3f}</td>
              <td>{item.predicted_band_gap:.3f}</td>
              <td>{avg_mass:.1f}</td>
              <td>{item.score:.3f}</td>
              <td>{_promising_score(item):.3f}</td>
              <td>{extra_value_fn(item):.3f}</td>
            </tr>
            """
        )
    return f"""
    <section class="table-card">
      <h2>{html.escape(title)}</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Formula</th>
              <th>Material ID</th>
              <th>Order</th>
              <th>Moment</th>
              <th>Tc/TN (K)</th>
              <th>E hull</th>
              <th>Gap</th>
              <th>Avg mass</th>
              <th>Score</th>
              <th>Promising</th>
              <th>{html.escape(extra_label)}</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </div>
    </section>
    """


def _build_report_html(
    predictions: list[GNoMEPrediction],
    config: GNoMEScreeningConfig,
    *,
    stats: dict[str, object],
    shortlists: dict[str, list[GNoMEPrediction]],
) -> str:
    cards: list[str] = []
    for rank, item in enumerate(predictions, start=1):
        prob_text = " · ".join(
            f"{label} {item.ordering_probs[label]:.2f}" for label in MAGNETIC_ORDERING_CLASSES
        )
        tc_text = "n/a" if item.predicted_transition_temperature is None else f"{item.predicted_transition_temperature:.1f} K"
        cards.append(
            f"""
            <article class="card candidate-card">
              <div class="meta">
                <div class="rank">Rank {rank}</div>
                <h3>{html.escape(item.formula)}</h3>
                <p>{html.escape(item.material_id)} · {item.num_sites} atoms · {html.escape(item.space_group)}</p>
              </div>
              <div class="metrics">
                <div><span>Pred moment</span><strong>{item.predicted_moment_per_atom:.3f} mu_B/atom</strong></div>
                <div><span>Pred order</span><strong>{html.escape(item.predicted_ordering)}</strong></div>
                <div><span>Pred hull</span><strong>{item.predicted_energy_above_hull:.3f} eV</strong></div>
                <div><span>Pred form E</span><strong>{item.predicted_formation_energy_per_atom:.3f} eV</strong></div>
                <div><span>Pred Tc/TN</span><strong>{tc_text}</strong></div>
                <div><span>Pred gap</span><strong>{item.predicted_band_gap:.3f} eV</strong></div>
                <div><span>Site |mu| mean</span><strong>{item.predicted_site_moment_abs_mean:.3f} mu_B</strong></div>
                <div><span>Moment from sites</span><strong>{0.0 if item.predicted_moment_from_sites is None else item.predicted_moment_from_sites:.3f} mu_B/atom</strong></div>
                <div><span>Avg atomic mass</span><strong>{_average_atomic_mass_per_atom(item):.1f}</strong></div>
                <div><span>Score</span><strong>{item.score:.3f}</strong></div>
              </div>
              <p class="probs">{html.escape(prob_text)}</p>
            </article>
            """
        )
    stats_cards = [
        ("Screened", f"{stats.get('stable_candidates_below_0p05_ev', 0):,} stable<=0.05 eV"),
        ("FM/FiM", f"{stats.get('fm_or_fim_candidates', 0):,} candidates"),
        ("Rare-earth-free", f"{stats.get('rare_earth_free_candidates', 0):,} candidates"),
        ("Room-temp-like", f"{stats.get('room_temperature_magnet_like_candidates', 0):,} candidates"),
    ]
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GNoME Magnetic Screening</title>
  <style>
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, sans-serif;
      background: linear-gradient(180deg, #08131f 0%, #10253a 100%);
      color: #f5efe3;
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .layout {{
      display: block;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 34px;
    }}
    .subtitle {{
      color: #c6bfaf;
      margin: 0 0 24px;
    }}
    .card {{
      background: rgba(11, 21, 32, 0.84);
      border: 1px solid rgba(245, 239, 227, 0.10);
      border-radius: 20px;
      padding: 18px;
      margin-bottom: 18px;
      box-shadow: 0 16px 48px rgba(0, 0, 0, 0.28);
    }}
    .meta h3 {{
      margin: 4px 0;
      font-size: 24px;
    }}
    .meta p, .probs {{
      color: #c6bfaf;
      margin: 6px 0 0;
    }}
    .rank {{
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #ffb703;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin: 14px 0 10px;
    }}
    .summary-strip {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 0 0 22px;
    }}
    .summary-stat {{
      background: rgba(11, 21, 32, 0.84);
      border: 1px solid rgba(245, 239, 227, 0.10);
      border-radius: 16px;
      padding: 14px 16px;
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.22);
    }}
    .summary-stat span {{
      display: block;
      color: #c6bfaf;
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .summary-stat strong {{
      font-size: 20px;
    }}
    .metrics div {{
      background: rgba(245, 239, 227, 0.05);
      border-radius: 14px;
      padding: 10px 12px;
    }}
    .metrics span {{
      display: block;
      font-size: 12px;
      color: #c6bfaf;
      margin-bottom: 4px;
    }}
    .metrics strong {{
      font-size: 17px;
    }}
    .table-card {{
      background: rgba(11, 21, 32, 0.84);
      border: 1px solid rgba(245, 239, 227, 0.10);
      border-radius: 20px;
      padding: 18px;
      margin-top: 24px;
      box-shadow: 0 16px 48px rgba(0, 0, 0, 0.28);
    }}
    .table-card h2 {{
      margin: 0 0 14px;
      font-size: 22px;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid rgba(245, 239, 227, 0.08);
      white-space: nowrap;
    }}
    th {{
      color: #ffb703;
      font-weight: 700;
      font-size: 12px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
  </style>
</head>
<body>
  <main>
    <h1>GNoME Magnetic Screening</h1>
    <p class="subtitle">Filtered to {("structures containing at least one of " + ", ".join(config.required_elements)) if config.required_elements else "all elements"}{(" and only using elements from " + ", ".join(config.allowed_elements)) if config.allowed_elements else ""}, at most {config.max_sites} atoms, and at most {config.max_elements} elements.</p>
    <section class="summary-strip">
      {"".join(f'<div class="summary-stat"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>' for label, value in stats_cards)}
    </section>
    <div class="layout">
      <section>
        {"".join(cards)}
      </section>
    </div>
    {_shortlist_table("Top 30 Overall", shortlists["overall_top30"], extra_label="Overall", extra_value_fn=lambda item: item.score)}
    {_shortlist_table("Top 30 Rare-Earth-Free", shortlists["rare_earth_free_top30"], extra_label="Overall", extra_value_fn=lambda item: item.score)}
    {_shortlist_table("Top 30 Most Promising", shortlists["most_promising_top30"], extra_label="Promising", extra_value_fn=_promising_score)}
    {_shortlist_table("Top 30 Lightweight", shortlists["lightweight_top30"], extra_label="Lightweight", extra_value_fn=_lightweight_score)}
    {_shortlist_table("Top 30 Low Cost", shortlists["low_cost_top30"], extra_label="Low-cost", extra_value_fn=_low_cost_score)}
  </main>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI: screen / merge
# ---------------------------------------------------------------------------


def _build_screen_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Screen GNoME structures for magnetic materials, or merge sharded results."
    )
    subparsers = parser.add_subparsers(dest="command")

    # Default (no subcommand) = screen
    parser.add_argument("--run-dir", type=Path, default=Path("runs/masked_multitask_mp_large_v2"))
    parser.add_argument("--summary-csv", type=Path, default=Path("data/gnome/stable_materials_summary.csv"))
    parser.add_argument("--cif-zip", type=Path, default=Path("data/gnome/by_id.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/gnome_screen_masked"))
    parser.add_argument("--required-elements", type=str, default="Fe,Co,Ni,Mn,Cr")
    parser.add_argument("--allowed-elements", type=str, default="")
    parser.add_argument("--max-sites", type=int, default=40)
    parser.add_argument("--max-elements", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--report-examples", type=int, default=12)

    # Merge subcommand
    merge_parser = subparsers.add_parser("merge", help="Merge sharded screening outputs into one report.")
    merge_parser.add_argument("--run-dir", type=Path, required=True)
    merge_parser.add_argument("--summary-csv", type=Path, required=True)
    merge_parser.add_argument("--cif-zip", type=Path, required=True)
    merge_parser.add_argument("--shards-dir", type=Path, required=True)
    merge_parser.add_argument("--output-dir", type=Path, required=True)
    merge_parser.add_argument("--top-k", type=int, default=200)
    merge_parser.add_argument("--report-examples", type=int, default=16)

    return parser


def _run_screen(args) -> None:
    config = GNoMEScreeningConfig(
        summary_csv=args.summary_csv,
        cif_zip=args.cif_zip,
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        required_elements=tuple(item.strip() for item in args.required_elements.split(",") if item.strip()),
        allowed_elements=tuple(item.strip() for item in args.allowed_elements.split(",") if item.strip()),
        max_sites=args.max_sites,
        max_elements=args.max_elements,
        max_candidates=args.max_candidates,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        batch_size=args.batch_size,
        device=args.device,
        top_k=args.top_k,
        report_examples=args.report_examples,
    )

    predictions, summary = run_gnome_screen(config)
    outputs = write_gnome_report(predictions=predictions, config=config, summary=summary)

    top_preview = [item.to_dict() for item in predictions[:5]]
    print(json.dumps({"summary": summary, "top_preview": top_preview}, indent=2))
    for name, path in outputs.items():
        print(f"{name}: {path}")


def _run_merge(args) -> None:
    shard_paths = sorted(args.shards_dir.glob("shard_*/screened_candidates.json"))
    if not shard_paths:
        raise SystemExit(f"No shard outputs found under {args.shards_dir}")

    predictions: list[GNoMEPrediction] = []
    shard_summaries: list[dict[str, object]] = []
    for path in shard_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        shard_summaries.append(payload["summary"])
        predictions.extend(GNoMEPrediction(**row) for row in payload["predictions"])

    predictions.sort(key=lambda item: item.score, reverse=True)
    first_filters = shard_summaries[0]["filters"]
    config = GNoMEScreeningConfig(
        run_dir=args.run_dir,
        summary_csv=args.summary_csv,
        cif_zip=args.cif_zip,
        output_dir=args.output_dir,
        required_elements=tuple(first_filters["required_elements"]),
        allowed_elements=tuple(first_filters["allowed_elements"]),
        max_sites=int(first_filters["max_sites"]),
        max_elements=int(first_filters["max_elements"]),
        top_k=args.top_k,
        report_examples=args.report_examples,
    )
    summary = {
        "source_run": str(args.run_dir),
        "summary_csv": str(args.summary_csv),
        "cif_zip": str(args.cif_zip),
        "device": "merged",
        "filters": {
            **first_filters,
            "shard_index": None,
            "num_shards": len(shard_paths),
        },
        "screened_rows": int(sum(int(item["screened_rows"]) for item in shard_summaries)),
        "successful_predictions": int(sum(int(item["successful_predictions"]) for item in shard_summaries)),
        "skipped_structures": int(sum(int(item["skipped_structures"]) for item in shard_summaries)),
        "shard_dirs": [str(path.parent) for path in shard_paths],
    }
    outputs = write_gnome_report(predictions=predictions, config=config, summary=summary)
    print(json.dumps({"summary": summary, "outputs": {key: str(value) for key, value in outputs.items()}}, indent=2))


def main() -> None:
    args = _build_screen_parser().parse_args()
    if args.command == "merge":
        _run_merge(args)
    else:
        _run_screen(args)


if __name__ == "__main__":
    main()
