"""Data loading, crystal graphs, and PyG datasets for magnetic materials screening."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import urllib.parse
import urllib.request
import warnings
import zipfile
from abc import ABC, abstractmethod
from collections import Counter, deque
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Element, Structure
from torch_geometric.data import Data, InMemoryDataset
from tqdm.auto import tqdm

warnings.filterwarnings(
    "ignore",
    message=r"No data available for .*",
    module=r"pymatgen\.core\.periodic_table",
)
warnings.filterwarnings(
    "ignore",
    message=r"No Pauling electronegativity for .*",
    module=r"pymatgen\.core\.periodic_table",
)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if requested == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


# ---------------------------------------------------------------------------
# Graph configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphConfig:
    cutoff: float = 5.0
    num_radial: int = 64
    radial_width: float = 0.5
    max_atomic_number: int = 118
    feature_version: str = "chem_node_geo_v2"

    def cache_key(self) -> str:
        cutoff = f"{self.cutoff:.2f}".replace(".", "p")
        width = f"{self.radial_width:.2f}".replace(".", "p")
        return f"{self.feature_version}_cutoff_{cutoff}_rbf_{self.num_radial}_width_{width}"


# ---------------------------------------------------------------------------
# Chemistry features
# ---------------------------------------------------------------------------

BLOCK_INDEX = {"s": 0, "p": 1, "d": 2, "f": 3}
NODE_SCALAR_FEATURE_DIM = 13


@lru_cache(maxsize=256)
def element_scalar_features(atomic_number: int) -> tuple[float, ...]:
    element = Element.from_Z(int(atomic_number))
    full_shell = element.full_electronic_structure
    max_n = max(shell[0] for shell in full_shell)

    valence_electrons = 0.0
    d_electrons = 0.0
    f_electrons = 0.0
    for n, orbital, electrons in full_shell:
        if orbital == "d":
            d_electrons += float(electrons)
        elif orbital == "f":
            f_electrons += float(electrons)
        if n == max_n:
            valence_electrons += float(electrons)
        elif orbital == "d" and n == (max_n - 1):
            valence_electrons += float(electrons)
        elif orbital == "f" and n == (max_n - 2):
            valence_electrons += float(electrons)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"No data available for .*")
        warnings.filterwarnings("ignore", message=r"No Pauling electronegativity for .*")
        electronegativity = _safe_float(getattr(element, "X", 0.0))
        atomic_radius = _radius_value(element.atomic_radius, element.atomic_radius_calculated)
        metallic_radius = _radius_value(getattr(element, "metallic_radius", None))
    mendeleev = _safe_float(getattr(element, "mendeleev_no", 0.0))
    block = getattr(element, "block", None)
    block_index = BLOCK_INDEX.get(block, -1)

    block_one_hot = [0.0, 0.0, 0.0, 0.0]
    if block_index >= 0:
        block_one_hot[block_index] = 1.0

    return (
        float((element.group or 0) / 18.0),
        float((element.row or 0) / 7.0),
        electronegativity / 4.0,
        atomic_radius / 3.0,
        metallic_radius / 3.0,
        valence_electrons / 16.0,
        d_electrons / 10.0,
        f_electrons / 14.0,
        mendeleev / 103.0,
        *block_one_hot,
    )


def build_node_scalar_features(atomic_numbers: torch.Tensor) -> torch.Tensor:
    rows = [element_scalar_features(int(z)) for z in atomic_numbers.tolist()]
    return torch.tensor(rows, dtype=torch.float32)


def _radius_value(*candidates) -> float:
    for value in candidates:
        if value is None:
            continue
        try:
            return _safe_float(value)
        except TypeError:
            continue
    return 0.0


def _safe_float(value) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(resolved):
        return 0.0
    return resolved


# ---------------------------------------------------------------------------
# Crystal graph builder
# ---------------------------------------------------------------------------

NODE_GEOMETRY_FEATURE_DIM = 10


class GaussianRadialBasis:
    def __init__(self, cutoff: float, num_radial: int, width: float) -> None:
        self.cutoff = cutoff
        self.num_radial = num_radial
        self.width = width
        self.centers = torch.linspace(0.0, cutoff, num_radial)
        self.gamma = 1.0 / max(width * width, 1e-12)

    def __call__(self, distances: torch.Tensor) -> torch.Tensor:
        centered = distances.unsqueeze(-1) - self.centers.to(distances.device)
        basis = torch.exp(-self.gamma * centered.pow(2))
        cutoff_ratio = (distances / max(self.cutoff, 1e-6)).clamp(min=0.0, max=1.0)
        envelope = 0.5 * (torch.cos(torch.pi * cutoff_ratio) + 1.0)
        return basis * envelope.unsqueeze(-1)


class CrystalGraphBuilder:
    def __init__(self, config: GraphConfig | None = None) -> None:
        self.config = config or GraphConfig()
        self.radial_basis = GaussianRadialBasis(
            cutoff=self.config.cutoff,
            num_radial=self.config.num_radial,
            width=self.config.radial_width,
        )

    def build(
        self,
        structure: Structure,
        *,
        target: float | int,
        target_dtype: torch.dtype = torch.float32,
        material_id: str | None = None,
        formula: str | None = None,
    ) -> Data:
        center_indices, neighbor_indices, image_offsets, distances = structure.get_neighbor_list(
            self.config.cutoff
        )
        if len(distances) == 0:
            raise ValueError("No neighbors found within the cutoff.")

        cart_coords = np.asarray(structure.cart_coords, dtype=np.float32)
        lattice = np.asarray(structure.lattice.matrix, dtype=np.float32)
        image_offsets = np.asarray(image_offsets, dtype=np.float32)
        cartesian_offsets = image_offsets @ lattice
        edge_vectors = cart_coords[neighbor_indices] + cartesian_offsets - cart_coords[center_indices]
        atomic_numbers = torch.tensor([_site_atomic_number(site) for site in structure], dtype=torch.long)

        edge_distance = torch.as_tensor(distances, dtype=torch.float32)
        edge_vectors_tensor = torch.as_tensor(edge_vectors, dtype=torch.float32)
        node_scalar_features = build_node_scalar_features(atomic_numbers)
        node_geometry_features = _compute_node_geometry_features(
            num_nodes=len(structure),
            center_indices=center_indices,
            neighbor_indices=neighbor_indices,
            edge_vectors=edge_vectors_tensor,
            edge_distance=edge_distance,
            neighbor_atomic_numbers=atomic_numbers[torch.as_tensor(neighbor_indices, dtype=torch.long)],
            cutoff=self.config.cutoff,
        )
        data = Data(
            z=atomic_numbers,
            node_scalar_features=node_scalar_features,
            node_geometry_features=node_geometry_features,
            pos=torch.as_tensor(cart_coords, dtype=torch.float32),
            lattice=torch.as_tensor(lattice, dtype=torch.float32).unsqueeze(0),
            edge_index=torch.tensor(
                np.vstack([center_indices, neighbor_indices]),
                dtype=torch.long,
            ),
            edge_attr=self.radial_basis(edge_distance),
            edge_distance=edge_distance.unsqueeze(-1),
            edge_vector=edge_vectors_tensor,
            cell_offset=torch.as_tensor(image_offsets, dtype=torch.float32),
            y=torch.tensor([target], dtype=target_dtype),
            num_nodes=len(structure),
        )
        if "triplet" in self.config.feature_version:
            triplet_edge_index, triplet_angle = _build_triplet_interactions(
                num_nodes=len(structure),
                center_indices=center_indices,
                edge_vectors=edge_vectors_tensor,
                edge_distance=edge_distance,
            )
            data.triplet_edge_index = triplet_edge_index
            data.triplet_angle = triplet_angle
        if material_id is not None:
            data.material_id = material_id
        if formula is not None:
            data.formula = formula
        return data


def _site_atomic_number(site: object) -> int:
    specie = getattr(site, "specie", None)
    if specie is not None:
        return int(specie.Z)

    species = getattr(site, "species", None)
    if species is None:
        raise ValueError("Site is missing both specie and species information.")

    best_atomic_number: int | None = None
    best_occupancy = -1.0
    for entry, occupancy in species.items():
        if occupancy is None:
            continue
        occupancy_value = float(occupancy)
        if occupancy_value <= 0.0:
            continue
        atomic_number = getattr(entry, "Z", None)
        if atomic_number is None and getattr(entry, "element", None) is not None:
            atomic_number = getattr(entry.element, "Z", None)
        if atomic_number is None:
            continue
        if occupancy_value > best_occupancy:
            best_atomic_number = int(atomic_number)
            best_occupancy = occupancy_value

    if best_atomic_number is None:
        raise ValueError("Could not infer an atomic number from a disordered site.")
    return best_atomic_number


def _build_center_edge_groups(
    center_indices: np.ndarray,
    *,
    num_nodes: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    counts = np.bincount(center_indices, minlength=num_nodes)
    groups: list[torch.Tensor] = [torch.empty(0, dtype=torch.long) for _ in range(num_nodes)]
    if center_indices.size == 0:
        return groups, torch.zeros(num_nodes, dtype=torch.float32)

    order = np.argsort(center_indices, kind="stable")
    offset = 0
    for node_index, count in enumerate(counts):
        count_int = int(count)
        if count_int > 0:
            groups[node_index] = torch.as_tensor(order[offset : offset + count_int], dtype=torch.long)
        offset += count_int

    return groups, torch.as_tensor(counts, dtype=torch.float32)


def _compute_node_geometry_features(
    *,
    num_nodes: int,
    center_indices: np.ndarray,
    neighbor_indices: np.ndarray,
    edge_vectors: torch.Tensor,
    edge_distance: torch.Tensor,
    neighbor_atomic_numbers: torch.Tensor,
    cutoff: float,
) -> torch.Tensor:
    features = torch.zeros((num_nodes, NODE_GEOMETRY_FEATURE_DIM), dtype=torch.float32)
    center_edge_groups, counts = _build_center_edge_groups(center_indices, num_nodes=num_nodes)
    unit_vectors = edge_vectors / edge_distance.clamp(min=1e-6).unsqueeze(-1)
    log_norm = torch.log(torch.tensor(33.0, dtype=torch.float32))
    cutoff_tensor = torch.tensor(cutoff, dtype=torch.float32)

    for node_index, edge_ids in enumerate(center_edge_groups):
        if edge_ids.numel() == 0:
            continue

        node_distances = edge_distance.index_select(0, edge_ids)
        node_units = unit_vectors.index_select(0, edge_ids)
        node_neighbor_z = neighbor_atomic_numbers.index_select(0, edge_ids).to(torch.float32)

        mean_distance = node_distances.mean()
        std_distance = node_distances.std(unbiased=False)
        min_distance = node_distances.min()
        max_distance = node_distances.max()
        mean_neighbor_z = node_neighbor_z.mean()
        std_neighbor_z = node_neighbor_z.std(unbiased=False)
        dipole_norm = node_units.mean(dim=0).norm()

        orientation_tensor = (node_units.unsqueeze(-1) * node_units.unsqueeze(-2)).mean(dim=0)
        tensor_fro = orientation_tensor.norm()
        tensor_det = torch.det(orientation_tensor).clamp(min=0.0)

        features[node_index, 0] = torch.log1p(counts[node_index]) / log_norm
        features[node_index, 1] = mean_distance / cutoff_tensor
        features[node_index, 2] = std_distance / cutoff_tensor
        features[node_index, 3] = min_distance / cutoff_tensor
        features[node_index, 4] = max_distance / cutoff_tensor
        features[node_index, 5] = mean_neighbor_z / 118.0
        features[node_index, 6] = std_neighbor_z / 118.0
        features[node_index, 7] = dipole_norm
        features[node_index, 8] = tensor_fro
        features[node_index, 9] = tensor_det * 27.0

    return features


def _build_triplet_interactions(
    *,
    num_nodes: int,
    center_indices: np.ndarray,
    edge_vectors: torch.Tensor,
    edge_distance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    center_edge_groups, _ = _build_center_edge_groups(center_indices, num_nodes=num_nodes)
    unit_vectors = edge_vectors / edge_distance.clamp(min=1e-6).unsqueeze(-1)

    source_triplets: list[torch.Tensor] = []
    target_triplets: list[torch.Tensor] = []
    triplet_angles: list[torch.Tensor] = []

    for edge_ids in center_edge_groups:
        degree = int(edge_ids.numel())
        if degree <= 1:
            continue

        row = edge_ids.repeat_interleave(degree)
        col = edge_ids.repeat(degree)
        mask = row != col
        triplet_src = row[mask]
        triplet_dst = col[mask]
        cosine = (
            unit_vectors.index_select(0, triplet_src) * unit_vectors.index_select(0, triplet_dst)
        ).sum(dim=-1).clamp(min=-1.0, max=1.0)
        angle = torch.arccos(cosine)

        source_triplets.append(triplet_src)
        target_triplets.append(triplet_dst)
        triplet_angles.append(angle)

    if not source_triplets:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
        )

    return (
        torch.stack(
            [
                torch.cat(source_triplets, dim=0),
                torch.cat(target_triplets, dim=0),
            ],
            dim=0,
        ),
        torch.cat(triplet_angles, dim=0).to(torch.float32),
    )


# ---------------------------------------------------------------------------
# Materials Project records
# ---------------------------------------------------------------------------

BASE_SUMMARY_FIELDS = [
    "material_id",
    "formula_pretty",
    "nsites",
    "structure",
]

MOMENT_SUMMARY_FIELDS = [
    *BASE_SUMMARY_FIELDS,
    "total_magnetization",
]

BENCHMARK_SUMMARY_FIELDS = [
    *BASE_SUMMARY_FIELDS,
    "total_magnetization",
    "ordering",
    "energy_above_hull",
    "formation_energy_per_atom",
    "band_gap",
    "is_stable",
    "theoretical",
]

MAGNETIC_ORDERING_CLASSES = ("NM", "FM", "FiM", "AFM")
UNKNOWN_ORDERING = "Unknown"


@dataclass(frozen=True)
class MaterialsProjectRecord:
    material_id: str
    formula: str
    num_sites: int
    total_magnetization: float | None
    moment_per_atom: float | None
    structure: dict[str, Any]
    source: str = "materials_project"
    source_id: str | None = None
    ordering: str | None = None
    energy_above_hull: float | None = None
    formation_energy_per_atom: float | None = None
    band_gap: float | None = None
    transition_temperature_k: float | None = None
    transition_temperature_type: str | None = None
    transition_temperature_match_strategy: str | None = None
    transition_temperature_source: str | None = None
    transition_temperature_hint_k: float | None = None
    transition_temperature_hint_type: str | None = None
    transition_temperature_hint_match_strategy: str | None = None
    transition_temperature_hint_source: str | None = None
    is_stable: bool | None = None
    is_theoretical: bool | None = None
    site_moments: list[float] | None = None
    source_tags: tuple[str, ...] = ()

    @classmethod
    def from_summary_doc(cls, doc: Any) -> MaterialsProjectRecord:
        structure = getattr(doc, "structure", None)
        total_magnetization = getattr(doc, "total_magnetization", None)
        num_sites = getattr(doc, "nsites", None)

        if structure is None:
            raise ValueError("Missing structure.")
        if total_magnetization is None:
            raise ValueError("Missing total_magnetization.")
        if num_sites is None:
            raise ValueError("Missing nsites.")

        formula = getattr(doc, "formula_pretty", None) or "unknown"
        return cls(
            material_id=str(getattr(doc, "material_id")),
            formula=str(formula),
            num_sites=int(num_sites),
            total_magnetization=float(total_magnetization),
            moment_per_atom=float(total_magnetization) / float(num_sites),
            structure=structure_as_serializable_dict(structure),
            source="materials_project",
            source_id=str(getattr(doc, "material_id")),
            ordering=normalize_ordering_label(getattr(doc, "ordering", None)),
            energy_above_hull=_optional_float(getattr(doc, "energy_above_hull", None)),
            formation_energy_per_atom=_optional_float(getattr(doc, "formation_energy_per_atom", None)),
            band_gap=_optional_float(getattr(doc, "band_gap", None)),
            transition_temperature_match_strategy=None,
            transition_temperature_source=None,
            is_stable=_optional_bool(getattr(doc, "is_stable", None)),
            is_theoretical=_optional_bool(getattr(doc, "theoretical", None)),
            site_moments=None,
            source_tags=("materials_project",),
        )

    @classmethod
    def from_summary_doc_partial(cls, doc: Any) -> MaterialsProjectRecord:
        structure = getattr(doc, "structure", None)
        num_sites = getattr(doc, "nsites", None)

        if structure is None:
            raise ValueError("Missing structure.")
        if num_sites is None:
            raise ValueError("Missing nsites.")

        total_magnetization = _optional_float(getattr(doc, "total_magnetization", None))
        formula = getattr(doc, "formula_pretty", None) or "unknown"
        return cls(
            material_id=str(getattr(doc, "material_id")),
            formula=str(formula),
            num_sites=int(num_sites),
            total_magnetization=total_magnetization,
            moment_per_atom=(
                None
                if total_magnetization is None
                else float(total_magnetization) / float(num_sites)
            ),
            structure=structure_as_serializable_dict(structure),
            source="materials_project",
            source_id=str(getattr(doc, "material_id")),
            ordering=normalize_ordering_label(getattr(doc, "ordering", None)),
            energy_above_hull=_optional_float(getattr(doc, "energy_above_hull", None)),
            formation_energy_per_atom=_optional_float(getattr(doc, "formation_energy_per_atom", None)),
            band_gap=_optional_float(getattr(doc, "band_gap", None)),
            transition_temperature_match_strategy=None,
            transition_temperature_source=None,
            is_stable=_optional_bool(getattr(doc, "is_stable", None)),
            is_theoretical=_optional_bool(getattr(doc, "theoretical", None)),
            site_moments=None,
            source_tags=("materials_project",),
        )

    @classmethod
    def from_json(cls, line: str) -> MaterialsProjectRecord:
        payload = json.loads(line)
        return cls(
            material_id=str(payload["material_id"]),
            formula=str(payload["formula"]),
            num_sites=int(payload["num_sites"]),
            total_magnetization=_optional_float(payload.get("total_magnetization")),
            moment_per_atom=_optional_float(payload.get("moment_per_atom")),
            structure=payload["structure"],
            source=str(payload.get("source", "materials_project")),
            source_id=_optional_str(payload.get("source_id")),
            ordering=normalize_ordering_label(payload.get("ordering")),
            energy_above_hull=_optional_float(payload.get("energy_above_hull")),
            formation_energy_per_atom=_optional_float(payload.get("formation_energy_per_atom")),
            band_gap=_optional_float(payload.get("band_gap")),
            transition_temperature_k=_optional_float(payload.get("transition_temperature_k")),
            transition_temperature_type=_optional_str(payload.get("transition_temperature_type")),
            transition_temperature_match_strategy=_optional_str(
                payload.get("transition_temperature_match_strategy")
            ),
            transition_temperature_source=_optional_str(payload.get("transition_temperature_source")),
            transition_temperature_hint_k=_optional_float(payload.get("transition_temperature_hint_k")),
            transition_temperature_hint_type=_optional_str(payload.get("transition_temperature_hint_type")),
            transition_temperature_hint_match_strategy=_optional_str(
                payload.get("transition_temperature_hint_match_strategy")
            ),
            transition_temperature_hint_source=_optional_str(
                payload.get("transition_temperature_hint_source")
            ),
            is_stable=_optional_bool(payload.get("is_stable")),
            is_theoretical=_optional_bool(payload.get("is_theoretical")),
            site_moments=_optional_float_list(payload.get("site_moments")),
            source_tags=tuple(
                payload.get("source_tags")
                or [str(payload.get("source", "materials_project"))]
            ),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    def to_structure(self) -> Structure:
        return Structure.from_dict(self.structure)

    @property
    def has_any_label(self) -> bool:
        return any(
            (
                self.energy_above_hull is not None,
                self.formation_energy_per_atom is not None,
                self.band_gap is not None,
                self.moment_per_atom is not None,
                self.ordering in MAGNETIC_ORDERING_CLASSES,
                self.transition_temperature_k is not None,
                self.site_moments is not None and len(self.site_moments) == self.num_sites,
            )
        )


@dataclass(frozen=True)
class DatasetSummary:
    num_materials: int
    mean_sites: float
    min_sites: int
    max_sites: int
    mean_abs_moment_per_atom: float | None
    mean_abs_total_magnetization: float | None
    formation_energy_coverage: float | None
    band_gap_coverage: float | None
    site_moment_coverage: float | None
    transition_temperature_coverage: float | None
    transition_temperature_hint_coverage: float | None
    ordering_counts: dict[str, int]
    source_counts: dict[str, int]
    transition_temperature_match_counts: dict[str, int]
    transition_temperature_hint_match_counts: dict[str, int]
    stable_fraction: float | None
    theoretical_fraction: float | None
    mean_energy_above_hull: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_materials_dataset(
    records: list[MaterialsProjectRecord],
) -> DatasetSummary:
    if not records:
        raise ValueError("Need at least one record to summarize the dataset.")

    num_sites = [record.num_sites for record in records]
    orderings = Counter(
        record.ordering
        for record in records
        if getattr(record, "ordering", None) is not None
    )
    source_counts = Counter(record.source for record in records)
    transition_temperature_match_counts = Counter(
        record.transition_temperature_match_strategy
        for record in records
        if getattr(record, "transition_temperature_match_strategy", None) is not None
    )
    transition_temperature_hint_match_counts = Counter(
        record.transition_temperature_hint_match_strategy
        for record in records
        if getattr(record, "transition_temperature_hint_match_strategy", None) is not None
    )
    energy_above_hull = [
        record.energy_above_hull
        for record in records
        if getattr(record, "energy_above_hull", None) is not None
    ]
    stable_flags = [
        record.is_stable
        for record in records
        if getattr(record, "is_stable", None) is not None
    ]
    theoretical_flags = [
        record.is_theoretical
        for record in records
        if getattr(record, "is_theoretical", None) is not None
    ]
    moment_per_atom = [
        record.moment_per_atom
        for record in records
        if getattr(record, "moment_per_atom", None) is not None
    ]
    total_magnetization = [
        record.total_magnetization
        for record in records
        if getattr(record, "total_magnetization", None) is not None
    ]
    formation_energy_flags = [
        1 for record in records if getattr(record, "formation_energy_per_atom", None) is not None
    ]
    band_gap_flags = [
        1 for record in records if getattr(record, "band_gap", None) is not None
    ]
    site_moment_flags = [
        1
        for record in records
        if (
            getattr(record, "site_moments", None) is not None
            and len(getattr(record, "site_moments")) == record.num_sites
        )
    ]
    transition_temperature_flags = [
        1 for record in records if getattr(record, "transition_temperature_k", None) is not None
    ]
    transition_temperature_hint_flags = [
        1 for record in records if getattr(record, "transition_temperature_hint_k", None) is not None
    ]

    return DatasetSummary(
        num_materials=len(records),
        mean_sites=sum(num_sites) / len(num_sites),
        min_sites=min(num_sites),
        max_sites=max(num_sites),
        mean_abs_moment_per_atom=(
            sum(abs(value) for value in moment_per_atom) / len(moment_per_atom)
            if moment_per_atom
            else None
        ),
        mean_abs_total_magnetization=(
            sum(abs(value) for value in total_magnetization) / len(total_magnetization)
            if total_magnetization
            else None
        ),
        formation_energy_coverage=(sum(formation_energy_flags) / len(records) if records else None),
        band_gap_coverage=(sum(band_gap_flags) / len(records) if records else None),
        site_moment_coverage=(sum(site_moment_flags) / len(records) if records else None),
        transition_temperature_coverage=(
            sum(transition_temperature_flags) / len(records) if records else None
        ),
        transition_temperature_hint_coverage=(
            sum(transition_temperature_hint_flags) / len(records) if records else None
        ),
        ordering_counts=dict(sorted(orderings.items())),
        source_counts=dict(sorted(source_counts.items())),
        transition_temperature_match_counts=dict(sorted(transition_temperature_match_counts.items())),
        transition_temperature_hint_match_counts=dict(
            sorted(transition_temperature_hint_match_counts.items())
        ),
        stable_fraction=(
            sum(1 for flag in stable_flags if flag) / len(stable_flags)
            if stable_flags
            else None
        ),
        theoretical_fraction=(
            sum(1 for flag in theoretical_flags if flag) / len(theoretical_flags)
            if theoretical_flags
            else None
        ),
        mean_energy_above_hull=(
            sum(energy_above_hull) / len(energy_above_hull)
            if energy_above_hull
            else None
        ),
    )


def download_materials_project_records(
    *,
    api_key: str | None = None,
    chunk_size: int = 500,
    num_chunks: int = 20,
    min_sites: int = 1,
    max_sites: int = 40,
    magnetic_threshold: float = 0.1,
    max_nonmagnetic_ratio: float | None = 2.0,
    seed: int = 7,
) -> list[MaterialsProjectRecord]:
    from mp_api.client import MPRester

    resolved_api_key = api_key or os.getenv("MP_API_KEY")
    if not resolved_api_key:
        raise ValueError("Set MP_API_KEY or pass api_key explicitly.")

    records: list[MaterialsProjectRecord] = []
    with MPRester(resolved_api_key) as rester:
        docs = rester.materials.summary.search(
            **_mp_search_kwargs(
                min_sites=min_sites,
                max_sites=max_sites,
                chunk_size=chunk_size,
                num_chunks=num_chunks,
                fields=MOMENT_SUMMARY_FIELDS,
            )
        )
        for doc in tqdm(docs, desc="Downloading Materials Project records", unit="material"):
            try:
                records.append(MaterialsProjectRecord.from_summary_doc(doc))
            except ValueError:
                continue

    return balance_records(
        records,
        magnetic_threshold=magnetic_threshold,
        max_nonmagnetic_ratio=max_nonmagnetic_ratio,
        seed=seed,
    )


def download_materials_project_benchmark_records(
    *,
    api_key: str | None = None,
    chunk_size: int = 500,
    num_chunks: int = 20,
    min_sites: int = 1,
    max_sites: int = 40,
    magnetic_threshold: float = 0.1,
    max_nonmagnetic_ratio: float | None = 2.0,
    max_energy_above_hull: float | None = 0.1,
    require_known_ordering: bool = False,
    exclude_theoretical: bool = False,
    include_site_moments: bool = False,
    require_site_moments: bool = False,
    magnetism_chunk_size: int = 1000,
    seed: int = 7,
) -> list[MaterialsProjectRecord]:
    from mp_api.client import MPRester

    resolved_api_key = api_key or os.getenv("MP_API_KEY")
    if not resolved_api_key:
        raise ValueError("Set MP_API_KEY or pass api_key explicitly.")

    records: list[MaterialsProjectRecord] = []
    with MPRester(resolved_api_key) as rester:
        docs = rester.materials.summary.search(
            **_mp_search_kwargs(
                min_sites=min_sites,
                max_sites=max_sites,
                chunk_size=chunk_size,
                num_chunks=num_chunks,
                fields=BENCHMARK_SUMMARY_FIELDS,
            )
        )
        for doc in tqdm(docs, desc="Downloading magnetic benchmark records", unit="material"):
            try:
                record = MaterialsProjectRecord.from_summary_doc(doc)
            except ValueError:
                continue

            if max_energy_above_hull is not None:
                if record.energy_above_hull is None or record.energy_above_hull > max_energy_above_hull:
                    continue
            if require_known_ordering and record.ordering not in MAGNETIC_ORDERING_CLASSES:
                continue
            if exclude_theoretical and record.is_theoretical:
                continue
            records.append(record)

        if include_site_moments or require_site_moments:
            records = _attach_site_moments(
                records,
                rester=rester,
                chunk_size=magnetism_chunk_size,
            )

    if require_site_moments:
        records = [
            record
            for record in records
            if record.site_moments is not None and len(record.site_moments) == record.num_sites
        ]

    return balance_records(
        records,
        magnetic_threshold=magnetic_threshold,
        max_nonmagnetic_ratio=max_nonmagnetic_ratio,
        seed=seed,
    )


def download_materials_project_masked_records(
    *,
    api_key: str | None = None,
    chunk_size: int = 500,
    num_chunks: int = 120,
    min_sites: int = 1,
    max_sites: int = 40,
    max_energy_above_hull: float | None = None,
    exclude_theoretical: bool = False,
    include_site_moments: bool = True,
    magnetism_chunk_size: int = 1000,
) -> list[MaterialsProjectRecord]:
    from mp_api.client import MPRester

    resolved_api_key = api_key or os.getenv("MP_API_KEY")
    if not resolved_api_key:
        raise ValueError("Set MP_API_KEY or pass api_key explicitly.")

    records: list[MaterialsProjectRecord] = []
    with MPRester(resolved_api_key) as rester:
        docs = rester.materials.summary.search(
            **_mp_search_kwargs(
                min_sites=min_sites,
                max_sites=max_sites,
                chunk_size=chunk_size,
                num_chunks=num_chunks,
                fields=BENCHMARK_SUMMARY_FIELDS,
            )
        )
        for doc in tqdm(docs, desc="Downloading masked MP records", unit="material"):
            try:
                record = MaterialsProjectRecord.from_summary_doc_partial(doc)
            except ValueError:
                continue

            if max_energy_above_hull is not None:
                if record.energy_above_hull is None or record.energy_above_hull > max_energy_above_hull:
                    continue
            if exclude_theoretical and record.is_theoretical:
                continue
            records.append(record)

        if include_site_moments:
            records = _attach_site_moments(
                records,
                rester=rester,
                chunk_size=magnetism_chunk_size,
            )

    return [record for record in records if record.has_any_label]


def balance_records(
    records: list[MaterialsProjectRecord],
    *,
    magnetic_threshold: float,
    max_nonmagnetic_ratio: float | None,
    seed: int,
) -> list[MaterialsProjectRecord]:
    if max_nonmagnetic_ratio is None:
        return records

    magnetic: list[MaterialsProjectRecord] = []
    nonmagnetic: list[MaterialsProjectRecord] = []
    for record in records:
        if record.ordering in MAGNETIC_ORDERING_CLASSES:
            if record.ordering == "NM":
                nonmagnetic.append(record)
            else:
                magnetic.append(record)
            continue

        if abs(record.moment_per_atom) >= magnetic_threshold:
            magnetic.append(record)
        else:
            nonmagnetic.append(record)

    if not magnetic or not nonmagnetic:
        return records

    max_nonmagnetic = max(0, int(round(max_nonmagnetic_ratio * len(magnetic))))
    if len(nonmagnetic) <= max_nonmagnetic:
        return records

    rng = random.Random(seed)
    rng.shuffle(nonmagnetic)
    balanced = magnetic + nonmagnetic[:max_nonmagnetic]
    rng.shuffle(balanced)
    return balanced


def normalize_ordering_label(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    return [float(item) for item in value]


def normalize_formula(formula: str) -> str:
    """Normalize a chemical formula for cross-database matching."""
    try:
        return Composition(formula).reduced_formula
    except Exception:
        return formula.strip()


def infer_ordering_from_site_moments(
    site_moments: list[float],
    *,
    magnetic_threshold: float = 0.1,
    compensation_threshold: float = 0.1,
) -> str:
    if not site_moments:
        return "NM"
    magnitudes = [abs(float(value)) for value in site_moments]
    if max(magnitudes, default=0.0) < magnetic_threshold:
        return "NM"
    signed_sum = float(sum(float(value) for value in site_moments))
    abs_sum = float(sum(magnitudes))
    if abs_sum <= 1e-8:
        return "NM"
    pos = any(float(value) >= magnetic_threshold for value in site_moments)
    neg = any(float(value) <= -magnetic_threshold for value in site_moments)
    if pos and neg:
        if abs(signed_sum) <= compensation_threshold * abs_sum:
            return "AFM"
        return "FiM"
    return "FM"


def structure_as_serializable_dict(structure: Structure) -> dict[str, Any]:
    payload = structure.as_dict()
    raw_magmoms = list(structure.site_properties.get("magmom", []))
    for idx, site in enumerate(payload.get("sites", [])):
        props = site.get("properties")
        if not props:
            continue
        cleaned: dict[str, Any] = {}
        for key, value in props.items():
            if key == "magmom":
                source_value = raw_magmoms[idx] if idx < len(raw_magmoms) else value
                try:
                    cleaned[key] = float(source_value)
                except Exception:
                    cleaned[key] = str(source_value)
                continue
            try:
                json.dumps(value)
                cleaned[key] = value
            except TypeError:
                cleaned[key] = str(value)
        site["properties"] = cleaned
    return payload


@dataclass(frozen=True)
class TransitionTemperatureObservation:
    temperature_k: float
    temperature_type: str
    source: str
    doi: str | None = None


def _append_temperature_entry(
    lookup: dict[str, list[TransitionTemperatureObservation]],
    *,
    formula: str,
    temperature: Any,
    temperature_type: str,
    source: str,
    doi: str | None = None,
) -> None:
    if not formula:
        return
    try:
        temperature_value = float(temperature)
    except (TypeError, ValueError):
        return
    if temperature_value <= 0:
        return
    normalized_formula = normalize_formula(formula)
    if not normalized_formula:
        return
    lookup.setdefault(normalized_formula, []).append(
        TransitionTemperatureObservation(
            temperature_k=temperature_value,
            temperature_type=temperature_type,
            source=source,
            doi=doi,
        )
    )


def _merge_source_tags(
    existing: tuple[str, ...],
    incoming: tuple[str, ...],
) -> tuple[str, ...]:
    merged: list[str] = []
    for tag in (*existing, *incoming):
        if tag and tag not in merged:
            merged.append(tag)
    return tuple(merged)


def _record_bucket_key(record: "MaterialsProjectRecord") -> tuple[str, int]:
    return normalize_formula(record.formula), int(record.num_sites)


_MERGE_VOLUME_BIN_WIDTH = 0.5
_MERGE_RATIO_BIN_WIDTH = 0.1


def _record_bucket_signature(record: "MaterialsProjectRecord") -> tuple[str, int, int, int, int]:
    formula, num_sites = _record_bucket_key(record)
    lattice = record.structure.get("lattice", {})
    lengths = sorted(
        [
            max(_optional_float(lattice.get("a")) or 0.0, 1e-6),
            max(_optional_float(lattice.get("b")) or 0.0, 1e-6),
            max(_optional_float(lattice.get("c")) or 0.0, 1e-6),
        ]
    )
    volume = max(_optional_float(lattice.get("volume")) or 0.0, 0.0)
    volume_per_atom = volume / max(num_sites, 1)
    ratio_b = lengths[1] / lengths[0]
    ratio_c = lengths[2] / lengths[0]
    return (
        formula,
        num_sites,
        int(round(volume_per_atom / _MERGE_VOLUME_BIN_WIDTH)),
        int(round(ratio_b / _MERGE_RATIO_BIN_WIDTH)),
        int(round(ratio_c / _MERGE_RATIO_BIN_WIDTH)),
    )


def _record_bucket_search_keys(record: "MaterialsProjectRecord") -> list[tuple[str, int, int, int, int]]:
    formula, num_sites, volume_bin, ratio_b_bin, ratio_c_bin = _record_bucket_signature(record)
    keys: list[tuple[str, int, int, int, int]] = [
        (formula, num_sites, volume_bin, ratio_b_bin, ratio_c_bin)
    ]
    for delta_volume in (-2, -1, 0, 1, 2):
        for delta_ratio_b in (-1, 0, 1):
            for delta_ratio_c in (-1, 0, 1):
                candidate = (
                    formula,
                    num_sites,
                    volume_bin + delta_volume,
                    ratio_b_bin + delta_ratio_b,
                    ratio_c_bin + delta_ratio_c,
                )
                if candidate not in keys:
                    keys.append(candidate)
    return keys


def _records_match_by_structure(
    left: "MaterialsProjectRecord",
    right: "MaterialsProjectRecord",
    *,
    matcher: StructureMatcher,
) -> bool:
    if left.source == right.source and left.source_id and right.source_id:
        return left.source_id == right.source_id
    if left.num_sites != right.num_sites:
        return False
    if normalize_formula(left.formula) != normalize_formula(right.formula):
        return False
    try:
        return bool(matcher.fit(left.to_structure(), right.to_structure()))
    except Exception:
        return False


def _merge_record_fields(
    existing: "MaterialsProjectRecord",
    incoming: "MaterialsProjectRecord",
    *,
    transition_temperature_match_strategy: str | None = None,
) -> "MaterialsProjectRecord":
    updated = replace(
        existing,
        total_magnetization=(
            existing.total_magnetization
            if existing.total_magnetization is not None
            else incoming.total_magnetization
        ),
        moment_per_atom=(
            existing.moment_per_atom
            if existing.moment_per_atom is not None
            else incoming.moment_per_atom
        ),
        ordering=existing.ordering if existing.ordering is not None else incoming.ordering,
        energy_above_hull=(
            existing.energy_above_hull
            if existing.energy_above_hull is not None
            else incoming.energy_above_hull
        ),
        formation_energy_per_atom=(
            existing.formation_energy_per_atom
            if existing.formation_energy_per_atom is not None
            else incoming.formation_energy_per_atom
        ),
        band_gap=existing.band_gap if existing.band_gap is not None else incoming.band_gap,
        transition_temperature_k=(
            existing.transition_temperature_k
            if existing.transition_temperature_k is not None
            else incoming.transition_temperature_k
        ),
        transition_temperature_type=(
            existing.transition_temperature_type
            if existing.transition_temperature_type is not None
            else incoming.transition_temperature_type
        ),
        transition_temperature_match_strategy=(
            existing.transition_temperature_match_strategy
            if existing.transition_temperature_match_strategy is not None
            else transition_temperature_match_strategy
            or incoming.transition_temperature_match_strategy
        ),
        transition_temperature_source=(
            existing.transition_temperature_source
            if existing.transition_temperature_source is not None
            else incoming.transition_temperature_source
        ),
        transition_temperature_hint_k=(
            existing.transition_temperature_hint_k
            if existing.transition_temperature_hint_k is not None
            else incoming.transition_temperature_hint_k
        ),
        transition_temperature_hint_type=(
            existing.transition_temperature_hint_type
            if existing.transition_temperature_hint_type is not None
            else incoming.transition_temperature_hint_type
        ),
        transition_temperature_hint_match_strategy=(
            existing.transition_temperature_hint_match_strategy
            if existing.transition_temperature_hint_match_strategy is not None
            else incoming.transition_temperature_hint_match_strategy
        ),
        transition_temperature_hint_source=(
            existing.transition_temperature_hint_source
            if existing.transition_temperature_hint_source is not None
            else incoming.transition_temperature_hint_source
        ),
        is_stable=existing.is_stable if existing.is_stable is not None else incoming.is_stable,
        is_theoretical=(
            existing.is_theoretical
            if existing.is_theoretical is not None
            else incoming.is_theoretical
        ),
        site_moments=(
            existing.site_moments
            if existing.site_moments is not None
            else incoming.site_moments
        ),
        source_tags=_merge_source_tags(
            existing.source_tags or (existing.source,),
            incoming.source_tags or (incoming.source,),
        ),
    )
    return updated


def _attach_site_moments(
    records: list[MaterialsProjectRecord],
    *,
    rester,
    chunk_size: int,
) -> list[MaterialsProjectRecord]:
    if not records:
        return records

    material_ids = [record.material_id for record in records]
    magnetism_lookup: dict[str, tuple[list[float] | None, str | None, float | None]] = {}

    for start in range(0, len(material_ids), chunk_size):
        batch_ids = material_ids[start : start + chunk_size]
        docs = rester.materials.magnetism.search(
            material_ids=batch_ids,
            chunk_size=min(chunk_size, len(batch_ids)),
            fields=["material_id", "ordering", "magmoms", "total_magnetization"],
        )
        for doc in docs:
            magnetism_lookup[str(doc.material_id)] = (
                _optional_float_list(getattr(doc, "magmoms", None)),
                normalize_ordering_label(getattr(doc, "ordering", None)),
                _optional_float(getattr(doc, "total_magnetization", None)),
            )

    enriched: list[MaterialsProjectRecord] = []
    for record in records:
        site_moments, magnetism_ordering, magnetism_total = magnetism_lookup.get(
            record.material_id,
            (None, None, None),
        )
        if site_moments is not None and len(site_moments) != record.num_sites:
            site_moments = None

        enriched.append(
            replace(
                record,
                ordering=record.ordering or magnetism_ordering,
                total_magnetization=(
                    record.total_magnetization
                    if record.total_magnetization is not None
                    else magnetism_total
                ),
                site_moments=site_moments,
            )
        )
    return enriched


# ---------------------------------------------------------------------------
# External Curie / Neel temperature data
# ---------------------------------------------------------------------------
#
# Court & Cole (2018): ~5.7k Tc/TN entries (NLP-extracted from papers).
#   Figshare collection: https://doi.org/10.6084/m9.figshare.c.3954418
#
# Updated dataset (2025): ~56k entries.
#   Paper: doi:10.1038/s41597-025-06244-6
#
# NEMAD (2025): 67k magnetic materials entries.
#   Website: https://www.nemad.org
#   Paper: Nature Communications (2025), arXiv:2409.15675
#
# NEMAD + ICSD aligned (2026): ~8.2k Tc + 11.3k TN with CIF structures.
#   Paper: arXiv:2602.00756
#
# Expected CSV format (auto-detected):
#   formula/compound, curie_temp/neel_temp/value, [doi]
# Expected JSON format:
#   [{"formula": "Fe", "curie_temp": 1043.0, ...}, ...]


def _download_file(url: str, dest: Path, *, desc: str = "") -> Path:
    """Download a file from URL if it doesn't already exist."""
    if dest.exists():
        return dest
    ensure_parent_dir(dest)
    print(f"Downloading {desc or url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


def _download_and_extract_jarvis(dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / "jdft_3d-12-12-2022.json.zip"
    json_path = dest_dir / "jdft_3d.json"
    if json_path.exists():
        return json_path
    _download_file(_JARVIS_LATEST_ZIP_URL, archive_path, desc="JARVIS jdft_3d")
    with zipfile.ZipFile(archive_path) as zf:
        json_members = [name for name in zf.namelist() if name.lower().endswith(".json")]
        if not json_members:
            raise FileNotFoundError("JARVIS archive does not contain a JSON file.")
        member = json_members[0]
        with zf.open(member) as src, json_path.open("wb") as dst:
            dst.write(src.read())
    return json_path


def _fetch_text(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    return payload.decode("utf-8", errors="ignore")


def _mp_search_kwargs(
    *,
    min_sites: int,
    max_sites: int,
    chunk_size: int,
    num_chunks: int | None,
    fields: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_sites": (min_sites, max_sites),
        "chunk_size": chunk_size,
        "fields": fields,
    }
    if num_chunks is not None and num_chunks > 0:
        kwargs["num_chunks"] = num_chunks
    return kwargs


_TC_CSV_COLUMN_NAMES = {
    "formula",
    "compound",
    "chemical_formula",
    "clean_chemical_formula",
    "normalized_composition",
    "example_formula",
    "material",
    "material_name",
    "new_column_concatenated",
}
_CURIE_VALUE_COLUMN_NAMES = {
    "curie_temp",
    "tc",
    "curie_temperature",
    "curie",
    "curie(tc)",
    "mean_tc_k",
}
_NEEL_VALUE_COLUMN_NAMES = {
    "neel_temp",
    "tn",
    "neel_temperature",
    "neel",
    "neel(tn)",
    "mean_tn_k",
}
_GENERIC_TEMPERATURE_COLUMN_NAMES = {"value", "temperature", "transition_temperature", "temperature_k"}

_NEMAD_GITHUB_URLS = {
    "fm_with_curie": "https://raw.githubusercontent.com/sumanitani/NEMAD-MagneticML/main/Dataset/FM_with_curie.csv",
    "afm_with_neel": "https://raw.githubusercontent.com/sumanitani/NEMAD-MagneticML/main/Dataset/AFM_with_Neel.csv",
}
_JARVIS_LATEST_ZIP_URL = "https://ndownloader.figshare.com/files/38521619"

_MAGNDATA_BASE_URL = "https://cryst.ehu.es/magndata/"
_MAGNDATA_ENTRY_URL = _MAGNDATA_BASE_URL + "index.php?index={index}"
_MAGNDATA_TEMPERATURE_RE = re.compile(
    r"Transition Temperature:\s*</b>\s*([0-9]+(?:\.[0-9]+)?)\s*K",
    flags=re.IGNORECASE,
)
_MAGNDATA_MCID_RE = re.compile(r'href="([^"]+\.mcif)"', flags=re.IGNORECASE)
_MAGNDATA_NAV_RE = re.compile(
    r'<input\s+type=hidden\s+name=index\s+value=([^ >]+)><input\s+type=submit\s+name=submit\s+value="(Previous|Next) entry"',
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class MagndataEntry:
    index: str
    temperature_k: float
    mcif_href: str
    previous_index: str | None
    next_index: str | None


def _parse_magndata_navigation(html: str) -> tuple[str | None, str | None]:
    previous_index: str | None = None
    next_index: str | None = None
    for value, direction in _MAGNDATA_NAV_RE.findall(html):
        if direction.lower().startswith("previous"):
            previous_index = value
        elif direction.lower().startswith("next"):
            next_index = value
    return previous_index, next_index


def _parse_magndata_entry(index: str, html: str) -> MagndataEntry | None:
    temp_match = _MAGNDATA_TEMPERATURE_RE.search(html)
    mcif_match = _MAGNDATA_MCID_RE.search(html)
    if temp_match is None or mcif_match is None:
        return None

    previous_index, next_index = _parse_magndata_navigation(html)

    return MagndataEntry(
        index=index,
        temperature_k=float(temp_match.group(1)),
        mcif_href=urllib.parse.urljoin(_MAGNDATA_BASE_URL, mcif_match.group(1)),
        previous_index=previous_index,
        next_index=next_index,
    )


def _detect_columns(
    headers: list[str],
) -> tuple[str | None, str | None, list[tuple[str, str]]]:
    """Auto-detect formula, type, and temperature columns from CSV headers."""
    lower_headers = {h.lower().strip(): h for h in headers}
    formula_col = None
    for name in _TC_CSV_COLUMN_NAMES:
        if name in lower_headers:
            formula_col = lower_headers[name]
            break
    type_col = None
    for name in ("type", "temp_type", "temperature_type", "ordering_type"):
        if name in lower_headers:
            type_col = lower_headers[name]
            break
    temperature_columns: list[tuple[str, str]] = []
    for name in _CURIE_VALUE_COLUMN_NAMES:
        if name in lower_headers:
            temperature_columns.append((lower_headers[name], "Curie"))
    for name in _NEEL_VALUE_COLUMN_NAMES:
        if name in lower_headers:
            temperature_columns.append((lower_headers[name], "Neel"))
    if not temperature_columns:
        for name in _GENERIC_TEMPERATURE_COLUMN_NAMES:
            if name in lower_headers:
                temperature_columns.append((lower_headers[name], "Generic"))
                break
    return formula_col, type_col, temperature_columns


def load_curie_neel_lookup(
    *paths: str | Path,
    default_type: str | None = None,
) -> dict[str, list[TransitionTemperatureObservation]]:
    """Load Curie/Neel temperature data from CSV or JSON files.

    Returns a dict mapping normalized formula to [(temperature_k, type)].
    The *default_type* is used when the file has no type column — if ``None``,
    it is guessed from the filename (e.g. ``curie_temps.csv`` → ``"Curie"``).

    Supports Court & Cole, NEMAD exports, and similar tabular formats.
    """
    lookup: dict[str, list[TransitionTemperatureObservation]] = {}

    for raw_path in paths:
        path = Path(raw_path)
        # Guess default type from filename if not specified
        file_default = default_type
        if file_default is None:
            name_lower = path.stem.lower()
            if "neel" in name_lower or "_tn" in name_lower:
                file_default = "Neel"
            else:
                file_default = "Curie"
        file_source = path.stem.lower()

        if path.suffix == ".json":
            with path.open(encoding="utf-8") as f:
                entries = json.load(f)
            if isinstance(entries, dict):
                entries = list(entries.values())
            for entry in entries:
                formula = str(
                    entry.get("formula")
                    or entry.get("compound")
                    or entry.get("material_name")
                    or entry.get("chemical_formula", "")
                )
                entry_source = str(entry.get("source") or file_source)
                entry_doi = _optional_str(entry.get("doi") or entry.get("DOI"))
                appended = False
                for key in ("curie_temp", "tc", "curie_temperature"):
                    if entry.get(key) is not None:
                        _append_temperature_entry(
                            lookup,
                            formula=formula,
                            temperature=entry.get(key),
                            temperature_type="Curie",
                            source=entry_source,
                            doi=entry_doi,
                        )
                        appended = True
                for key in ("neel_temp", "tn", "neel_temperature"):
                    if entry.get(key) is not None:
                        _append_temperature_entry(
                            lookup,
                            formula=formula,
                            temperature=entry.get(key),
                            temperature_type="Neel",
                            source=entry_source,
                            doi=entry_doi,
                        )
                        appended = True
                if not appended:
                    _append_temperature_entry(
                        lookup,
                        formula=formula,
                        temperature=entry.get("value") or entry.get("temperature"),
                        temperature_type=str(entry.get("type", file_default)),
                        source=entry_source,
                        doi=entry_doi,
                    )

        elif path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    continue
                formula_col, type_col, temperature_columns = _detect_columns(list(reader.fieldnames))
                if formula_col is None or not temperature_columns:
                    print(f"Warning: could not detect columns in {path} (headers: {reader.fieldnames})")
                    continue
                for row in reader:
                    formula = row.get(formula_col, "").strip()
                    if not formula:
                        continue
                    row_source = str(row.get("source") or file_source)
                    row_doi = _optional_str(row.get("doi") or row.get("DOI"))
                    for temp_col, inferred_type in temperature_columns:
                        temp_str = row.get(temp_col, "").strip()
                        if not temp_str:
                            continue
                        temperature_type = inferred_type
                        if inferred_type == "Generic":
                            temperature_type = str(row.get(type_col, file_default)) if type_col else str(file_default)
                        _append_temperature_entry(
                            lookup,
                            formula=formula,
                            temperature=temp_str,
                            temperature_type=temperature_type,
                            source=row_source,
                            doi=row_doi,
                        )
        else:
            print(f"Warning: unsupported file format {path.suffix} for {path}")

    n_formulas = len(lookup)
    n_entries = sum(len(v) for v in lookup.values())
    print(f"Loaded {n_entries} Tc/TN entries for {n_formulas} unique formulas")
    return lookup


MAX_PLAUSIBLE_TC_K = 2000.0


def enrich_with_transition_temperatures(
    records: list[MaterialsProjectRecord],
    tc_lookup: dict[str, list[TransitionTemperatureObservation]],
    *,
    match_strategy: str = "formula_only",
    max_tc_k: float = MAX_PLAUSIBLE_TC_K,
) -> list[MaterialsProjectRecord]:
    """Attach formula-matched transition-temperature hints to records.

    When multiple temperature values exist for a formula, uses the median to reduce
    noise from NLP-extracted data. Values above *max_tc_k* are discarded as likely
    extraction errors (highest known bulk Tc is ~1388 K for Co alloys).
    """
    enriched = []
    matched = 0
    dropped_high = 0
    for record in records:
        if (
            record.transition_temperature_k is not None
            or record.transition_temperature_hint_k is not None
        ):
            enriched.append(record)
            continue

        key = normalize_formula(record.formula)
        observations = tc_lookup.get(key)
        if observations:
            # Filter out implausible values before taking the median
            plausible = [obs for obs in observations if obs.temperature_k <= max_tc_k]
            if not plausible:
                dropped_high += 1
                enriched.append(record)
                continue
            plausible_sorted = sorted(plausible, key=lambda obs: obs.temperature_k)
            median_idx = len(plausible_sorted) // 2
            chosen = plausible_sorted[median_idx]
            hint_source = chosen.source
            enriched.append(
                replace(
                    record,
                    transition_temperature_hint_k=chosen.temperature_k,
                    transition_temperature_hint_type=chosen.temperature_type,
                    transition_temperature_hint_match_strategy=match_strategy,
                    transition_temperature_hint_source=hint_source,
                    source_tags=_merge_source_tags(
                        record.source_tags or (record.source,),
                        (hint_source,),
                    ),
                )
            )
            matched += 1
        else:
            enriched.append(record)

    print(
        f"Enriched {matched}/{len(records)} records with transition temperatures"
        f" ({dropped_high} dropped as > {max_tc_k} K)"
    )
    return enriched


MOMENT_NOISE_FLOOR = 0.01  # μ_B/atom — below this, magnetism is numerical noise


def clean_transition_temperatures(
    records: list[MaterialsProjectRecord],
    *,
    moment_threshold: float = MOMENT_NOISE_FLOOR,
) -> list[MaterialsProjectRecord]:
    """Remove formula-only transition-temperature hints that contradict DFT data.

    Strips hint Tc/TN from:
      1. Records with ordering == "NM" (DFT says non-magnetic)
      2. Records with |moment_per_atom| < threshold (effectively zero moment)

    These are likely formula-matching errors where the Tc belongs to a different
    polymorph or a different oxidation state of the same composition.
    """
    cleaned = []
    dropped_nm = 0
    dropped_low_moment = 0

    for record in records:
        if record.transition_temperature_hint_k is None:
            cleaned.append(record)
            continue

        # Rule 1: NM ordering contradicts having a Curie/Neel temperature
        if record.ordering == "NM":
            cleaned.append(
                replace(
                    record,
                    transition_temperature_hint_k=None,
                    transition_temperature_hint_type=None,
                    transition_temperature_hint_match_strategy=None,
                    transition_temperature_hint_source=None,
                )
            )
            dropped_nm += 1
            continue

        # Rule 2: negligible moment contradicts magnetic transition
        if (
            record.moment_per_atom is not None
            and abs(record.moment_per_atom) < moment_threshold
        ):
            cleaned.append(
                replace(
                    record,
                    transition_temperature_hint_k=None,
                    transition_temperature_hint_type=None,
                    transition_temperature_hint_match_strategy=None,
                    transition_temperature_hint_source=None,
                )
            )
            dropped_low_moment += 1
            continue

        cleaned.append(record)

    total_dropped = dropped_nm + dropped_low_moment
    print(
        f"Tc cleanup: removed {total_dropped} contradictory labels "
        f"({dropped_nm} NM ordering, {dropped_low_moment} low moment < {moment_threshold} μ_B/atom)"
    )
    return cleaned


def _record_exact_structure_hash(record: MaterialsProjectRecord) -> str:
    payload = json.dumps(record.structure, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _record_supervision_count(record: MaterialsProjectRecord) -> int:
    return sum(
        int(flag)
        for flag in (
            record.energy_above_hull is not None,
            record.formation_energy_per_atom is not None,
            record.band_gap is not None,
            record.moment_per_atom is not None,
            record.ordering in MAGNETIC_ORDERING_CLASSES,
            record.transition_temperature_k is not None,
            record.transition_temperature_hint_k is not None,
            record.site_moments is not None and len(record.site_moments) == record.num_sites,
        )
    )


def _record_priority_key(record: MaterialsProjectRecord) -> tuple[int, int, str]:
    source_priority = {
        "materials_project": 3,
        "magndata": 2,
        "cif_tc": 2,
        "jarvis_dft": 1,
    }.get(record.source, 0)
    return (
        _record_supervision_count(record),
        source_priority,
        str(record.material_id),
    )


def finalize_records(
    records: list[MaterialsProjectRecord],
    *,
    moment_threshold: float = MOMENT_NOISE_FLOOR,
) -> list[MaterialsProjectRecord]:
    """Final dataset cleanup before writing.

    This step:
      1. Normalizes the small MAGNDATA sign inconsistency where
         `moment_per_atom = abs(total_magnetization) / nsites` but
         `total_magnetization` can still be negative.
      2. Drops trusted Tc/TN labels that are internally contradictory with the
         same row's magnetic ordering.
      3. Deduplicates exact repeated structures that can survive the source
         merge, especially duplicate MP material IDs pointing to the same
         crystal.
    """

    sanitized: list[MaterialsProjectRecord] = []
    fixed_moment_sign = 0
    dropped_trusted_tc_nm = 0
    dropped_trusted_tc_type_conflict = 0

    for record in records:
        updated = record

        if (
            updated.total_magnetization is not None
            and updated.moment_per_atom is not None
            and updated.num_sites > 0
        ):
            signed = updated.total_magnetization / updated.num_sites
            magnitude = abs(updated.total_magnetization) / updated.num_sites
            if (
                abs(updated.moment_per_atom - magnitude) <= 1e-8
                and abs(updated.moment_per_atom - signed) > 1e-8
            ):
                updated = replace(updated, total_magnetization=abs(updated.total_magnetization))
                fixed_moment_sign += 1

        if updated.transition_temperature_k is not None:
            tc_conflicts_nm = updated.ordering == "NM"
            tc_conflicts_type = (
                (updated.transition_temperature_type == "Neel" and updated.ordering == "FM")
                or (updated.transition_temperature_type == "Curie" and updated.ordering == "AFM")
            )
            if tc_conflicts_nm or tc_conflicts_type:
                updated = replace(
                    updated,
                    transition_temperature_k=None,
                    transition_temperature_type=None,
                    transition_temperature_match_strategy=None,
                    transition_temperature_source=None,
                )
                if tc_conflicts_nm:
                    dropped_trusted_tc_nm += 1
                if tc_conflicts_type:
                    dropped_trusted_tc_type_conflict += 1

        sanitized.append(updated)

    deduped: list[MaterialsProjectRecord] = []
    exact_index: dict[str, int] = {}
    collapsed_duplicates = 0
    for record in sanitized:
        key = _record_exact_structure_hash(record)
        existing_idx = exact_index.get(key)
        if existing_idx is None:
            exact_index[key] = len(deduped)
            deduped.append(record)
            continue

        existing = deduped[existing_idx]
        preferred, other = (
            (record, existing)
            if _record_priority_key(record) > _record_priority_key(existing)
            else (existing, record)
        )
        transition_strategy = preferred.transition_temperature_match_strategy
        if (
            transition_strategy is None
            and other.transition_temperature_k is not None
            and other.transition_temperature_match_strategy == "direct_structure"
        ):
            transition_strategy = "structure_match"
        merged = _merge_record_fields(
            preferred,
            other,
            transition_temperature_match_strategy=transition_strategy,
        )
        deduped[existing_idx] = merged
        collapsed_duplicates += 1

    print(
        "Final cleanup: "
        f"fixed {fixed_moment_sign} signed-moment rows, "
        f"dropped {dropped_trusted_tc_nm} trusted Tc rows with NM ordering, "
        f"dropped {dropped_trusted_tc_type_conflict} trusted Tc rows with Tc/TN-type conflicts, "
        f"collapsed {collapsed_duplicates} exact duplicate structures"
    )
    return deduped


# ---------------------------------------------------------------------------
# JARVIS-DFT data
# ---------------------------------------------------------------------------
#
# JARVIS-DFT 3D: ~40k materials with DFT-computed properties.
#   Download: https://figshare.com/articles/dataset/jdft_3d-12-12-2022_json/21828416
#   Or via Python: from jarvis.db.figshare import data; entries = data('dft_3d')
#
# Each entry has: jid, atoms (lattice_mat, coords, elements), formation_energy_peratom,
# optb88vdw_bandgap, magmom_oszicar, ehull, formula, spg_number, etc.


def load_jarvis_records(
    path: str | Path,
    *,
    min_sites: int = 1,
    max_sites: int = 40,
) -> list[MaterialsProjectRecord]:
    """Load JARVIS-DFT 3D data and convert to MaterialsProjectRecord format."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        entries = json.load(f)

    records: list[MaterialsProjectRecord] = []
    skipped = 0
    for entry in tqdm(entries, desc="Loading JARVIS-DFT records", unit="material"):
        atoms = entry.get("atoms")
        if atoms is None:
            skipped += 1
            continue

        elements = atoms.get("elements", [])
        num_sites = len(elements)
        if num_sites < min_sites or num_sites > max_sites:
            continue

        try:
            structure = Structure(
                lattice=atoms["lattice_mat"],
                species=elements,
                coords=atoms["coords"],
                coords_are_cartesian=atoms.get("cartesian", False),
            )
        except Exception:
            skipped += 1
            continue

        magmom = _optional_float(entry.get("magmom_oszicar"))
        moment_per_atom = None
        if magmom is not None:
            moment_per_atom = abs(float(magmom)) / max(num_sites, 1)

        formula = entry.get("formula", structure.composition.reduced_formula)

        records.append(
            MaterialsProjectRecord(
                material_id=str(entry.get("jid", f"JARVIS-{len(records)}")),
                formula=str(formula),
                num_sites=num_sites,
                total_magnetization=_optional_float(magmom),
                moment_per_atom=moment_per_atom,
                structure=structure_as_serializable_dict(structure),
                source="jarvis_dft",
                source_id=str(entry.get("jid", "")),
                energy_above_hull=_optional_float(entry.get("ehull")),
                formation_energy_per_atom=_optional_float(entry.get("formation_energy_peratom")),
                band_gap=_optional_float(entry.get("optb88vdw_bandgap")),
                source_tags=("jarvis_dft",),
            )
        )

    print(f"Loaded {len(records)} JARVIS records ({skipped} skipped)")
    return records


# ---------------------------------------------------------------------------
# CIF-based Curie/Neel datasets (NEMAD+ICSD, MAGNDATA, etc.)
# ---------------------------------------------------------------------------
#
# For datasets that provide CIF files with associated Tc/TN values,
# use this loader. Expected format: a directory of CIF files with a
# companion CSV/JSON mapping filename/material_id -> Tc/TN.


def load_cif_tc_records(
    cif_dir: str | Path,
    labels_path: str | Path,
    *,
    formula_col: str = "formula",
    temperature_col: str = "value",
    type_col: str | None = "type",
    id_col: str = "material_id",
    cif_col: str | None = None,
    min_sites: int = 1,
    max_sites: int = 40,
) -> list[MaterialsProjectRecord]:
    """Load materials from CIF files with associated transition temperature labels.

    Works with the NEMAD+ICSD aligned dataset (arXiv:2602.00756) and similar
    datasets that pair CIF structures with Tc/TN values.
    """
    from pymatgen.io.cif import CifParser

    cif_dir = Path(cif_dir)
    labels_path = Path(labels_path)

    # Load labels
    if labels_path.suffix == ".csv":
        with labels_path.open(newline="", encoding="utf-8") as f:
            labels = list(csv.DictReader(f))
    else:
        with labels_path.open(encoding="utf-8") as f:
            labels = json.load(f)

    records: list[MaterialsProjectRecord] = []
    skipped = 0
    for entry in tqdm(labels, desc="Loading CIF+Tc records", unit="material"):
        mat_id = str(entry.get(id_col, f"CIF-{len(records)}"))
        formula = str(entry.get(formula_col, ""))
        temp_str = entry.get(temperature_col)
        if temp_str is None:
            continue
        try:
            temp_k = float(temp_str)
        except (ValueError, TypeError):
            continue
        temp_type = str(entry.get(type_col, "Curie")) if type_col else "Curie"

        # Find CIF file
        cif_name = entry.get(cif_col) if cif_col else None
        if cif_name is None:
            candidates = [cif_dir / f"{mat_id}.cif", cif_dir / f"{formula}.cif"]
            cif_path = next((c for c in candidates if c.exists()), None)
        else:
            cif_path = cif_dir / str(cif_name)

        if cif_path is None or not cif_path.exists():
            skipped += 1
            continue

        try:
            parser = CifParser(str(cif_path))
            structures = parser.parse_structures(primitive=True)
            if not structures:
                skipped += 1
                continue
            structure = structures[0]
        except Exception:
            skipped += 1
            continue

        num_sites = len(structure)
        if num_sites < min_sites or num_sites > max_sites:
            continue

        records.append(
            MaterialsProjectRecord(
                material_id=mat_id,
                formula=formula or structure.composition.reduced_formula,
                num_sites=num_sites,
                total_magnetization=None,
                moment_per_atom=None,
                structure=structure_as_serializable_dict(structure),
                source="cif_tc",
                source_id=mat_id,
                transition_temperature_k=temp_k,
                transition_temperature_type=temp_type,
                transition_temperature_match_strategy="direct_structure",
                transition_temperature_source="cif_tc",
                source_tags=("cif_tc",),
            )
        )

    print(f"Loaded {len(records)} CIF+Tc records ({skipped} skipped)")
    return records


def download_magndata_records(
    download_dir: str | Path,
    *,
    seed_index: str = "1.52",
    max_entries: int | None = None,
    min_sites: int = 1,
    max_sites: int = 40,
) -> list[MaterialsProjectRecord]:
    """Download and parse trusted structure-resolved Tc/TN records from MAGNDATA."""
    from pymatgen.io.cif import CifParser

    download_dir = Path(download_dir)
    pages_dir = download_dir / "pages"
    mcif_dir = download_dir / "mcif"
    pages_dir.mkdir(parents=True, exist_ok=True)
    mcif_dir.mkdir(parents=True, exist_ok=True)

    queue: deque[str] = deque([seed_index])
    queued: set[str] = {seed_index}
    seen: set[str] = set()
    records: list[MaterialsProjectRecord] = []
    skipped = 0

    while queue and (max_entries is None or len(seen) < max_entries):
        index = queue.popleft()
        seen.add(index)

        page_path = pages_dir / f"{index}.html"
        if page_path.exists():
            html = page_path.read_text(encoding="utf-8", errors="ignore")
        else:
            try:
                html = _fetch_text(_MAGNDATA_ENTRY_URL.format(index=index))
            except Exception:
                skipped += 1
                continue
            page_path.write_text(html, encoding="utf-8")

        previous_index, next_index = _parse_magndata_navigation(html)
        for neighbor in (previous_index, next_index):
            if neighbor and neighbor not in seen and neighbor not in queued:
                queue.append(neighbor)
                queued.add(neighbor)

        entry = _parse_magndata_entry(index, html)
        if entry is None:
            skipped += 1
            continue

        mcif_path = mcif_dir / f"{index}.mcif"
        try:
            _download_file(entry.mcif_href, mcif_path, desc=f"magndata {index}")
            structures = CifParser(str(mcif_path)).parse_structures(primitive=True)
        except Exception:
            skipped += 1
            continue
        if not structures:
            skipped += 1
            continue

        structure = structures[0]
        num_sites = len(structure)
        if num_sites < min_sites or num_sites > max_sites:
            continue

        raw_magmoms = structure.site_properties.get("magmom")
        site_moments: list[float] | None = None
        total_magnetization: float | None = None
        moment_per_atom: float | None = None
        ordering: str | None = None
        transition_type = "Curie"
        if raw_magmoms is not None and len(raw_magmoms) == num_sites:
            site_moments = [float(value) for value in raw_magmoms]
            total_magnetization = float(sum(site_moments))
            moment_per_atom = abs(total_magnetization) / max(num_sites, 1)
            ordering = infer_ordering_from_site_moments(site_moments)
            if ordering == "AFM":
                transition_type = "Neel"

        records.append(
            MaterialsProjectRecord(
                material_id=f"magndata-{index}",
                formula=str(structure.composition.reduced_formula),
                num_sites=num_sites,
                total_magnetization=total_magnetization,
                moment_per_atom=moment_per_atom,
                ordering=ordering,
                structure=structure_as_serializable_dict(structure),
                source="magndata",
                source_id=index,
                transition_temperature_k=entry.temperature_k,
                transition_temperature_type=transition_type,
                transition_temperature_match_strategy="direct_structure",
                transition_temperature_source="magndata",
                site_moments=site_moments,
                source_tags=("magndata",),
            )
        )

    print(
        f"Loaded {len(records)} MAGNDATA records from seed {seed_index} "
        f"({skipped} skipped, {len(seen)} pages visited)"
    )
    return records


# ---------------------------------------------------------------------------
# Multi-source merge
# ---------------------------------------------------------------------------


def merge_records(
    primary: list[MaterialsProjectRecord],
    *secondary_sources: list[MaterialsProjectRecord],
    structure_match: bool = True,
) -> list[MaterialsProjectRecord]:
    """Merge records from multiple sources, primary taking precedence.

    Records are matched by structure, not just formula. Formula-only matching is
    too noisy for polymorph-sensitive labels like hull, band gap, ordering, and
    transition temperature.
    """
    matcher = StructureMatcher(
        primitive_cell=True,
        scale=True,
        attempt_supercell=False,
        ltol=0.1,
        stol=0.1,
        angle_tol=2.0,
    )
    bucket_index: dict[tuple[str, int, int, int, int], list[int]] = {}
    merged = list(primary)
    for i, record in enumerate(merged):
        bucket_index.setdefault(_record_bucket_signature(record), []).append(i)

    added = 0
    enriched = 0
    for source_records in secondary_sources:
        for record in source_records:
            candidate_indices: list[int] = []
            seen_indices: set[int] = set()
            for key in _record_bucket_search_keys(record):
                for idx in bucket_index.get(key, []):
                    if idx not in seen_indices:
                        seen_indices.add(idx)
                        candidate_indices.append(idx)
            match_idx: int | None = None
            for idx in candidate_indices:
                existing = merged[idx]
                if not structure_match or _records_match_by_structure(
                    existing,
                    record,
                    matcher=matcher,
                ):
                    match_idx = idx
                    break

            if match_idx is None:
                bucket_index.setdefault(_record_bucket_signature(record), []).append(len(merged))
                merged.append(record)
                added += 1
                continue

            existing = merged[match_idx]
            transition_strategy = existing.transition_temperature_match_strategy
            if (
                transition_strategy is None
                and record.transition_temperature_k is not None
                and record.transition_temperature_match_strategy == "direct_structure"
            ):
                transition_strategy = "structure_match"
            updated = _merge_record_fields(
                existing,
                record,
                transition_temperature_match_strategy=transition_strategy,
            )
            if updated != existing:
                merged[match_idx] = updated
                enriched += 1

    print(
        f"Merge: {len(primary)} primary + {added} new materials, "
        f"{enriched} enriched with missing fields = {len(merged)} total"
    )
    return merged


def write_records(
    path: str | Path,
    records: list[MaterialsProjectRecord],
) -> None:
    output_path = Path(path)
    ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json())
            handle.write("\n")


def load_records(path: str | Path) -> list[MaterialsProjectRecord]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        return [MaterialsProjectRecord.from_json(line) for line in handle if line.strip()]


# ---------------------------------------------------------------------------
# PyG datasets
# ---------------------------------------------------------------------------


def _infer_is_magnetic(record) -> bool | None:
    if getattr(record, "ordering", None) in MAGNETIC_ORDERING_CLASSES:
        return bool(record.ordering != "NM")

    site_moments = getattr(record, "site_moments", None)
    num_sites = getattr(record, "num_sites", None)
    if site_moments is not None and num_sites is not None and len(site_moments) == num_sites:
        return bool(max(abs(float(value)) for value in site_moments) >= 0.1)

    moment_per_atom = getattr(record, "moment_per_atom", None)
    if moment_per_atom is not None:
        return bool(abs(float(moment_per_atom)) >= 0.1)

    return None


class BaseCrystalDataset(InMemoryDataset, ABC):
    cache_label = ""

    def __init__(
        self,
        root: str | Path,
        *,
        raw_filename: str = "mp_magnetization.jsonl",
        graph_config: GraphConfig | None = None,
        load_processed: bool = True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ) -> None:
        self.raw_filename = raw_filename
        self.graph_config = graph_config or GraphConfig()
        self.load_processed = load_processed
        self.builder = CrystalGraphBuilder(self.graph_config)
        self.metadata: dict[str, object] = {}
        super().__init__(str(root), transform, pre_transform, pre_filter)
        if not self.load_processed:
            self.data = None
            self.slices = None
            return
        payload = torch.load(self.processed_paths[0], weights_only=False)
        self.data = payload["data"]
        self.slices = payload["slices"]
        self.metadata = payload.get("metadata", {})

    @property
    def raw_file_names(self) -> list[str]:
        return [self.raw_filename]

    @property
    def processed_file_names(self) -> list[str]:
        stem = Path(self.raw_filename).stem
        prefix = f"{stem}_{self.cache_label}" if self.cache_label else stem
        return [f"{prefix}_{self.graph_config.cache_key()}.pt"]

    def process(self) -> None:
        records = self.load_source_records(self.raw_paths[0])
        data_list = []
        accepted_records = []
        skipped: list[str] = []

        for record in tqdm(records, desc="Building crystal graphs", unit="material"):
            try:
                graph = self.build_graph_from_record(record)
            except ValueError:
                skipped.append(record.material_id)
                continue

            if self.pre_filter is not None and not self.pre_filter(graph):
                continue
            if self.pre_transform is not None:
                graph = self.pre_transform(graph)
            data_list.append(graph)
            accepted_records.append(record)

        if not data_list:
            raise RuntimeError("Graph construction produced an empty dataset.")

        data, slices = self.collate(data_list)
        metadata = {
            "num_graphs": len(data_list),
            "skipped_material_ids": skipped,
            "graph_config": asdict(self.graph_config),
        }
        metadata.update(self.extra_metadata(accepted_records))
        torch.save(
            {
                "data": data,
                "slices": slices,
                "metadata": metadata,
            },
            self.processed_paths[0],
        )

    @abstractmethod
    def load_source_records(self, path: str | Path) -> list[object]:
        raise NotImplementedError

    @abstractmethod
    def build_graph_from_record(self, record: object):
        raise NotImplementedError

    def extra_metadata(self, accepted_records: list[object]) -> dict[str, object]:
        return {}


class CrystalMagneticBenchmarkDataset(BaseCrystalDataset):
    def __init__(
        self,
        root: str | Path,
        *,
        raw_filename: str = "mp_magnetic_benchmark.jsonl",
        graph_config: GraphConfig | None = None,
        require_site_moments: bool = False,
        load_processed: bool = True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ) -> None:
        self.require_site_moments = require_site_moments
        self.cache_label = (
            "magnetic_benchmark_moment_per_atom_with_site_moments"
            if require_site_moments
            else "magnetic_benchmark_moment_per_atom"
        )
        super().__init__(
            root,
            raw_filename=raw_filename,
            graph_config=graph_config,
            load_processed=load_processed,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

    def load_source_records(self, path: str | Path) -> list[object]:
        return load_records(path)

    def build_graph_from_record(self, record: object):
        if record.energy_above_hull is None:
            raise ValueError("Missing energy_above_hull.")
        if record.total_magnetization is None:
            raise ValueError("Missing total_magnetization.")
        if record.ordering not in MAGNETIC_ORDERING_CLASSES:
            raise ValueError("Missing known magnetic ordering.")
        if self.require_site_moments:
            if record.site_moments is None or len(record.site_moments) != record.num_sites:
                raise ValueError("Missing aligned site moments.")

        graph = self.builder.build(
            record.to_structure(),
            target=record.moment_per_atom,
            material_id=record.material_id,
            formula=record.formula,
        )
        graph.y_magnetization = torch.tensor([float(record.moment_per_atom)], dtype=torch.float32)
        graph.y_total_magnetization = torch.tensor([float(record.total_magnetization)], dtype=torch.float32)
        graph.y_energy = torch.tensor([float(record.energy_above_hull)], dtype=torch.float32)
        graph.y_formation_energy = torch.tensor(
            [0.0 if record.formation_energy_per_atom is None else float(record.formation_energy_per_atom)],
            dtype=torch.float32,
        )
        graph.formation_energy_mask = torch.tensor(
            [record.formation_energy_per_atom is not None],
            dtype=torch.bool,
        )
        graph.y_band_gap = torch.tensor(
            [0.0 if record.band_gap is None else float(record.band_gap)],
            dtype=torch.float32,
        )
        graph.band_gap_mask = torch.tensor([record.band_gap is not None], dtype=torch.bool)
        graph.y_ordering = torch.tensor(
            [MAGNETIC_ORDERING_CLASSES.index(record.ordering)],
            dtype=torch.long,
        )
        graph.ordering = record.ordering
        graph.energy_above_hull = float(record.energy_above_hull)
        graph.moment_per_atom = float(record.moment_per_atom)
        graph.total_magnetization = float(record.total_magnetization)
        is_magnetic = _infer_is_magnetic(record)
        graph.y_is_magnetic = torch.tensor([0.0 if is_magnetic is None else float(is_magnetic)], dtype=torch.float32)
        graph.magnetic_mask = torch.tensor([is_magnetic is not None], dtype=torch.bool)
        if record.site_moments is not None and len(record.site_moments) == record.num_sites:
            graph.y_site_moments = torch.tensor(record.site_moments, dtype=torch.float32)
            graph.site_moment_mask = torch.ones(record.num_sites, dtype=torch.bool)
        else:
            graph.y_site_moments = torch.zeros(record.num_sites, dtype=torch.float32)
            graph.site_moment_mask = torch.zeros(record.num_sites, dtype=torch.bool)
        return graph

    def extra_metadata(self, accepted_records: list[object]) -> dict[str, object]:
        class_counts = {label: 0 for label in MAGNETIC_ORDERING_CLASSES}
        stable_count = 0
        energy_values: list[float] = []
        formation_energy_count = 0
        band_gap_count = 0
        site_moment_count = 0

        for record in accepted_records:
            class_counts[record.ordering] += 1
            if record.is_stable:
                stable_count += 1
            if record.energy_above_hull is not None:
                energy_values.append(float(record.energy_above_hull))
            if record.formation_energy_per_atom is not None:
                formation_energy_count += 1
            if record.band_gap is not None:
                band_gap_count += 1
            if record.site_moments is not None and len(record.site_moments) == record.num_sites:
                site_moment_count += 1

        payload: dict[str, object] = {
            "class_counts": class_counts,
            "class_names": list(MAGNETIC_ORDERING_CLASSES),
            "stable_fraction": stable_count / max(len(accepted_records), 1),
            "magnetization_target": "moment_per_atom",
            "site_moment_coverage": site_moment_count / max(len(accepted_records), 1),
            "formation_energy_coverage": formation_energy_count / max(len(accepted_records), 1),
            "band_gap_coverage": band_gap_count / max(len(accepted_records), 1),
        }
        if energy_values:
            payload["mean_energy_above_hull"] = sum(energy_values) / len(energy_values)
        return payload


class CrystalMaskedMagneticDataset(BaseCrystalDataset):
    cache_label = "magnetic_masked_multitask"

    def load_source_records(self, path: str | Path) -> list[object]:
        return load_records(path)

    def build_graph_from_record(self, record: object):
        if not record.has_any_label:
            raise ValueError("Record has no supervised targets.")

        graph = self.builder.build(
            record.to_structure(),
            target=0.0 if record.moment_per_atom is None else float(record.moment_per_atom),
            material_id=record.material_id,
            formula=record.formula,
        )
        graph.y_energy = torch.tensor(
            [0.0 if record.energy_above_hull is None else float(record.energy_above_hull)],
            dtype=torch.float32,
        )
        graph.energy_mask = torch.tensor([record.energy_above_hull is not None], dtype=torch.bool)
        graph.y_formation_energy = torch.tensor(
            [0.0 if record.formation_energy_per_atom is None else float(record.formation_energy_per_atom)],
            dtype=torch.float32,
        )
        graph.formation_energy_mask = torch.tensor(
            [record.formation_energy_per_atom is not None],
            dtype=torch.bool,
        )
        graph.y_band_gap = torch.tensor(
            [0.0 if record.band_gap is None else float(record.band_gap)],
            dtype=torch.float32,
        )
        graph.band_gap_mask = torch.tensor([record.band_gap is not None], dtype=torch.bool)
        # MAGNDATA net moments are zero by AFM cancellation — not comparable to
        # MP/JARVIS |total_mag|/N, so mask them out to avoid conflicting supervision.
        _magndata_afm = record.source == "magndata" and record.ordering == "AFM"
        graph.y_magnetization = torch.tensor(
            [0.0 if record.moment_per_atom is None else float(record.moment_per_atom)],
            dtype=torch.float32,
        )
        graph.magnetization_mask = torch.tensor(
            [record.moment_per_atom is not None and not _magndata_afm],
            dtype=torch.bool,
        )
        graph.y_total_magnetization = torch.tensor(
            [0.0 if record.total_magnetization is None else float(record.total_magnetization)],
            dtype=torch.float32,
        )
        has_ordering = record.ordering in MAGNETIC_ORDERING_CLASSES
        graph.y_ordering = torch.tensor(
            [0 if not has_ordering else MAGNETIC_ORDERING_CLASSES.index(record.ordering)],
            dtype=torch.long,
        )
        graph.ordering_mask = torch.tensor([has_ordering], dtype=torch.bool)
        graph.ordering = record.ordering or "Unknown"
        graph.energy_above_hull = (
            float("nan") if record.energy_above_hull is None else float(record.energy_above_hull)
        )
        graph.moment_per_atom = (
            float("nan") if record.moment_per_atom is None else float(record.moment_per_atom)
        )
        graph.total_magnetization = (
            float("nan") if record.total_magnetization is None else float(record.total_magnetization)
        )
        is_magnetic = _infer_is_magnetic(record)
        graph.y_is_magnetic = torch.tensor([0.0 if is_magnetic is None else float(is_magnetic)], dtype=torch.float32)
        graph.magnetic_mask = torch.tensor([is_magnetic is not None], dtype=torch.bool)

        graph.y_transition_temperature = torch.tensor(
            [
                0.0
                if record.transition_temperature_k is None
                else float(record.transition_temperature_k)
            ],
            dtype=torch.float32,
        )
        graph.transition_temperature_mask = torch.tensor(
            [record.transition_temperature_k is not None], dtype=torch.bool
        )

        # MAGNDATA sublattice moments use a different convention than MP/JARVIS
        _has_site_moments = (
            record.site_moments is not None
            and len(record.site_moments) == record.num_sites
            and not _magndata_afm
        )
        if _has_site_moments:
            graph.y_site_moments = torch.tensor(record.site_moments, dtype=torch.float32)
            graph.site_moment_mask = torch.ones(record.num_sites, dtype=torch.bool)
        else:
            graph.y_site_moments = torch.zeros(record.num_sites, dtype=torch.float32)
            graph.site_moment_mask = torch.zeros(record.num_sites, dtype=torch.bool)
        return graph

    def extra_metadata(self, accepted_records: list[object]) -> dict[str, object]:
        class_counts = {label: 0 for label in MAGNETIC_ORDERING_CLASSES}
        stable_count = 0
        energy_count = 0
        formation_energy_count = 0
        band_gap_count = 0
        magnetization_count = 0
        ordering_count = 0
        site_moment_count = 0
        transition_temperature_count = 0
        transition_temperature_hint_count = 0
        energy_values: list[float] = []
        tc_values: list[float] = []
        tc_hint_values: list[float] = []

        for record in accepted_records:
            if record.is_stable:
                stable_count += 1
            if record.energy_above_hull is not None:
                energy_count += 1
                energy_values.append(float(record.energy_above_hull))
            if record.formation_energy_per_atom is not None:
                formation_energy_count += 1
            if record.band_gap is not None:
                band_gap_count += 1
            if record.moment_per_atom is not None:
                magnetization_count += 1
            if record.ordering in MAGNETIC_ORDERING_CLASSES:
                ordering_count += 1
                class_counts[record.ordering] += 1
            if record.site_moments is not None and len(record.site_moments) == record.num_sites:
                site_moment_count += 1
            if record.transition_temperature_k is not None:
                transition_temperature_count += 1
                tc_values.append(float(record.transition_temperature_k))
            if getattr(record, "transition_temperature_hint_k", None) is not None:
                transition_temperature_hint_count += 1
                tc_hint_values.append(float(record.transition_temperature_hint_k))

        n = max(len(accepted_records), 1)
        payload: dict[str, object] = {
            "class_counts": class_counts,
            "class_names": list(MAGNETIC_ORDERING_CLASSES),
            "stable_fraction": stable_count / n,
            "energy_label_coverage": energy_count / n,
            "formation_energy_label_coverage": formation_energy_count / n,
            "band_gap_label_coverage": band_gap_count / n,
            "magnetization_label_coverage": magnetization_count / n,
            "ordering_label_coverage": ordering_count / n,
            "site_moment_coverage": site_moment_count / n,
            "transition_temperature_coverage": transition_temperature_count / n,
            "transition_temperature_hint_coverage": transition_temperature_hint_count / n,
            "magnetization_target": "moment_per_atom",
        }
        if energy_values:
            payload["mean_energy_above_hull"] = sum(energy_values) / len(energy_values)
        if tc_values:
            payload["mean_transition_temperature_k"] = sum(tc_values) / len(tc_values)
        if tc_hint_values:
            payload["mean_transition_temperature_hint_k"] = sum(tc_hint_values) / len(tc_hint_values)
        return payload


# ---------------------------------------------------------------------------
# CLI: download dataset
# ---------------------------------------------------------------------------


def _build_download_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a unified magnetic materials dataset from multiple sources."
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--min-sites", type=int, default=1)
    parser.add_argument("--max-sites", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)

    # Materials Project options
    mp = parser.add_argument_group("Materials Project")
    mp.add_argument(
        "--base-jsonl",
        type=Path,
        default=None,
        help="Use an existing JSONL as the base instead of downloading from MP.",
    )
    mp.add_argument("--api-key", type=str, default=None)
    mp.add_argument("--chunk-size", type=int, default=500)
    mp.add_argument("--num-chunks", type=int, default=120)
    mp.add_argument(
        "--full-mp",
        action="store_true",
        help="Download all qualifying MP summary records instead of stopping after num_chunks.",
    )
    mp.add_argument("--max-energy-above-hull", type=float, default=None)
    mp.add_argument("--exclude-theoretical", action="store_true")
    mp.add_argument("--include-site-moments", action=argparse.BooleanOptionalAction, default=True)
    mp.add_argument("--magnetism-chunk-size", type=int, default=1000)

    # External Curie/Neel temperature data
    tc = parser.add_argument_group("Curie/Neel temperature enrichment")
    tc.add_argument(
        "--curie-csv",
        type=Path,
        nargs="*",
        default=None,
        help="CSV/JSON files with Curie temperature data (auto-detected columns).",
    )
    tc.add_argument(
        "--neel-csv",
        type=Path,
        nargs="*",
        default=None,
        help="CSV/JSON files with Neel temperature data (auto-detected columns).",
    )
    tc.add_argument(
        "--magnetic-csv",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "General magnetic-property CSV/JSON files with formula/material_name and "
            "Curie/Neel columns (for example NEMAD exports)."
        ),
    )
    tc.add_argument(
        "--allow-formula-tc-enrichment",
        action="store_true",
        help=(
            "Attach formula-matched Tc/TN as hint metadata. Disabled by default because "
            "polymorph mismatch makes these values noisy."
        ),
    )
    tc.add_argument(
        "--download-nemad-github",
        action="store_true",
        help=(
            "Download the published NEMAD GitHub FM/AFM transition-temperature CSVs "
            "and include them in the formula-matched Tc/TN hint overlay."
        ),
    )
    tc.add_argument(
        "--nemad-download-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory where downloaded NEMAD GitHub CSVs should be stored.",
    )

    # JARVIS-DFT
    jarvis = parser.add_argument_group("JARVIS-DFT")
    jarvis.add_argument(
        "--jarvis-json",
        type=Path,
        default=None,
        help="Path to JARVIS-DFT 3D JSON file (jdft_3d.json).",
    )
    jarvis.add_argument(
        "--download-jarvis",
        action="store_true",
        help="Download the latest public JARVIS jdft_3d JSON from Figshare.",
    )
    jarvis.add_argument(
        "--jarvis-download-dir",
        type=Path,
        default=Path("data/raw/sources"),
        help="Directory where downloaded JARVIS files should be stored.",
    )

    # CIF + Tc datasets (NEMAD+ICSD, MAGNDATA, etc.)
    cif = parser.add_argument_group("CIF + Tc datasets")
    cif.add_argument(
        "--cif-tc-dir",
        type=Path,
        default=None,
        help="Directory of CIF files with associated Tc/TN labels.",
    )
    cif.add_argument(
        "--cif-tc-labels",
        type=Path,
        default=None,
        help="CSV/JSON file mapping CIF filenames to Tc/TN values.",
    )
    cif.add_argument(
        "--download-magndata",
        action="store_true",
        help="Download and ingest trusted structure-resolved Tc/TN records from MAGNDATA.",
    )
    cif.add_argument(
        "--magndata-dir",
        type=Path,
        default=Path("data/raw/sources/magndata"),
        help="Directory used to cache downloaded MAGNDATA entry pages and mcif files.",
    )
    cif.add_argument(
        "--magndata-seed-index",
        type=str,
        default="1.52",
        help="MAGNDATA entry index used as the crawl seed.",
    )
    cif.add_argument(
        "--magndata-max-entries",
        type=int,
        default=250,
        help="Maximum number of MAGNDATA entries to crawl in one run.",
    )

    parser.add_argument(
        "--download-all-public-sources",
        action="store_true",
        help=(
            "Download the supported public sources automatically: JARVIS, NEMAD GitHub "
            "tables, and MAGNDATA. Formula-based Tc/TN hint enrichment is enabled too."
        ),
    )

    return parser


def download_dataset() -> None:
    args = _build_download_parser().parse_args()

    output = args.output or Path("data/raw/magnetic_unified.jsonl")
    if args.download_all_public_sources:
        args.download_jarvis = True
        args.download_nemad_github = True
        args.download_magndata = True
        args.allow_formula_tc_enrichment = True
    mp_num_chunks = None if args.full_mp else args.num_chunks

    # 1. Load or download Materials Project (backbone)
    if args.base_jsonl is not None:
        print("=" * 60)
        print(f"Step 1: Loading base records from {args.base_jsonl}...")
        print("=" * 60)
        records = load_records(args.base_jsonl)
        print(f"  -> {len(records)} records from {args.base_jsonl}")
    else:
        print("=" * 60)
        print("Step 1: Downloading from Materials Project...")
        print("=" * 60)
        records = download_materials_project_masked_records(
            api_key=args.api_key,
            chunk_size=args.chunk_size,
            num_chunks=mp_num_chunks,
            min_sites=args.min_sites,
            max_sites=args.max_sites,
            max_energy_above_hull=args.max_energy_above_hull,
            exclude_theoretical=args.exclude_theoretical,
            include_site_moments=args.include_site_moments,
            magnetism_chunk_size=args.magnetism_chunk_size,
        )
        print(f"  -> {len(records)} records from Materials Project")

    # 2. Merge JARVIS-DFT records (adds materials not in MP)
    jarvis_json_path = args.jarvis_json
    if args.download_jarvis:
        print("=" * 60)
        print("Step 2a: Downloading JARVIS-DFT...")
        print("=" * 60)
        jarvis_json_path = _download_and_extract_jarvis(args.jarvis_download_dir)

    if jarvis_json_path is not None:
        print("=" * 60)
        print("Step 2: Loading JARVIS-DFT data...")
        print("=" * 60)
        jarvis_records = load_jarvis_records(
            jarvis_json_path,
            min_sites=args.min_sites,
            max_sites=args.max_sites,
        )
        records = merge_records(records, jarvis_records)

    # 3. Merge CIF + Tc datasets (NEMAD+ICSD, etc.)
    if args.cif_tc_dir is not None and args.cif_tc_labels is not None:
        print("=" * 60)
        print("Step 3: Loading CIF + Tc dataset...")
        print("=" * 60)
        cif_records = load_cif_tc_records(
            args.cif_tc_dir,
            args.cif_tc_labels,
            min_sites=args.min_sites,
            max_sites=args.max_sites,
        )
        records = merge_records(records, cif_records)

    if args.download_magndata:
        print("=" * 60)
        print("Step 3b: Downloading MAGNDATA records...")
        print("=" * 60)
        magndata_records = download_magndata_records(
            args.magndata_dir,
            seed_index=args.magndata_seed_index,
            max_entries=args.magndata_max_entries,
            min_sites=args.min_sites,
            max_sites=args.max_sites,
        )
        records = merge_records(records, magndata_records)

    # 4. Enrich with Curie/Neel temperature data (formula matching)
    tc_files: list[Path] = []
    if args.curie_csv:
        tc_files.extend(args.curie_csv)
    if args.neel_csv:
        tc_files.extend(args.neel_csv)
    if args.magnetic_csv:
        tc_files.extend(args.magnetic_csv)
    if args.download_nemad_github:
        print("=" * 60)
        print("Step 4a: Downloading published NEMAD GitHub datasets...")
        print("=" * 60)
        for name, url in _NEMAD_GITHUB_URLS.items():
            dest = args.nemad_download_dir / f"{name}.csv"
            tc_files.append(_download_file(url, dest, desc=name))

    if tc_files:
        print("=" * 60)
        print("Step 4: Enriching with Curie/Neel temperature data...")
        print("=" * 60)
        tc_lookup = load_curie_neel_lookup(*tc_files)
        if args.allow_formula_tc_enrichment:
            records = enrich_with_transition_temperatures(
                records,
                tc_lookup,
                match_strategy="formula_only",
            )
        else:
            print(
                "Skipping formula-only Tc/TN enrichment by default. "
                "Use --allow-formula-tc-enrichment to enable this noisier overlay."
            )

    # 5. Clean contradictory Tc labels (NM materials, zero-moment materials)
    records = clean_transition_temperatures(records)
    records = finalize_records(records)

    # 6. Filter: keep only records with at least one supervised label
    records = [r for r in records if r.has_any_label]
    print(f"\nFinal dataset: {len(records)} records with at least one label")

    # Write output
    write_records(output, records)

    summary = summarize_materials_dataset(records).to_dict()
    manifest_path = args.manifest or output.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nSaved {len(records)} records to {output}")
    print(f"Saved manifest to {manifest_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    download_dataset()
