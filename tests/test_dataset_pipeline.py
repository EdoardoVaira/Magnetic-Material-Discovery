from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

pytest.importorskip("pymatgen")
pytest.importorskip("torch_geometric")

from pymatgen.core import Lattice, Structure

from dataset import (
    CrystalGraphBuilder,
    CrystalMaskedMagneticDataset,
    GraphConfig,
    _parse_magndata_entry,
    finalize_records,
    MaterialsProjectRecord,
    TransitionTemperatureObservation,
    clean_transition_temperatures,
    enrich_with_transition_temperatures,
    infer_ordering_from_site_moments,
    load_curie_neel_lookup,
    merge_records,
    normalize_formula,
)


def _make_record(
    material_id: str,
    structure: Structure,
    *,
    source: str = "materials_project",
    source_id: str | None = None,
    total_magnetization: float | None = None,
    moment_per_atom: float | None = None,
    energy_above_hull: float | None = None,
    formation_energy_per_atom: float | None = None,
    band_gap: float | None = None,
    ordering: str | None = None,
    transition_temperature_k: float | None = None,
    transition_temperature_type: str | None = None,
    transition_temperature_match_strategy: str | None = None,
    transition_temperature_source: str | None = None,
    transition_temperature_hint_k: float | None = None,
    transition_temperature_hint_type: str | None = None,
    transition_temperature_hint_match_strategy: str | None = None,
    transition_temperature_hint_source: str | None = None,
) -> MaterialsProjectRecord:
    return MaterialsProjectRecord(
        material_id=material_id,
        formula=structure.composition.reduced_formula,
        num_sites=len(structure),
        total_magnetization=total_magnetization,
        moment_per_atom=moment_per_atom,
        structure=structure.as_dict(),
        source=source,
        source_id=source_id or material_id,
        ordering=ordering,
        energy_above_hull=energy_above_hull,
        formation_energy_per_atom=formation_energy_per_atom,
        band_gap=band_gap,
        transition_temperature_k=transition_temperature_k,
        transition_temperature_type=transition_temperature_type,
        transition_temperature_match_strategy=transition_temperature_match_strategy,
        transition_temperature_source=transition_temperature_source,
        transition_temperature_hint_k=transition_temperature_hint_k,
        transition_temperature_hint_type=transition_temperature_hint_type,
        transition_temperature_hint_match_strategy=transition_temperature_hint_match_strategy,
        transition_temperature_hint_source=transition_temperature_hint_source,
        source_tags=(source,),
    )


def test_merge_records_keeps_distinct_structures_with_same_formula() -> None:
    rocksalt_like = Structure(
        Lattice.cubic(3.0),
        ["Li", "Na"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    different_phase = Structure(
        Lattice.cubic(3.0),
        ["Li", "Na"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )

    primary = [_make_record("mp-1", rocksalt_like, energy_above_hull=0.01)]
    secondary = [_make_record("jv-1", different_phase, source="jarvis_dft", energy_above_hull=0.02)]

    merged = merge_records(primary, secondary)

    assert len(merged) == 2


def test_merge_records_enriches_same_structure_missing_fields() -> None:
    structure = Structure(
        Lattice.cubic(3.0),
        ["Fe", "O"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    primary = [
        _make_record(
            "mp-1",
            structure,
            energy_above_hull=0.03,
        )
    ]
    secondary = [
        _make_record(
            "jv-1",
            structure,
            source="jarvis_dft",
            total_magnetization=4.0,
            moment_per_atom=2.0,
            formation_energy_per_atom=-1.2,
            band_gap=0.8,
        )
    ]

    merged = merge_records(primary, secondary)

    assert len(merged) == 1
    record = merged[0]
    assert record.total_magnetization == pytest.approx(4.0)
    assert record.moment_per_atom == pytest.approx(2.0)
    assert record.formation_energy_per_atom == pytest.approx(-1.2)
    assert record.band_gap == pytest.approx(0.8)
    assert record.source_tags == ("materials_project", "jarvis_dft")


def test_load_curie_neel_lookup_reads_both_temperature_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "temps.csv"
    csv_path.write_text(
        "formula,curie_temp,neel_temp\nFe,1043,\nFeO,,198\n",
        encoding="utf-8",
    )

    lookup = load_curie_neel_lookup(csv_path)

    assert lookup["Fe"] == [
        TransitionTemperatureObservation(
            temperature_k=1043.0,
            temperature_type="Curie",
            source="temps",
            doi=None,
        )
    ]
    assert lookup["FeO"] == [
        TransitionTemperatureObservation(
            temperature_k=198.0,
            temperature_type="Neel",
            source="temps",
            doi=None,
        )
    ]


def test_enrich_with_transition_temperatures_marks_formula_only_matches() -> None:
    structure = Structure(
        Lattice.cubic(3.0),
        ["Fe"],
        [[0.0, 0.0, 0.0]],
    )
    records = [_make_record("mp-1", structure)]

    enriched = enrich_with_transition_temperatures(
        records,
        {
            "Fe": [
                TransitionTemperatureObservation(1000.0, "Curie", "nemad"),
                TransitionTemperatureObservation(1043.0, "Curie", "nemad"),
                TransitionTemperatureObservation(1100.0, "Curie", "court_cole"),
            ]
        },
    )

    assert enriched[0].transition_temperature_k is None
    assert enriched[0].transition_temperature_hint_k == pytest.approx(1043.0)
    assert enriched[0].transition_temperature_hint_type == "Curie"
    assert enriched[0].transition_temperature_hint_match_strategy == "formula_only"
    assert enriched[0].transition_temperature_hint_source == "nemad"
    assert enriched[0].source_tags == ("materials_project", "nemad")


def test_load_curie_neel_lookup_reads_nemad_style_combined_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "nemad_magnetic.csv"
    csv_path.write_text(
        "Material_Name,Curie,Neel,DOI,source\nFe3O4,858.15,,10.1/example,nemad\nNd0.5Sr0.5MnO3,,50,10.2/example,nemad\n",
        encoding="utf-8",
    )

    lookup = load_curie_neel_lookup(csv_path)

    assert lookup["Fe3O4"][0].temperature_type == "Curie"
    assert lookup["Fe3O4"][0].source == "nemad"
    assert lookup[normalize_formula("Nd0.5Sr0.5MnO3")][0].temperature_type == "Neel"


def test_load_curie_neel_lookup_reads_nemad_github_exports(tmp_path: Path) -> None:
    curie_csv = tmp_path / "FM_with_curie.csv"
    curie_csv.write_text(
        "Normalized_Composition,Mean_TC_K\nFe3O4,858.15\n",
        encoding="utf-8",
    )
    neel_csv = tmp_path / "AFM_with_Neel.csv"
    neel_csv.write_text(
        "Normalized_Composition,Example_Formula,Clean_Chemical_Formula,Mean_TN_K\n"
        "Ag0.2Ga0.8Mn3.0N1.0,N1Mn3.0Ga0.8Ag0.2,Ag0.2Ga0.8Mn3.0N1.0,276.0\n",
        encoding="utf-8",
    )

    lookup = load_curie_neel_lookup(curie_csv, neel_csv)

    assert lookup["Fe3O4"][0].temperature_k == pytest.approx(858.15)
    assert lookup["Fe3O4"][0].temperature_type == "Curie"
    assert lookup[normalize_formula("Ag0.2Ga0.8Mn3.0N1.0")][0].temperature_k == pytest.approx(276.0)
    assert lookup[normalize_formula("Ag0.2Ga0.8Mn3.0N1.0")][0].temperature_type == "Neel"


def test_clean_transition_temperatures_only_strips_hints() -> None:
    structure = Structure(
        Lattice.cubic(3.0),
        ["Fe"],
        [[0.0, 0.0, 0.0]],
    )
    records = [
        _make_record(
            "mp-1",
            structure,
            ordering="NM",
            transition_temperature_hint_k=1043.0,
            transition_temperature_hint_type="Curie",
            transition_temperature_hint_match_strategy="formula_only",
            transition_temperature_hint_source="nemad",
        ),
        _make_record(
            "exp-1",
            structure,
            source="cif_tc",
            ordering="NM",
            transition_temperature_k=643.0,
            transition_temperature_type="Neel",
            transition_temperature_match_strategy="direct_structure",
            transition_temperature_source="magndata",
        ),
    ]

    cleaned = clean_transition_temperatures(records)

    assert cleaned[0].transition_temperature_hint_k is None
    assert cleaned[1].transition_temperature_k == pytest.approx(643.0)


def test_finalize_records_collapses_exact_duplicates_and_normalizes_moment_sign() -> None:
    structure = Structure(
        Lattice.cubic(3.0),
        ["Fe", "O"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    records = [
        _make_record(
            "mp-1",
            structure,
            source="materials_project",
            energy_above_hull=0.01,
        ),
        _make_record(
            "mp-2",
            structure,
            source="materials_project",
            total_magnetization=2.0,
            moment_per_atom=1.0,
        ),
        _make_record(
            "mag-1",
            structure,
            source="magndata",
            total_magnetization=-2.0,
            moment_per_atom=1.0,
            transition_temperature_k=300.0,
            transition_temperature_type="Curie",
            transition_temperature_match_strategy="direct_structure",
            transition_temperature_source="magndata",
        ),
    ]

    finalized = finalize_records(records)

    assert len(finalized) == 1
    record = finalized[0]
    assert record.energy_above_hull == pytest.approx(0.01)
    assert record.total_magnetization == pytest.approx(2.0)
    assert record.moment_per_atom == pytest.approx(1.0)
    assert set(record.source_tags) == {"materials_project", "magndata"}
    assert record.transition_temperature_k == pytest.approx(300.0)


def test_finalize_records_drops_contradictory_trusted_tc_rows() -> None:
    structure = Structure(
        Lattice.cubic(3.0),
        ["Fe"],
        [[0.0, 0.0, 0.0]],
    )
    other_structure = Structure(
        Lattice.cubic(3.2),
        ["Fe"],
        [[0.0, 0.0, 0.0]],
    )
    records = [
        _make_record(
            "nm-1",
            structure,
            ordering="NM",
            transition_temperature_k=81.0,
            transition_temperature_type="Curie",
            transition_temperature_match_strategy="structure_match",
            transition_temperature_source="magndata",
        ),
        _make_record(
            "fm-neel",
            other_structure,
            ordering="FM",
            transition_temperature_k=120.0,
            transition_temperature_type="Neel",
            transition_temperature_match_strategy="structure_match",
            transition_temperature_source="magndata",
        ),
    ]

    finalized = finalize_records(records)

    assert finalized[0].transition_temperature_k is None
    assert finalized[1].transition_temperature_k is None


def test_parse_magndata_entry_extracts_temperature_mcif_and_navigation() -> None:
    html = """
    <html><body>
    <form method=GET action="index.php"><input type=hidden name=index value=1.51><input type=submit name=submit value="Previous entry"></form>
    <form method=GET action="index.php"><input type=hidden name=index value=1.53><input type=submit name=submit value="Next entry"></form>
    <b>Transition Temperature: </b>173 K<br>
    <a href="tmp/1.52_CaFe2As2.mcif">Download mcif file</a>
    </body></html>
    """
    entry = _parse_magndata_entry("1.52", html)
    assert entry is not None
    assert entry.index == "1.52"
    assert entry.temperature_k == pytest.approx(173.0)
    assert entry.previous_index == "1.51"
    assert entry.next_index == "1.53"
    assert entry.mcif_href.endswith("tmp/1.52_CaFe2As2.mcif")


def test_infer_ordering_from_site_moments_handles_basic_cases() -> None:
    assert infer_ordering_from_site_moments([0.0, 0.0]) == "NM"
    assert infer_ordering_from_site_moments([2.0, 2.1, 1.9]) == "FM"
    assert infer_ordering_from_site_moments([2.0, -2.0]) == "AFM"
    assert infer_ordering_from_site_moments([3.0, -1.0]) == "FiM"


def test_masked_dataset_collates_unknown_ordering_rows(tmp_path: Path) -> None:
    structure = Structure(
        Lattice.cubic(3.0),
        ["Fe", "O"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    records = [
        _make_record(
            "mp-known",
            structure,
            total_magnetization=4.0,
            moment_per_atom=2.0,
            energy_above_hull=0.01,
            ordering="FM",
        ),
        _make_record(
            "mp-unknown",
            structure.copy(),
            energy_above_hull=0.02,
            ordering=None,
        ),
    ]

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "subset.jsonl"
    raw_path.write_text("".join(json.dumps(asdict(record)) + "\n" for record in records), encoding="utf-8")

    dataset = CrystalMaskedMagneticDataset(
        root=tmp_path,
        raw_filename="subset.jsonl",
        graph_config=GraphConfig(cutoff=5.0, num_radial=8, radial_width=0.5),
    )

    assert len(dataset) == 2
    assert dataset[1].ordering == "Unknown"
    assert bool(dataset[1].ordering_mask.item()) is False


def test_crystal_graph_builder_handles_disordered_sites() -> None:
    structure = Structure(
        Lattice.cubic(3.5),
        [{"Fe": 0.75, "Co": 0.25}, "O"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )

    builder = CrystalGraphBuilder(GraphConfig(cutoff=5.0, num_radial=8, radial_width=0.5))
    graph = builder.build(structure, target=0.0)

    assert graph.z.tolist() == [26, 8]
    assert graph.num_nodes == 2
