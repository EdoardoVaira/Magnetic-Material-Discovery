from __future__ import annotations

from screen import GNoMEPrediction, _build_shortlists, _is_rare_earth_free


def _prediction(
    material_id: str,
    formula: str,
    elements: tuple[str, ...],
    *,
    score: float,
    energy: float = 0.01,
    gap: float = 0.0,
    moment: float = 2.0,
    tc: float | None = 350.0,
    ordering: str = "FM",
) -> GNoMEPrediction:
    probs = {"NM": 0.01, "FM": 0.96, "FiM": 0.02, "AFM": 0.01}
    if ordering == "FiM":
        probs = {"NM": 0.01, "FM": 0.15, "FiM": 0.82, "AFM": 0.02}
    return GNoMEPrediction(
        material_id=material_id,
        formula=formula,
        composition=formula,
        elements=elements,
        num_sites=4,
        crystal_system="cubic",
        space_group="Fm-3m",
        predicted_energy_above_hull=energy,
        predicted_formation_energy_per_atom=-0.2,
        predicted_band_gap=gap,
        predicted_moment_per_atom=moment,
        predicted_transition_temperature=tc,
        predicted_site_moments=(2.5, 1.5, 0.5, 0.1),
        predicted_site_moment_mean=1.15,
        predicted_site_moment_abs_mean=1.15,
        predicted_ordering=ordering,
        ordering_probs=probs,
        score=score,
        predicted_moment_from_sites=moment,
        predicted_is_magnetic_probability=0.99,
    )


def test_shortlists_include_rare_earth_free_partition() -> None:
    predictions = [
        _prediction("a", "GdFe2", ("Fe", "Gd"), score=10.0),
        _prediction("b", "FeCoNi", ("Co", "Fe", "Ni"), score=9.0),
        _prediction("c", "MnAl", ("Al", "Mn"), score=8.0),
    ]

    shortlists = _build_shortlists(predictions, shortlist_size=2)

    assert [item.material_id for item in shortlists["overall_top30"]] == ["a", "b"]
    assert [item.material_id for item in shortlists["rare_earth_free_top30"]] == ["b", "c"]
    assert not _is_rare_earth_free(predictions[0])
    assert _is_rare_earth_free(predictions[1])


def test_shortlists_include_lightweight_and_low_cost_views() -> None:
    predictions = [
        _prediction("a", "GdFe2", ("Fe", "Gd"), score=10.0),
        _prediction("b", "Fe2Co", ("Co", "Fe"), score=9.0),
        _prediction("c", "MnAl", ("Al", "Mn"), score=8.0),
    ]

    shortlists = _build_shortlists(predictions, shortlist_size=2)

    assert [item.material_id for item in shortlists["lightweight_top30"]] == ["c", "b"]
    assert [item.material_id for item in shortlists["low_cost_top30"]] == ["c", "b"]
