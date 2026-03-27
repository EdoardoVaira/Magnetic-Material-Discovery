# Magnetic Material Discovery

End-to-end magnetic materials discovery from crystal structures:

- build a unified magnetic dataset
- train a multitask crystal GNN
- screen large candidate sets such as GNoME
- validate the best hits with DFT

This repo is the working codebase behind that pipeline. It is focused on magnetic ordering, magnetic moment, stability, and follow-up anisotropy checks for rare-earth-free permanent-magnet candidates.

![Model pipeline overview](image.png)

The diagram above shows the core screening loop in this repo: crystal structure in, multitask magnetic and stability predictions out.

## What This Project Does

The project combines fast ML screening with slower first-principles validation:

1. merge public crystal and magnetic-property sources into one masked multitask dataset
2. train a periodic graph neural network on magnetic and stability targets
3. run the trained model over large candidate structure sets
4. take the best candidates into DFT for physical validation

The target use case is simple:

- use ML to search huge structure spaces quickly
- use DFT only on a small shortlist
- identify materials that look like real magnetic or permanent-magnet candidates

## Current Highlight

The main validated example in this repo right now is `Fe6Co2Ge`.

Current first-pass story:

- the GNN predicts `Fe6Co2Ge` as a strong `FM` candidate
- simple DFT confirms `FM` is clearly favored over `NM`
- a first-pass SOC force-theorem calculation gives a large positive anisotropy with easy axis along `[001]`

The current frozen result note is:

- [`validation/fe6co2ge_mae_qe/Fe6Co2Ge_result_summary_2026-03-27.md`](validation/fe6co2ge_mae_qe/Fe6Co2Ge_result_summary_2026-03-27.md)

This should be treated as an exciting candidate result, not as a final publication-grade claim without tighter verification.

## Repository Layout

Core project files:

- [`dataset.py`](dataset.py)
  Unified dataset builder and dataset definitions.
- [`model.py`](model.py)
  Multitask periodic GNN for scalar and magnetic outputs.
- [`train.py`](train.py)
  Main training entrypoint.
- [`screen.py`](screen.py)
  Screening utilities for running trained checkpoints on candidate crystal sets.

Supporting project areas:

- [`scripts/cluster/`](scripts/cluster/)
  Slurm helpers and QE cluster utilities.
- [`tests/`](tests/)
  Unit tests for dataset, graph building, model, training, and screening utilities.
- [`validation/`](validation/)
  DFT inputs, validation notes, and candidate-specific follow-up runs.

## Main Targets

The current dataset / model path supports these crystal-level targets:

- `energy_above_hull`
- `formation_energy_per_atom`
- `band_gap`
- `moment_per_atom`
- `ordering` as `NM / FM / FiM / AFM`
- `site_moments`

Temperature handling is more conservative:

- `transition_temperature_k`
  Only for trusted structure-matched labels.
- `transition_temperature_hint_k`
  Formula-level hint metadata, not default ground truth.

## Data Sources

The unified dataset pipeline can merge or download from:

- Materials Project
- JARVIS
- NEMAD
- MAGNDATA
- structure-resolved CIF plus `Tc/TN` labels when available

It also supports:

- structure-aware deduplication across sources
- formula-level `Tc/TN` hint overlays
- loading existing JSONL dumps instead of redownloading everything

## Install

Install `torch` and `torch-geometric` in the way your machine requires, then:

```bash
pip install -e ".[dev]"
```

If you want to download Materials Project data:

```bash
export MP_API_KEY=...
```

## Quick Start

### 1. Build the unified dataset

```bash
python3 dataset.py \
  --output data/raw/magnetic_unified.jsonl \
  --num-chunks 120 \
  --chunk-size 500 \
  --max-sites 40
```

A fuller public build can look like:

```bash
python3 dataset.py \
  --output data/raw/magnetic_unified_full.jsonl \
  --manifest data/raw/magnetic_unified_full.manifest.json \
  --full-mp \
  --max-sites 40 \
  --download-all-public-sources \
  --jarvis-download-dir data/raw/sources \
  --nemad-download-dir data/raw/sources \
  --magndata-dir data/raw/sources/magndata \
  --magndata-max-entries 1000 \
  --curie-csv data/raw/sources/court_cole_curie.csv data/raw/sources/nemad_curie.csv \
  --neel-csv data/raw/sources/court_cole_neel.csv data/raw/sources/nemad_neel.csv \
  --magnetic-csv data/raw/sources/nemad_magnetic.csv
```

### 2. Train the model

```bash
python3 train.py \
  --dataset-root data \
  --raw-filename magnetic_unified.jsonl \
  --output-dir runs/magnetic_model
```

The trainer uses masked multitask losses, so each record can contribute only to the targets it actually has.

### 3. Screen candidate structures

```bash
python3 screen.py \
  --run-dir runs/magnetic_model \
  --summary-csv data/gnome/stable_materials_summary.csv \
  --cif-zip data/gnome/by_id.zip \
  --output-dir runs/gnome_screen
```

The screening outputs can include:

- hull / formation energy predictions
- band gap
- magnetic moment
- site moments
- ordering probabilities
- transition temperature predictions if that head is present

## Cluster / DFT Workflow

Relevant cluster helpers:

- [`scripts/cluster/train_masked_multitask_mp.slurm`](scripts/cluster/train_masked_multitask_mp.slurm)
- [`scripts/cluster/screen_gnome_masked.slurm`](scripts/cluster/screen_gnome_masked.slurm)
- [`scripts/cluster/qe_pw.slurm`](scripts/cluster/qe_pw.slurm)
- [`scripts/cluster/qe_prepare_force_theorem_state.sh`](scripts/cluster/qe_prepare_force_theorem_state.sh)
- [`scripts/cluster/qe_clone_restart_state.sh`](scripts/cluster/qe_clone_restart_state.sh)

The DFT validation history and candidate-specific inputs are kept under [`validation/`](validation/).

Useful notes:

- [`validation/DFT_validation_summary.md`](validation/DFT_validation_summary.md)
- [`validation/fe6co2ge_mae_qe/Fe6Co2Ge_result_summary_2026-03-27.md`](validation/fe6co2ge_mae_qe/Fe6Co2Ge_result_summary_2026-03-27.md)

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

## What Is Not In Git

Large generated artifacts are intentionally ignored:

- `data/raw/`
- `data/processed/`
- `runs/`
- QE `tmp/` and backup directories under `validation/`

This keeps the repo small and suitable for a private code snapshot while leaving heavyweight data and cluster artifacts outside git.

## Project Status

This is an active research codebase, not a polished library release.

Current direction:

- keep one unified dataset path
- keep one main model/training path
- use screening to produce high-value magnetic candidates
- use DFT only for the shortlist

The immediate scientific goal is to turn the ML + DFT pipeline into a repeatable engine for identifying strong rare-earth-free magnetic materials, with `Fe6Co2Ge` as the current lead example.
