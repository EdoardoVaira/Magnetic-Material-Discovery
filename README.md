# Magnets

One clean path for magnetic materials screening from crystal structures.

The repo is now organized around exactly four things:

1. one dataset builder
2. one model
3. one training script
4. a small set of screening / cluster utilities

## What The Dataset Supports

These are the core crystal-level targets the current dataset supports cleanly:

- `energy_above_hull`
- `formation_energy_per_atom`
- `band_gap`
- `moment_per_atom`
- `ordering` as `NM / FM / FiM / AFM`
- `site_moments`

Transition temperatures are handled more carefully:

- `transition_temperature_k`
  - only for trusted structure-matched labels
  - currently supported from MAGNDATA and any CIF/mcif dataset you provide
- `transition_temperature_hint_k`
  - formula-matched experimental hint metadata
  - not meant to be treated as ground-truth supervision

## Minimal Repo Layout

- [dataset.py](/Users/edo/Edo/Magnets/dataset.py)
  The single dataset builder and dataset definitions.
- [model.py](/Users/edo/Edo/Magnets/model.py)
  The single scalar/vector site-first multitask model architecture.
- [train.py](/Users/edo/Edo/Magnets/train.py)
  The single training entrypoint.
- [screen.py](/Users/edo/Edo/Magnets/screen.py)
  Screening utilities.
- [scripts/cluster](/Users/edo/Edo/Magnets/scripts/cluster)
  Slurm helpers.

Everything else is support code for data loading, graph building, metrics, and visualization.

## Install

Install `torch` and `torch-geometric` in the way your machine requires, then:

```bash
pip install -e ".[dev]"
```

For Materials Project downloads:

```bash
export MP_API_KEY=...
```

## 1. Build The Unified Dataset

Use the single dataset script:

```bash
python3 dataset.py \
  --output data/raw/magnetic_unified.jsonl \
  --num-chunks 120 \
  --chunk-size 500 \
  --max-sites 40
```

For the full public build:

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

What it can do:
- download MP records directly
- download the public JARVIS dump automatically
- download the public NEMAD GitHub CSVs automatically
- crawl MAGNDATA for trusted structure-resolved `Tc/TN` data
- load an existing MP JSONL instead of downloading
- normalize raw JARVIS JSON exports
- merge structure-matched CIF + `Tc/TN` datasets when available
- overlay sparse formula-matched experimental `Tc/TN` as hint metadata
- deduplicate cross-source records by structure-aware matching

Useful inputs:
- `--base-jsonl path/to/base.jsonl`
- `--jarvis-json path/to/jarvis_dump.json`
- `--download-jarvis`
- `--jarvis-download-dir data/raw/sources`
- `--curie-csv path/to/curie.csv`
- `--neel-csv path/to/neel.csv`
- `--magnetic-csv path/to/nemad_magnetic.csv`
- `--download-nemad-github`
- `--nemad-download-dir data/raw/sources`
- `--cif-tc-dir path/to/cifs --cif-tc-labels path/to/labels.csv`
- `--download-magndata --magndata-max-entries 250`
- `--download-all-public-sources`
- `--full-mp`
- `--allow-formula-tc-enrichment`

The output is one masked multitask JSONL plus a manifest:
- dataset: `data/raw/magnetic_unified.jsonl`
- manifest: `data/raw/magnetic_unified.manifest.json`

## 2. Train The Model

Use the single training script:

```bash
python3 train.py \
  --dataset-root data \
  --raw-filename magnetic_unified.jsonl \
  --output-dir runs/magnetic_model
```

The trainer uses masked multitask losses, so a record does not need every label.

Main loss heads:
- `energy_above_hull`
- `formation_energy_per_atom`
- `band_gap`
- `moment_per_atom`
- `site_moments`
- `transition_temperature_k` when trusted structure-level labels are present
- `ordering`

Only add `transition_temperature_k` as a supervised head once you have
structure-matched experimental labels. Formula-only `transition_temperature_hint_k`
should be treated as metadata, not default ground truth.

Useful options:
- `--holdout-material-ids-json` for leak-free evaluation
- `--balanced-sampler` to emphasize the rare ordering classes
- `--formation-energy-loss-weight`
- `--band-gap-loss-weight`
- `--transition-temperature-loss-weight`
- `--magnetic-loss-weight`
- `--moment-consistency-loss-weight`

## 3. Screen GNoME

```bash
python3 screen.py \
  --run-dir runs/magnetic_model \
  --summary-csv data/gnome/stable_materials_summary.csv \
  --cif-zip data/gnome/by_id.zip \
  --output-dir runs/gnome_screen
```

The screening output keeps the full model output set for each candidate:
- hull
- formation energy
- band gap
- moment per atom
- site moments
- ordering probabilities

If a later model includes a trusted `transition_temperature_k` head, screening can
surface that too.

## Cluster

Relevant Slurm helpers:
- [scripts/cluster/train_masked_multitask_mp.slurm](/Users/edo/Edo/Magnets/scripts/cluster/train_masked_multitask_mp.slurm)
- [scripts/cluster/screen_gnome_masked.slurm](/Users/edo/Edo/Magnets/scripts/cluster/screen_gnome_masked.slurm)

## Next Time You Come Back

Do not branch the codebase again into multiple model families.

The next work should stay on this single path:

1. improve the unified dataset
2. improve `model.py`
3. retrain with the same trainer
4. rerun screening

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
python3 -m compileall src
```
