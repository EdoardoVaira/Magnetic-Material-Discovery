#!/bin/bash -l
#SBATCH -J magnets-masked-mp
#SBATCH -p gpua40i
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Edo/Magnets}"
DATASET_FILE="${DATASET_FILE:-magnetic_unified.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/magnetic_model}"
FEATURE_VERSION="${FEATURE_VERSION:-}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
HIDDEN_DIM="${HIDDEN_DIM:-192}"
VECTOR_DIM="${VECTOR_DIM:-64}"
NUM_LAYERS="${NUM_LAYERS:-5}"
BATCH_SIZE="${BATCH_SIZE:-96}"
EPOCHS="${EPOCHS:-80}"
PATIENCE="${PATIENCE:-12}"
GRAPH_POOLING="${GRAPH_POOLING:-mean_max}"
ORDERING_POOLING="${ORDERING_POOLING:-mean_max_attention}"
SELECTION_METRIC="${SELECTION_METRIC:-loss}"
BALANCED_SAMPLER="${BALANCED_SAMPLER:-false}"
UNCERTAINTY_WEIGHTING="${UNCERTAINTY_WEIGHTING:-false}"
UNCERTAINTY_INIT_LOG_VAR="${UNCERTAINTY_INIT_LOG_VAR:-0.0}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-5.0}"
LR_SCHEDULER_FACTOR="${LR_SCHEDULER_FACTOR:-0.5}"
LR_SCHEDULER_PATIENCE="${LR_SCHEDULER_PATIENCE:-5}"
ENERGY_LOSS_WEIGHT="${ENERGY_LOSS_WEIGHT:-1.0}"
FORMATION_ENERGY_LOSS_WEIGHT="${FORMATION_ENERGY_LOSS_WEIGHT:-0.5}"
BAND_GAP_LOSS_WEIGHT="${BAND_GAP_LOSS_WEIGHT:-0.5}"
MAG_LOSS_WEIGHT="${MAG_LOSS_WEIGHT:-2.0}"
SITE_MOMENT_LOSS_WEIGHT="${SITE_MOMENT_LOSS_WEIGHT:-1.0}"
ORDERING_LOSS_WEIGHT="${ORDERING_LOSS_WEIGHT:-1.5}"
MAGNETIC_LOSS_WEIGHT="${MAGNETIC_LOSS_WEIGHT:-0.5}"
TRANSITION_TEMPERATURE_LOSS_WEIGHT="${TRANSITION_TEMPERATURE_LOSS_WEIGHT:-0.25}"
MOMENT_CONSISTENCY_LOSS_WEIGHT="${MOMENT_CONSISTENCY_LOSS_WEIGHT:-0.25}"
PREPROCESS_ONLY="${PREPROCESS_ONLY:-false}"

mkdir -p "$REPO_DIR/logs"
cd "$REPO_DIR"

export PYTHONPATH=.

echo "Host: $(hostname)"
echo "Python: $(python3 --version)"
echo "CUDA visible: ${CUDA_VISIBLE_DEVICES:-unset}"
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
PY

python3 train.py \
  --dataset-root data \
  --raw-filename "$DATASET_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --vector-dim "$VECTOR_DIM" \
  --num-layers "$NUM_LAYERS" \
  --dropout 0.05 \
  --graph-pooling "$GRAPH_POOLING" \
  --ordering-pooling "$ORDERING_POOLING" \
  --selection-metric "$SELECTION_METRIC" \
  --grad-clip-norm "$GRAD_CLIP_NORM" \
  --lr-scheduler-factor "$LR_SCHEDULER_FACTOR" \
  --lr-scheduler-patience "$LR_SCHEDULER_PATIENCE" \
  --uncertainty-init-log-var "$UNCERTAINTY_INIT_LOG_VAR" \
  --energy-loss-weight "$ENERGY_LOSS_WEIGHT" \
  --formation-energy-loss-weight "$FORMATION_ENERGY_LOSS_WEIGHT" \
  --band-gap-loss-weight "$BAND_GAP_LOSS_WEIGHT" \
  --magnetization-loss-weight "$MAG_LOSS_WEIGHT" \
  --site-moment-loss-weight "$SITE_MOMENT_LOSS_WEIGHT" \
  --ordering-loss-weight "$ORDERING_LOSS_WEIGHT" \
  --magnetic-loss-weight "$MAGNETIC_LOSS_WEIGHT" \
  --transition-temperature-loss-weight "$TRANSITION_TEMPERATURE_LOSS_WEIGHT" \
  --moment-consistency-loss-weight "$MOMENT_CONSISTENCY_LOSS_WEIGHT" \
  --device auto \
  --log-every 1 \
  $( [ "$PREPROCESS_ONLY" = "true" ] && printf '%s' --preprocess-only ) \
  $( [ -n "$FEATURE_VERSION" ] && printf '%s ' --feature-version "$FEATURE_VERSION" ) \
  $( [ -n "$INIT_CHECKPOINT" ] && printf '%s ' --init-checkpoint "$INIT_CHECKPOINT" ) \
  $( [ "$BALANCED_SAMPLER" = "true" ] && printf '%s' --balanced-sampler ) \
  $( [ "$UNCERTAINTY_WEIGHTING" = "true" ] && printf '%s' --uncertainty-weighting )
