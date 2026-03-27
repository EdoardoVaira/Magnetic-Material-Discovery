#!/bin/bash -l
#SBATCH -J gnome-masked
#SBATCH -p gpua40i
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Edo/Magnets}"
RUN_DIR="${RUN_DIR:-runs/magnetic_model}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/gnome_screen_masked_all_le40}"
SUMMARY_CSV="${SUMMARY_CSV:-data/gnome/stable_materials_summary.csv}"
CIF_ZIP="${CIF_ZIP:-data/gnome/by_id.zip}"
REQUIRED_ELEMENTS="${REQUIRED_ELEMENTS:-}"
ALLOWED_ELEMENTS="${ALLOWED_ELEMENTS:-}"
MAX_SITES="${MAX_SITES:-40}"
MAX_ELEMENTS="${MAX_ELEMENTS:-8}"
SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
NUM_SHARDS="${NUM_SHARDS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TOP_K="${TOP_K:-200}"
REPORT_EXAMPLES="${REPORT_EXAMPLES:-16}"

mkdir -p "$REPO_DIR/logs"
cd "$REPO_DIR"

if [ "$NUM_SHARDS" -gt 1 ]; then
  OUTPUT_DIR="${OUTPUT_DIR%/}/shard_${SHARD_INDEX}"
fi

export PYTHONPATH=.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"

echo "Host: $(hostname)"
echo "Python: $(python3 --version)"
echo "CUDA visible: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "Shard: ${SHARD_INDEX}/${NUM_SHARDS}"
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
print("torch_num_threads", torch.get_num_threads())
PY

python3 screen.py \
  --run-dir "$RUN_DIR" \
  --summary-csv "$SUMMARY_CSV" \
  --cif-zip "$CIF_ZIP" \
  --output-dir "$OUTPUT_DIR" \
  --required-elements "$REQUIRED_ELEMENTS" \
  --allowed-elements "$ALLOWED_ELEMENTS" \
  --max-sites "$MAX_SITES" \
  --max-elements "$MAX_ELEMENTS" \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS" \
  --batch-size "$BATCH_SIZE" \
  --device cuda \
  --top-k "$TOP_K" \
  --report-examples "$REPORT_EXAMPLES"
