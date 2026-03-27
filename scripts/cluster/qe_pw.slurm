#!/bin/bash -l
#SBATCH -J qe-pw
#SBATCH -p lu48
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Edo/Magnets}"
CASE_DIR="${CASE_DIR:-$REPO_DIR/validation/fe2co_qe}"
INPUT_FILE="${INPUT_FILE:-scf_fm.in}"
OUTPUT_FILE="${OUTPUT_FILE:-${INPUT_FILE%.in}.out}"
LOCAL_ROOT_BASE="${LOCAL_ROOT_BASE:-}"
LOCAL_ROOT="${LOCAL_ROOT:-}"
if [[ -n "$LOCAL_ROOT" ]]; then
  :
elif [[ -n "$LOCAL_ROOT_BASE" ]]; then
  LOCAL_ROOT="${LOCAL_ROOT_BASE}/${SLURM_JOB_ID:-qe}"
elif [[ -n "${SLURM_TMPDIR:-}" ]]; then
  LOCAL_ROOT="${SLURM_TMPDIR}"
elif [[ -n "${SNIC_TMP:-}" ]] && [[ "${SNIC_TMP}" = /* ]]; then
  LOCAL_ROOT="${SNIC_TMP}"
elif [[ -n "${TMPDIR:-}" ]] && [[ "${TMPDIR}" = /* ]]; then
  LOCAL_ROOT="${TMPDIR}"
else
  LOCAL_ROOT="/tmp/$USER/${SLURM_JOB_ID:-qe}"
fi
LOCAL_CASE_DIR="${LOCAL_ROOT}/qe_case"
SHARED_TMP_DIR="${CASE_DIR}/tmp"
SYNC_INTERVAL_SEC="${SYNC_INTERVAL_SEC:-300}"
TMP_SYNC_PREFIXES="${TMP_SYNC_PREFIXES:-}"
TMP_PREFIX_ARRAY=()

mkdir -p "$REPO_DIR/logs"
mkdir -p "$LOCAL_CASE_DIR"
cp "$CASE_DIR/$INPUT_FILE" "$LOCAL_CASE_DIR/$INPUT_FILE"
if [[ -d "$CASE_DIR/pseudo" ]]; then
  mkdir -p "$LOCAL_CASE_DIR/pseudo"
  cp "$CASE_DIR"/pseudo/* "$LOCAL_CASE_DIR/pseudo/"
fi
cd "$LOCAL_CASE_DIR"
mkdir -p "$LOCAL_CASE_DIR/tmp"
mkdir -p "$SHARED_TMP_DIR"

read_tmp_prefixes() {
  if [[ -z "$TMP_SYNC_PREFIXES" ]]; then
    return
  fi

  local old_ifs="$IFS"
  IFS=', '
  # shellcheck disable=SC2206
  TMP_PREFIX_ARRAY=($TMP_SYNC_PREFIXES)
  IFS="$old_ifs"
}

sync_tmp_from_shared() {
  if [[ ! -d "$SHARED_TMP_DIR" ]]; then
    return
  fi

  if [[ "${#TMP_PREFIX_ARRAY[@]}" -eq 0 ]]; then
    if [[ -n "$(find "$SHARED_TMP_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
      rsync -a "$SHARED_TMP_DIR"/ "$LOCAL_CASE_DIR/tmp"/
    fi
    return
  fi

  for prefix in "${TMP_PREFIX_ARRAY[@]}"; do
    if [[ -d "$SHARED_TMP_DIR/$prefix" ]]; then
      mkdir -p "$LOCAL_CASE_DIR/tmp/$prefix"
      rsync -a "$SHARED_TMP_DIR/$prefix"/ "$LOCAL_CASE_DIR/tmp/$prefix"/
    fi
  done
}

read_tmp_prefixes
sync_tmp_from_shared

module load GCC/12.3.0 OpenMPI/4.1.5 QuantumESPRESSO/7.4.1

QE_ROOT="${QE_ROOT:-/sw/pkg/QuantumESPRESSO/7.4.1}"
PW_EXEC="${PW_EXEC:-$QE_ROOT/bin/pw.x}"
QE_NK="${QE_NK:-}"
export OMPI_MCA_pml="ob1"
export OMPI_MCA_btl="self,vader,tcp"

echo "Host: $(hostname)"
echo "Case dir: $CASE_DIR"
echo "Local root: $LOCAL_ROOT"
echo "Local scratch: $LOCAL_CASE_DIR"
echo "Input: $INPUT_FILE"
echo "Tasks: ${SLURM_NTASKS:-unset}"
echo "pw.x: $PW_EXEC"
echo "QE_NK: ${QE_NK:-unset}"
echo "TMP_SYNC_PREFIXES: ${TMP_SYNC_PREFIXES:-all}"
echo "OMPI_MCA_pml: $OMPI_MCA_pml"
echo "OMPI_MCA_btl: $OMPI_MCA_btl"
if [[ -d "$LOCAL_CASE_DIR/pseudo" ]]; then
  echo "Local pseudo contents:"
  ls -l "$LOCAL_CASE_DIR/pseudo"
fi

sync_tmp_back() {
  if [[ -d "$LOCAL_CASE_DIR/tmp" ]]; then
    mkdir -p "$SHARED_TMP_DIR"
    if [[ "${#TMP_PREFIX_ARRAY[@]}" -eq 0 ]]; then
      rsync -a "$LOCAL_CASE_DIR/tmp"/ "$SHARED_TMP_DIR"/ || true
    else
      for prefix in "${TMP_PREFIX_ARRAY[@]}"; do
        if [[ -d "$LOCAL_CASE_DIR/tmp/$prefix" ]]; then
          mkdir -p "$SHARED_TMP_DIR/$prefix"
          rsync -a "$LOCAL_CASE_DIR/tmp/$prefix"/ "$SHARED_TMP_DIR/$prefix"/ || true
        fi
      done
    fi
  fi
}

sync_loop_pid=""
if [[ "${SYNC_INTERVAL_SEC}" -gt 0 ]]; then
  (
    while true; do
      sleep "${SYNC_INTERVAL_SEC}"
      sync_tmp_back
    done
  ) &
  sync_loop_pid=$!
fi

cleanup() {
  if [[ -n "$sync_loop_pid" ]]; then
    kill "$sync_loop_pid" 2>/dev/null || true
    wait "$sync_loop_pid" 2>/dev/null || true
  fi
  sync_tmp_back
}

trap cleanup EXIT TERM INT

pw_args=()
if [[ -n "$QE_NK" ]]; then
  pw_args+=(-nk "$QE_NK")
fi
pw_args+=(-in "$INPUT_FILE")

mpirun --mca pml ob1 --mca btl self,vader,tcp -np "${SLURM_NTASKS:-1}" "$PW_EXEC" "${pw_args[@]}" 2>&1 | tee "$CASE_DIR/$OUTPUT_FILE" > "$OUTPUT_FILE"
