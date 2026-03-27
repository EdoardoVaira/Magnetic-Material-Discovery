#!/bin/bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CASE_DIR" >&2
  exit 1
fi

CASE_DIR="$1"
PSEUDO_DIR="$CASE_DIR/pseudo"
BASE_URL="https://pseudopotentials.quantum-espresso.org/upf_files"

mkdir -p "$PSEUDO_DIR"

files=(
  "Fe.pbesol-spn-rrkjus_psl.1.0.0.UPF"
  "Co.pbesol-spn-rrkjus_psl.0.3.1.UPF"
  "Ge.pbesol-dn-rrkjus_psl.1.0.0.UPF"
  "Fe.rel-pbesol-spn-rrkjus_psl.1.0.0.UPF"
  "Co.rel-pbesol-spn-rrkjus_psl.0.3.1.UPF"
  "Ge.rel-pbesol-dn-rrkjus_psl.1.0.0.UPF"
)

for file in "${files[@]}"; do
  curl -fsSL "$BASE_URL/$file" -o "$PSEUDO_DIR/$file"
done

echo "Fetched ${#files[@]} USPP pseudopotentials into $PSEUDO_DIR"
