#!/bin/bash

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 CASE_DIR SOURCE_PREFIX TARGET_PREFIX [TARGET_PREFIX ...]" >&2
  exit 1
fi

CASE_DIR="$1"
SOURCE_PREFIX="$2"
shift 2
TARGET_PREFIXES=("$@")

SOURCE_DIR="$CASE_DIR/tmp/$SOURCE_PREFIX"
SOURCE_SAVE_DIR="$SOURCE_DIR/$SOURCE_PREFIX.save"

if [[ ! -d "$SOURCE_SAVE_DIR" ]]; then
  echo "missing source save directory: $SOURCE_SAVE_DIR" >&2
  exit 1
fi

for target_prefix in "${TARGET_PREFIXES[@]}"; do
  target_dir="$CASE_DIR/tmp/$target_prefix"
  target_save_dir="$target_dir/$target_prefix.save"

  rm -rf "$target_dir"
  mkdir -p "$target_dir"
  cp -a "$SOURCE_SAVE_DIR" "$target_save_dir"

  if [[ -f "$target_save_dir/data-file-schema.xml" ]]; then
    perl -0pi -e "s/\\Q$SOURCE_PREFIX.save\\E/$target_prefix.save/g; s/\\Q$SOURCE_PREFIX.\\E/$target_prefix./g" "$target_save_dir/data-file-schema.xml"
  fi

  echo "Prepared force-theorem state: $SOURCE_PREFIX -> $target_prefix"
done
