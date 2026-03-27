#!/bin/bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 CASE_DIR SOURCE_PREFIX TARGET_PREFIX" >&2
  exit 1
fi

CASE_DIR="$1"
SOURCE_PREFIX="$2"
TARGET_PREFIX="$3"

SOURCE_DIR="$CASE_DIR/tmp/$SOURCE_PREFIX"
TARGET_DIR="$CASE_DIR/tmp/$TARGET_PREFIX"
SOURCE_SAVE_DIR="$SOURCE_DIR/$SOURCE_PREFIX.save"
TARGET_SAVE_DIR="$TARGET_DIR/$TARGET_PREFIX.save"

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "missing source restart directory: $SOURCE_DIR" >&2
  exit 1
fi

rm -rf "$TARGET_DIR"
mkdir -p "$CASE_DIR/tmp"
cp -a "$SOURCE_DIR" "$TARGET_DIR"

for suffix in mix1 restart_k restart_scf wfc1; do
  src="$TARGET_DIR/$SOURCE_PREFIX.$suffix"
  dst="$TARGET_DIR/$TARGET_PREFIX.$suffix"
  if [[ -f "$src" ]]; then
    mv "$src" "$dst"
  fi
done

if [[ -d "$TARGET_DIR/$SOURCE_PREFIX.save" ]]; then
  mv "$TARGET_DIR/$SOURCE_PREFIX.save" "$TARGET_SAVE_DIR"
fi

if [[ -f "$TARGET_SAVE_DIR/data-file-schema.xml" ]]; then
  perl -0pi -e "s/\Q$SOURCE_PREFIX.save\E/$TARGET_PREFIX.save/g; s/\Q$SOURCE_PREFIX.\E/$TARGET_PREFIX./g" "$TARGET_SAVE_DIR/data-file-schema.xml"
fi

echo "Cloned $SOURCE_PREFIX -> $TARGET_PREFIX in $CASE_DIR/tmp"
