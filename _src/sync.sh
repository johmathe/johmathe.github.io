#!/usr/bin/env bash
# Re-vendor the bispectrum blog from a local checkout of the bispectrum repo.
#   BISPECTRUM_DIR=~/bispectrum _src/sync.sh
set -euo pipefail

src="${BISPECTRUM_DIR:-$HOME/bispectrum}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Excluded paths are also protected from --delete, so a local node_modules/ survives.
rsync -a --delete \
  --exclude node_modules/ \
  --exclude dist/ \
  --exclude scripts/ \
  --exclude __pycache__/ \
  --exclude '/*.md' \
  "$src/blog/" "$here/bispectrum/"

fonts=()
for font in $(grep -o 'fonts/[A-Za-z0-9-]*\.ttf' "$src/blog/src/style.css" | sort -u); do
  fonts+=(--include "/$font")
done

mkdir -p "$here/assets/brand"
rsync -a --delete --prune-empty-dirs \
  --include /__init__.py \
  --include /README.md \
  --include '/*.svg' \
  --include /glyphs/ \
  --include '/glyphs/*.svg' \
  --include /fonts/ \
  --include /fonts/OFL.txt \
  "${fonts[@]}" \
  --exclude '*' \
  "$src/assets/brand/" "$here/assets/brand/"

echo "synced $src -> $here"
