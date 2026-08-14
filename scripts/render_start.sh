#!/usr/bin/env bash
# Render start command for the Rust port (nhl-rust).
#
# Runs from the repo root (same checkout the build used). If the binary is
# missing it prints a clear diagnostic (with the target/release contents) so
# the deploy log shows exactly what happened instead of a bare "No such file".
set -euo pipefail

cd nhl_rust

if [ ! -x target/release/nhl-rust ]; then
  echo "ERROR: ./target/release/nhl-rust not found at $(pwd)" >&2
  echo "The build command may not have run cargo build. Check the build log" >&2
  echo "for '==> render_build: OK' (or 'Finished release profile')." >&2
  echo "target/release contents:" >&2
  if [ -d target/release ]; then
    ls -la target/release >&2
  else
    echo "  (no target/release directory)" >&2
  fi
  exit 1
fi

echo "==> render_start: launching $(pwd)/target/release/nhl-rust"
exec ./target/release/nhl-rust
