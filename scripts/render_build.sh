#!/usr/bin/env bash
# Render build script for the Rust port (nhl-rust) on the existing
# nhl-analytics service (env: python) — NO Docker.
#
# 1. Install Python deps (kept for the Render cron jobs: lineups + update_data).
# 2. Install/refresh the Rust toolchain.
# 3. cargo build --release --locked in nhl_rust/.
# 4. Verify the binary exists so a missing build fails loudly here, not in the
#    start command.
set -euo pipefail

echo "==> render_build: installing Python deps"
pip install -r requirements.txt

echo "==> render_build: installing Rust toolchain"
export RUSTUP_HOME="$HOME/.rustup" CARGO_HOME="$HOME/.cargo" PATH="/usr/local/cargo/bin:$PATH"
rustup toolchain install stable --profile minimal
rustup default stable

echo "==> render_build: cargo build --release --locked"
cd nhl_rust
cargo build --release --locked -j 2

if [ ! -x target/release/nhl-rust ]; then
  echo "ERROR: build finished but target/release/nhl-rust is missing" >&2
  ls -la target/release 2>/dev/null >&2 || true
  exit 1
fi

echo "==> render_build: OK ($(pwd)/target/release/nhl-rust)"
