# syntax=docker/dockerfile:1
# Build context: the repository root (c:\Apps\NHL), Dockerfile at nhl_rust/Dockerfile.
# Render: "Dockerfile path: ./nhl_rust/Dockerfile", root dir ".".

# ── Build stage ────────────────────────────────────────────────────────────
FROM rust:1.97-bookworm AS build
WORKDIR /build

# Prebuild dependencies using a dummy main for layer caching.
COPY nhl_rust/Cargo.toml nhl_rust/Cargo.lock* ./
RUN mkdir -p src && echo 'fn main() {}' > src/main.rs && cargo build --release

COPY nhl_rust/src ./src
RUN touch src/main.rs && cargo build --release

# ── Runtime stage (Python kept for Render cron jobs + admin scripts) ──────
FROM python:3.11-slim-bookworm
WORKDIR /app
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/target/release/nhl-rust /app/nhl-rust

# Shared assets/templates from the Flask tree + extracted about content.
COPY app/templates /app/templates
COPY app/static /app/static
COPY nhl_rust/data /app/data
COPY Teams.csv /app/Teams.csv
COPY Last_date.csv /app/Last_date.csv

# Rust xG models (converted JSON dumps) — loader joins {fname}.json directly.
COPY Model/rust /app/Model/rust

# Python deps for the Render cron jobs (lineups + update_data).
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Full `app` package + scripts so cron/admin Python scripts can import them
# (update_data/backfill run in HTTP mode against the Rust service via
# APP_INTERNAL_BASE_URL).
COPY app /app/app
COPY scripts /app/scripts

ENV TEMPLATE_DIR=/app/templates \
    STATIC_DIR=/app/static \
    ABOUT_DATA_JSON=/app/data/about.json \
    TEAMS_CSV_PATH=/app/Teams.csv \
    LAST_DATES_CSV_PATH=/app/Last_date.csv \
    MODEL_DIR=/app/Model/rust \
    XG_CACHE_DIR=/app/cache \
    PRESTART_DIR=/app/prestart

EXPOSE 5000
CMD ["/app/nhl-rust"]
