#!/usr/bin/env python
"""Flask vs Rust performance comparison (local).

Starts the Flask dev server (port 5001, threaded) and the Rust release
binary (port 5002), warms both, then measures per-endpoint latency
percentiles + throughput and writes results to a CSV file.
"""
import csv
import datetime
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = r"c:\Apps\NHL"
FLASK_PORT = 5001
RUST_PORT = 5002
FLASK_BASE = f"http://127.0.0.1:{FLASK_PORT}"
RUST_BASE = f"http://127.0.0.1:{RUST_PORT}"
CSV_PATH = os.path.join(ROOT, "perf_compare_flask_vs_rust.csv")

REQUESTS_PER_ENDPOINT = 30
CONCURRENCY = 6

# (method, path) — endpoints implemented by BOTH apps (no auth needed).
ENDPOINTS = [
    ("GET", "/"),
    ("GET", "/standings"),
    ("GET", "/skaters?team=BOS&season=20252026"),
    ("GET", "/api/standings/20252026"),
    ("GET", "/api/lineups/all"),
    ("GET", "/api/skaters/players?team=BOS&season=20252026"),
    ("GET", "/api/goalies/players?team=BOS&season=20252026"),
    ("GET", "/api/teams/card?team=BOS&season=20252026&seasonState=regular&strengthState=5v5"),
    ("GET", "/api/skaters/card?playerId=8471214&scope=season&season=20252026&seasonState=regular&strengthState=5v5&rates=Totals&metricIds=Offense%7CRAPM%20CF,Offense%7CRAPM%20xGF,Ice%20Time%7CGP"),
    ("GET", "/api/rapm/scale?season=20252026&rates=Rates&metric=corsi&playerId=8471214"),
    ("GET", "/api/rapm/player/8471214"),
    ("GET", "/api/projections/games"),
    ("POST", "/api/skaters/table"),  # body filled in main()
]

TABLE_BODY = None


def wait_ready(base, proc, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            r = requests.get(base + "/robots.txt", timeout=5)
            if r.status_code in (200, 404):
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def start_servers():
    env_flask = dict(os.environ)
    env_flask["XG_PRELOAD"] = "0"
    flask_cmd = [
        sys.executable,
        "-c",
        "from dotenv import load_dotenv; load_dotenv(); "
        "from app import create_app; "
        "create_app().run(host='127.0.0.1', port=%d, debug=False, threaded=True)" % FLASK_PORT,
    ]
    flask = subprocess.Popen(flask_cmd, cwd=ROOT, env=env_flask,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    env_rust = dict(os.environ)
    env_rust["PORT"] = str(RUST_PORT)
    rust_exe = os.path.join(ROOT, "nhl_rust", "target", "release", "nhl-rust.exe")
    rust = subprocess.Popen([rust_exe], cwd=os.path.join(ROOT, "nhl_rust"), env=env_rust,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ok = True
    if not wait_ready(FLASK_BASE, flask):
        print("Flask failed to start")
        ok = False
    if not wait_ready(RUST_BASE, rust):
        print("Rust failed to start")
        ok = False
    return flask, rust, ok


def build_table_body(base):
    """BOS roster + all skater metric ids for the table POST."""
    global TABLE_BODY
    if TABLE_BODY is not None:
        return TABLE_BODY
    s = requests.Session()
    defs = s.get(base + "/api/skaters/card/defs", timeout=120).json()
    metric_ids = [m["id"] for m in defs.get("metrics", [])]
    players = s.get(base + "/api/skaters/players?team=BOS&season=20252026", timeout=120).json()
    player_ids = [p["playerId"] for p in players.get("players", [])]
    TABLE_BODY = {
        "season": "20252026", "seasonState": "regular", "strengthState": "5v5",
        "xgModel": "xG_F", "rates": "Totals", "scope": "season",
        "minGP": 0, "minTOI": 0,
        "playerIds": player_ids, "metricIds": metric_ids,
    }
    return TABLE_BODY


def hit(base, method, path, session):
    t0 = time.perf_counter()
    try:
        if method == "POST":
            r = session.post(base + path, json=build_table_body(base), timeout=180)
        else:
            r = session.request(method, base + path, timeout=180)
        status = r.status_code
    except Exception:
        status = 0
    return status, (time.perf_counter() - t0) * 1000.0


def measure(base, method, path):
    session = requests.Session()
    session.headers["User-Agent"] = "perf-compare/1.0"
    # warm-up
    for _ in range(2):
        try:
            hit(base, method, path, session)
        except Exception:
            pass
    times = []
    statuses = {}
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futs = [pool.submit(hit, base, method, path, session)
                for _ in range(REQUESTS_PER_ENDPOINT)]
        for fut in futs:
            status, dt = fut.result()
            times.append(dt)
            statuses[status] = statuses.get(status, 0) + 1
    wall = time.perf_counter() - t_start
    times_sorted = sorted(times)
    n = len(times_sorted)
    return {
        "n": n,
        "avg": statistics.mean(times_sorted),
        "p50": times_sorted[min(n - 1, int(n * 0.50))],
        "p95": times_sorted[min(n - 1, int(n * 0.95))],
        "max": times_sorted[-1],
        "rps": n / wall if wall > 0 else 0.0,
        "status": "+".join(f"{k}x{v}" for k, v in sorted(statuses.items())),
    }


def main():
    print(f"Starting Flask on :{FLASK_PORT} and Rust on :{RUST_PORT} ...")
    flask, rust, ok = start_servers()
    if not ok:
        for p in (flask, rust):
            if p and p.poll() is None:
                p.kill()
        sys.exit(1)
    try:
        build_table_body(FLASK_BASE)  # metric ids from Flask defs (identical to Rust)
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        rows = []
        for app_name, base in (("flask", FLASK_BASE), ("rust", RUST_BASE)):
            print(f"\n=== {app_name} ===")
            for method, path in ENDPOINTS:
                label = f"{method} {path}"
                res = measure(base, method, path)
                rows.append({
                    "app": app_name,
                    "method": method,
                    "endpoint": path,
                    "status": res["status"],
                    "requests": res["n"],
                    "avg_ms": round(res["avg"], 1),
                    "p50_ms": round(res["p50"], 1),
                    "p95_ms": round(res["p95"], 1),
                    "max_ms": round(res["max"], 1),
                    "rps": round(res["rps"], 1),
                    "timestamp": timestamp,
                })
                print(f"{label:<70} n={res['n']:<3} avg={res['avg']:8.1f}ms "
                      f"p50={res['p50']:7.1f} p95={res['p95']:8.1f} max={res['max']:8.1f} "
                      f"rps={res['rps']:6.1f} status={res['status']}")
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults written to {CSV_PATH}")
    finally:
        for p in (flask, rust):
            if p.poll() is None:
                p.kill()


if __name__ == "__main__":
    main()
