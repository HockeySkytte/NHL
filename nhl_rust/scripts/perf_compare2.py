#!/usr/bin/env python
"""Flask vs Rust comparison: Line Tool + GM Mode simulation endpoints.

Starts Flask (port 5001, threaded dev server) and the Rust release binary
(port 5002), warms both, then measures latency/throughput per endpoint and
writes a CSV.
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
FLASK_BASE = "http://127.0.0.1:5001"
RUST_BASE = "http://127.0.0.1:5002"
CSV_PATH = os.path.join(ROOT, "perf_compare_line_tool_gm.csv")
CONCURRENCY = 4

# (method, path, n_requests) — heavier endpoints get fewer requests.
ENDPOINTS = [
    ("GET", "/api/line-tool/players?team=BOS&season=20252026", 30),
    ("GET", "/api/line-tool/data?team=BOS&season=20252026&seasonState=regular&strengthState=5v5", 10),
    ("GET", "/api/line-tool/wowy?team=BOS&season=20252026&seasonState=regular&strengthState=5v5&players=8471214,8475163", 10),
    ("GET", "/api/line-tool/versus?team=BOS&season=20252026&seasonState=regular&strengthState=5v5&vs_team=TOR&players=8471214", 10),
    ("GET", "/api/line-tool/lines?team=BOS&season=20252026&seasonState=regular&strengthState=5v5", 10),
    ("GET", "/api/player-projections/v2?season=20262027", 30),
    ("POST", "/api/projections/team-season-points-custom", 20),
    ("POST", "/api/projections/all-teams-custom", 10),
    ("POST", "/api/projections/simulate-season", 10),
    ("POST", "/api/projections/simulate-season-batch", 5),
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
    flask = subprocess.Popen(
        [sys.executable, "-c",
         "from dotenv import load_dotenv; load_dotenv(); "
         "from app import create_app; "
         "create_app().run(host='127.0.0.1', port=5001, debug=False, threaded=True)"],
        cwd=ROOT, env=env_flask,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    env_rust = dict(os.environ)
    env_rust["PORT"] = "5002"
    rust = subprocess.Popen([os.path.join(ROOT, "nhl_rust", "target", "release", "nhl-rust.exe")],
                            cwd=os.path.join(ROOT, "nhl_rust"), env=env_rust,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ok = wait_ready(FLASK_BASE, flask) and wait_ready(RUST_BASE, rust)
    return flask, rust, ok


def build_bodies(base):
    """A small but valid BOS custom lineup (real player ids)."""
    global TABLE_BODY
    if TABLE_BODY is not None:
        return
    s = requests.Session()
    sk = s.get(base + "/api/skaters/players?team=BOS&season=20252026", timeout=120).json()
    gl = s.get(base + "/api/goalies/players?team=BOS&season=20252026", timeout=120).json()
    sk_ids = [p["playerId"] for p in sk.get("players", [])][:6]
    gl_ids = [p["playerId"] for p in gl.get("players", [])][:1]
    lineup = [{"pid": pid, "pos": "F", "games": 82, "scratch": False} for pid in sk_ids]
    lineup += [{"pid": pid, "pos": "G", "games": 82, "scratch": False} for pid in gl_ids]
    TABLE_BODY = {
        "team_season_points": {
            "team": "BOS", "season": 20262027, "lineup": lineup,
        },
        "all_teams": {"season": 20262027},
        "sim": {"season": 20262027, "seed": 42},
        "sim_batch": {"season": 20262027, "seed": 42, "numSims": 3},
    }


def hit(base, method, path, session):
    t0 = time.perf_counter()
    try:
        if method == "POST":
            key = path.split("/")[-1]
            if key == "team-season-points-custom":
                body = TABLE_BODY["team_season_points"]
            elif key == "all-teams-custom":
                body = TABLE_BODY["all_teams"]
            elif key == "simulate-season":
                body = TABLE_BODY["sim"]
            else:
                body = TABLE_BODY["sim_batch"]
            r = session.post(base + path, json=body, timeout=300)
        else:
            r = session.get(base + path, timeout=300)
        status = r.status_code
    except Exception:
        status = 0
    return status, (time.perf_counter() - t0) * 1000.0


def measure(base, method, path, n):
    session = requests.Session()
    session.headers["User-Agent"] = "perf-compare/1.0"
    for _ in range(2):
        try:
            hit(base, method, path, session)
        except Exception:
            pass
    times, statuses = [], {}
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futs = [pool.submit(hit, base, method, path, session) for _ in range(n)]
        for fut in futs:
            status, dt = fut.result()
            times.append(dt)
            statuses[status] = statuses.get(status, 0) + 1
    wall = time.perf_counter() - t_start
    ts = sorted(times)
    m = len(ts)
    return {
        "n": m,
        "avg": statistics.mean(ts),
        "p50": ts[min(m - 1, int(m * 0.50))],
        "p95": ts[min(m - 1, int(m * 0.95))],
        "max": ts[-1],
        "rps": m / wall if wall > 0 else 0.0,
        "status": "+".join(f"{k}x{v}" for k, v in sorted(statuses.items())),
    }


def main():
    print(f"Starting Flask on :5001 and Rust on :5002 ...")
    flask, rust, ok = start_servers()
    if not ok:
        for p in (flask, rust):
            if p and p.poll() is None:
                p.kill()
        sys.exit(1)
    try:
        build_bodies(FLASK_BASE)
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        rows = []
        for app_name, base in (("flask", FLASK_BASE), ("rust", RUST_BASE)):
            print(f"\n=== {app_name} ===")
            for method, path, n in ENDPOINTS:
                res = measure(base, method, path, n)
                rows.append({
                    "app": app_name, "method": method, "endpoint": path,
                    "status": res["status"], "requests": res["n"],
                    "avg_ms": round(res["avg"], 1), "p50_ms": round(res["p50"], 1),
                    "p95_ms": round(res["p95"], 1), "max_ms": round(res["max"], 1),
                    "rps": round(res["rps"], 1), "timestamp": timestamp,
                })
                print(f"{method:<5} {path[:60]:<60} n={res['n']:<3} avg={res['avg']:8.1f}ms "
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
