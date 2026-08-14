#!/usr/bin/env python
"""Load test for the NHL service (Rust port vs Flask baseline).

Usage:
    python load_test.py --base http://127.0.0.1:5000 --requests 50 --concurrency 8

Hits a representative set of endpoints and reports per-endpoint latency
percentiles (avg / p50 / p95 / max) + status codes. Point --base at the Rust
service (port 5000) and at Flask (port 5001, say) to compare.
"""

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

import requests

ENDPOINTS = [
    ("GET", "/"),
    ("GET", "/standings"),
    ("GET", "/skaters?team=BOS&season=20252026"),
    ("GET", "/api/standings/20252026"),
    ("GET", "/api/lineups/all"),
    ("GET", "/api/skaters/players?team=BOS&season=20252026"),
    ("GET", "/api/projections/games"),
]


def run_one(base: str, method: str, path: str, session: requests.Session) -> tuple[str, int, float]:
    url = urljoin(base, path)
    t0 = time.perf_counter()
    try:
        r = session.request(method, url, timeout=120)
        status = r.status_code
    except Exception:
        status = 0
    dt = (time.perf_counter() - t0) * 1000.0
    return path, status, dt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:5000")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = "load-test/1.0"

    # Warm-up: one request per endpoint (models/caches load on first hit).
    for method, path in ENDPOINTS:
        try:
            run_one(args.base, method, path, session)
        except Exception:
            pass

    results: dict[str, list[float]] = {p: [] for _, p in ENDPOINTS}
    statuses: dict[str, dict[int, int]] = {p: {} for _, p in ENDPOINTS}
    total = 0
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = []
        for _ in range(max(1, args.requests)):
            for method, path in ENDPOINTS:
                futs.append(pool.submit(run_one, args.base, method, path, session))
        for fut in futs:
            path, status, dt = fut.result()
            total += 1
            results[path].append(dt)
            statuses[path][status] = statuses[path].get(status, 0) + 1
    wall = time.perf_counter() - t_start

    print(f"\nLoad test: base={args.base} requests={total} wall={wall:.1f}s rps={total / wall:.1f}")
    print(f"{'endpoint':<55}{'avg ms':>9}{'p50':>9}{'p95':>9}{'max':>9}  status")
    print("-" * 100)
    for _, path in ENDPOINTS:
        times = sorted(results[path])
        if not times:
            continue
        avg = statistics.mean(times)
        p50 = times[min(len(times) - 1, int(len(times) * 0.50))]
        p95 = times[min(len(times) - 1, int(len(times) * 0.95))]
        mx = max(times)
        st = statuses[path]
        print(f"{path:<55}{avg:>9.1f}{p50:>9.1f}{p95:>9.1f}{mx:>9.1f}  {st}")


if __name__ == "__main__":
    main()
