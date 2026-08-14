# Port Plan: Flask → Rust (`nhl_rust/`)

> Status: **PLANNING COMPLETE — ready for implementation approval**
> Date: 2026-08-13
> Constraint: the Flask app stays fully intact and running. The Rust app is a **new, separate service** in `nhl_rust/` that can later replace Flask on Render.

---

## 1. Executive summary

The current backend is a single-module Flask app: **`app/routes.py` ≈ 22,127 lines, 368 top-level functions, ~112 routes**, backed by Supabase (PostgREST + GoTrue), the NHL APIs, CSV/JSON fallbacks, and windowed XGBoost models loaded with `joblib`. It also handles auth, Stripe billing, admin jobs, and a background prestart logger.

The Rust port targets **identical HTTP behavior** (paths, query params, JSON shapes, cache semantics, env vars) with a fraction of the runtime cost and much lower tail latency. It is split into 7 milestones (M0–M6), each independently shippable, with a parity test harness comparing Flask vs Rust responses on recorded fixtures.

**Headline decisions**

| Concern | Decision |
|---|---|
| Web framework | **Axum** (Tokio) — modern, typed routing, first-class middleware, single static binary |
| Templates | **minijinja** — Jinja2-compatible; the 23 existing templates port over with tiny edits (`url_for` shim, `request` context) |
| HTTP client | **reqwest** (pooled) replicating all 37 `requests.get` call sites + timeouts |
| Caching | **moka** TTL+capacity caches (1:1 mapping to the ~40 module-level TTL dicts) + same on-disk JSON cache layout |
| Supabase | Raw **PostgREST HTTP** (`/rest/v1/...`) + **GoTrue admin API** (`/auth/v1/...`) — no SDK needed |
| xG models | One-time **Python conversion script** (`joblib` XGBClassifier → XGBoost UBJSON + feature metadata), loaded in Rust via the `xgboost` crate |
| HTML scraping | **`scraper`** (html5ever) for NHL `TV/TH*.HTM` shift reports and the community feed fallback |
| Stripe | **`stripe`** crate (official async client), raw-body webhook verification |
| Sessions/CSRF | HMAC-signed cookies (tower-cookies + `cookie::Key`) storing the same `auth_user` JSON record; same CSRF token pattern |
| Admin jobs / scripts | **All `scripts/*.py` stay Python.** The Rust app shells out to them for admin jobs; Render cron jobs keep running Python in the runtime image |
| Deployment | Multi-stage **Dockerfile**: Rust build stage + `python:3.11-slim` runtime (Python kept for cron jobs/scripts) → single binary on Render |

**Total estimate:** ~15–18k lines of Rust across ~25 modules, 6–9 developer-weeks including parity testing.

---

## 2. Current-state inventory (research summary)

### 2.1 Route families and line numbers in `app/routes.py`

| # | Family | Routes (count) | Lines | External deps |
|---|---|---|---|---|
| 1 | Auth + Stripe + account + admin users | 22 | 2464–3106 | Supabase GoTrue, Stripe |
| 2 | Admin ops + jobs + lineups upload | 7 | 1377–1837 | subprocess `python`, MySQL preflight |
| 3 | Page routes | 18 | 2382–2444, 3106–3393, 4922–4944, 5623–5662, 19384 | templates only |
| 4 | Line Tool APIs | 5 | 3393–4930 | Supabase `pbp/shifts/game_data/players/forward_lines/defense_pairings` |
| 5 | Odds / lineups / projections data | 6 | 5662–6086 | Supabase `odds_snapshots/lineups/player_projections` |
| 6 | GM Mode projections + simulations | 8 | 6086–8600 | Supabase `nhl_current_playerprojections/rapm_context`, NHL schedule/roster/bracket |
| 7 | Skaters/goalies/teams analytics (players, cards, tables, scatter, edge, series) | 16 | 8600–15210 | NHL stats REST, NHL Edge, Supabase `season_stats(+_teams)/rapm/rapm_context`, static CSVs |
| 8 | RAPM scale/career, seasons, standings, live, logos/headshots | 10 | 13826–15325 | Supabase `rapm/rapm_context`, NHL standings/schedule, assets.nhle.com |
| 9 | Projection league/trend/current, schedules | 7 | 17670–19384 | Supabase `player_current_projections/nhl_player_metrics/player_projections` |
| 10 | Game endpoints (boxscore, right-rail, **play-by-play**, shifts) | 4 | 19391–21615 | NHL gamecenter + **www.nhl.com HTML shift reports**, XGBoost models |
| 11 | Shooting / goaltending | 2 | 21615–end | Supabase `pbp` + NHL PBP fallback |
| 12 | Misc (robots, sitemap, diag, favicon, community) | 5 | 2382–2402, 3295–3330, 15111 | community site scrape |

### 2.2 External services

| Service | Endpoints used |
|---|---|
| `api-web.nhle.com` | `/v1/schedule/{date}`, `/v1/schedule/now`, `/v1/playoff-bracket/{year}`, `/v1/club-schedule-season/{TEAM}/{season}`, `/v1/club-stats-season/{TEAM}`, `/v1/roster/{TEAM}/current`, `/v1/roster/{team}/{season}`, `/v1/player/{pid}/landing`, `/v1/standings/now`, `/v1/standings/...` (4 variants), `/v1/standings-season/...`, `/v1/edge/*` (4 metric families), `/v1/gamecenter/{id}/boxscore|right-rail|play-by-play|landing`, `/v1/partner-game/US/{date}|now` |
| `api.nhle.com` (stats REST) | `/stats/rest/en/skater/summary`, `/goalie/summary`, `/team/summary`, `/team/summaryshooting`, `/skater/bios`, `/goalie/bios`, `/team/standings` (all with `cayenneExp` filters) |
| `www.nhl.com` | `/scores/htmlreports/{season_dir}/TV{suffix}.HTM` + `TH{suffix}.HTM` (shift reports, BeautifulSoup-parsed) |
| `assets.nhle.com` | `/logos/nhl/svg/{ABBR}_light.svg`, `/mugs/nhl/{season}/{team}/{pid}.png`, `/mugs/nhl/latest/{pid}.png` (proxied) |
| Community site | `COMMUNITY_SITE_URL` HTML feed scraped as fallback |
| Supabase | PostgREST tables (see 2.3) + GoTrue auth admin API |
| Stripe | Checkout sessions, billing portal, subscriptions, webhooks |
| MySQL (admin only) | `SELECT 1` preflight before `run-update-data` (SQLAlchemy) |

### 2.3 Supabase tables used by the app

`teams`, `last_dates`, `box_ids`, `players`, `pbp`, `shifts`, `season_stats`, `season_stats_teams`, `rapm`, `rapm_context`, `player_projections`, `lineups`, `odds_snapshots`, `started_overrides`, `community_posts`, `community_post_media`, `user_accounts`, `card_builder_layouts`, `game_data`, `player_game_projections`, `player_current_projections`, `nhl_current_playerprojections`, `nhl_player_metrics`, `forward_lines`, `defense_pairings`.

Reads go through `_sb_read()` (`app/routes.py:15325`): paginated `Range` queries of 1000 rows, PostgREST filters (`eq.`, `in.(...)`), snake_case→CSV column renames, returns `None` on any failure (callers then fall back to CSV). Writes use thin helpers in `app/supabase_client.py` (upserts with `on_conflict`, column-drop retries for lagging schemas).

### 2.4 Caches (~40 module-level TTL dicts)

Full table with TTL env vars in **§7**. Highlights: `_PBP_CACHE`/`_SHIFTS_CACHE`/`_BOX_CACHE` (600s, live 5s), `_LT_*` (1800s), `_SEASONSTATS_AGG_CACHE` + goalies career matrix (1800s + gzip/pickle disk), `_SKATERS_SHOOTING_CACHE`/`_GOALIES_GOALTENDING_CACHE` (180s), image proxies (30 days), `_LINEUPS_ALL_CACHE` (300s), `_GM_PROJECTIONS_CACHE` (300s), `_CUSTOM_LINEUPS_CACHE` (43200s, actor-keyed).

### 2.5 Models

- **XGBoost classifiers** (sklearn-wrapped, `predict_proba` only): families `xgbs` (shots→xG_S), `xgb` (fenwick→xG_F), `xgb2` (fenwick→xG_F2); windowed files `{prefix}_{start}_{end}.pkl`; 15 windows each. Used **only** by the play-by-play endpoint (`app/routes.py:19474–20410`).
- **Non-ML math**: V2 projections (`_EVPP_COEF` linear model), game-projection SITUATION logistics, Empty-Net fenwick logistic, Normal CDF + Poisson for simulations. All constants are port-critical (Appendix A).

---

## 3. Technical decisions (rationale)

### 3.1 Web framework — Axum

- Typed extractors map cleanly to Flask's `request.args`/`request.get_json()` patterns.
- `tower-http` gives `CompressionLayer` (replaces Flask-Compress), `ServeDir` for static assets, `TraceLayer` for logging.
- Middleware ordering: compression → premium-gating → sessions → routes.

### 3.2 Templates — minijinja (Jinja2-compatible)

The 23 templates are all Jinja with `extends base.html`. minijinja supports `extends/block/include/macros`, the built-in `tojson` filter, and custom functions. Porting steps per template:

1. Copy from `app/templates/` into `nhl_rust/templates/` (single source of truth during dev: env `TEMPLATE_DIR` pointing at `../app/templates`; the Docker build copies them).
2. Provide context globals mirroring Flask: `url_for(name, **kwargs)` function (route-name table), `request` object (`url_root` from `APP_BASE_URL`/host header), `auth_user`, `auth_enabled`, `auth_plan_options`, `auth_login_target`, `csrf_token`, `teams`, `meta_*`, `active_tab`, etc. — same context dict as `inject_auth_state()` + each page route (`app/routes.py:1345`).
3. Only built-in Jinja features are used (no custom filters) → syntax edits should be minimal.

### 3.3 JSON strategy

- Pass-through endpoints (boxscore, right-rail, landing, standings, rosters, edge) use `serde_json::Value`.
- Computed endpoints (cards, line tool, PBP) use `Value`/`json!` first for parity, with typed structs introduced later for hot paths. This mirrors how the Python code works (dicts everywhere) and minimizes shape drift.

### 3.4 Supabase client (custom module, no SDK)

PostgREST over reqwest:
- Read: `GET {url}/rest/v1/{table}?select={cols}&{col}=eq.{v}` + `Range: {offset}-{offset+999}` pagination; `in.(a,b,c)` → `in.()` with comma list; `order` → `order=` query.
- Upsert: `POST ...?on_conflict={cols}` with `Prefer: resolution=merge-duplicates`.
- GoTrue admin: `GET/POST/PUT/DELETE {url}/auth/v1/admin/users[...]` with service key; sign-in: `POST /auth/v1/token?grant_type=password` with anon key.
- Reproduce `supabase-py` quirks that callers rely on: `None` on error (distinct from empty list), `col_map` renames, the 10× column-dropping upsert retry for `user_accounts` (`_missing_column_name` regex).

### 3.5 Model inference — the core risk (see §6.3)

Python stores pickled `xgboost.sklearn.XGBClassifier` objects — **unreadable by any Rust crate directly**. Solution: a one-time conversion script (`scripts/export_models_for_rust.py`, kept in the Python side) that, per `.pkl`:

- `model.get_booster().save_model('Model/rust/{name}.ubj')` (UBJSON, the XGBoost interchange format), and
- writes `{name}.meta.json`: feature names (`feature_names_in_`), objective, base_score, window (prefix, start, end).

The Rust app loads UBJ with the `xgboost` crate; `predict()` returns raw margins; probability = `1/(1+e^-margin)` (binary:logistic, matching sklearn `predict_proba[:,1]`). One-hot vectorization (`_vectorize_row_for_model`, `app/routes.py:20190`) is re-implemented verbatim. Windows and averaging (`XG_WINDOWS`) replicate `window_filenames_for_season` (`:20328`).

**Fallback plan** if the C dependency is problematic on Render: dump trees to JSON (`xgb.dump_model(dump_format='json')`) and evaluate with a ~300-line pure-Rust tree evaluator (same math, no C lib).

### 3.6 Sessions & CSRF

- Flask signs a client-side cookie containing the `auth_user` dict. Rust equivalent: `tower-cookies` + `cookie::Key` (HMAC-SHA256) signing a JSON payload with the same 30-day permanent session semantics. Cookie flags replicate `HttpOnly; SameSite=Lax`.
- CSRF: 32-byte token stored in the session cookie, validated via `secrets.compare_digest`-equivalent (constant-time compare) on the same POST routes as today. `/login`, `/signup`, `/logout`, `/stripe/webhook` remain exempt, exactly as now.

### 3.7 Background work

- GM cache preload (`start_preload_gm_caches`): `tokio::spawn` with 1.5s delay — warms V2 projections builder, lineups, rosters. Toggle `PRELOAD_GM_CACHES`.
- Prestart logger (`app/routes.py:2130`): tokio interval loop, ET clock, `_build_games_for_date`, Kelly-0.3 fractions, append-only `prestart_snapshots.csv` in `PRESTART_DIR`/`XG_CACHE_DIR`. Toggle `PRESTART_LOGGER`.
- Admin jobs: `tokio::process::Command` → `python scripts/update_data.py ...` / `scripts/lineups.py --all`, job status JSON persisted in `{cache}/nhl_admin_jobs/{uuid}.json` (multi-worker safe, same as `_read_job`).
- Preload of xG models at startup (`XG_PRELOAD=1`, non-blocking).

### 3.8 Scripts & cron jobs (Python stays)

- Training/export/backfill scripts (`xG_*_model.py`, `Game_Projection_Model*.py`, `rapm.py`, `export_*`, `lineups.py`, `estimate_gp.py`, `run_simulations.py`, ...) **remain Python and unchanged**. The Rust app never trains or fits anything.
- **Cross-dependency to change:** `scripts/update_data.py` + `backfill_shifts_season.py` currently reach the app via `create_app().test_client()` (in-process). When the Rust service is deployed, add `APP_INTERNAL_BASE_URL` support to these scripts so they call the deployed service over HTTP (`requests`). Keep the test_client path as fallback — both apps keep working.
- Render cron jobs run **Python commands inside the web-service runtime** → the Rust Dockerfile's runtime stage includes Python 3.11 + `requirements.txt` (or a trimmed runtime subset: `requests`, `beautifulsoup4`, `pandas`, `joblib`, `numpy`, `gspread`, `sqlalchemy`, `mysql-connector-python`, `supabase`, `python-dotenv`). This is the main reason for not using a scratch/static runtime image.

### 3.9 Numerics

- Normal CDF via `libm::erf` (identical formula to `math.erf` usage), Poisson PMF with precomputed log-factorials (copy `_LOG_FACT` pattern), Knuth `_poisson_draw`, deterministic `rand` RNG seeded like `random.Random(seed)` for simulations.
- Parsing helpers to port exactly: `_parse_locale_float` (comma-decimal), `_safe_int`, `_safe_float` (non-finite→None), `_safe_val` (numpy scalar semantics), season-id normalization (`_normalize_season_id_list`, September boundary `current_season_id`).
- Timezone handling: `chrono-tz` `America/New_York` for ET dates; UTC otherwise.

### 3.10 Static assets & compression

- Serve `app/static/` with `tower-http::services::ServeDir`; the Docker image embeds or copies it. `lineups_all.json` is written at runtime by cron into the same folder (Render FS is ephemeral — same behavior as today).
- `CompressionLayer` with `min_size`/`level` from `COMPRESS_MIN_SIZE`/`COMPRESS_LEVEL` envs, enabled by `HTTP_COMPRESS`.

---

## 4. Module layout (`nhl_rust/`)

```
nhl_rust/
├── PORT_PLAN.md                ← this document
├── README.md
├── Cargo.toml
├── Dockerfile                  (multi-stage: rust build → python:3.11-slim runtime)
├── .dockerignore
├── render.yaml                 (service spec for the new Render web service, incl. env vars + cron jobs)
├── rust-toolchain.toml
├── templates/                  (copied from ../app/templates at build; dev reads ../app/templates)
├── src/
│   ├── main.rs                 (config, state, router, preload spawns)
│   ├── config.rs               (all env vars, defaults identical to Flask)
│   ├── state.rs                (AppState: caches, http client, supabase, xg models, stripe)
│   ├── error.rs                (ApiError → JSON 4xx/5xx + minimal bot responses)
│   ├── cache.rs                (moka wrappers mirroring _cache_get/_set/_prune_ttl_and_size/_dict_set_bounded)
│   ├── disk_cache.rs           (XG_CACHE_DIR layout: pbp_{id}.json, shifts_{id}.json, admin jobs, prestart csv)
│   ├── util/
│   │   ├── dates.rs            (ET/UTC now, ISO parse/format, season ids, game-id parsing)
│   │   ├── parse.rs            (_safe_int/_parse_locale_float/_safe_float, strength parsing)
│   │   ├── math.rs             (erf CDF, Poisson pmf/draw, logistic, Kelly-03, bisect percentiles)
│   │   └── csv_util.rs         (streaming CSV readers, col_map renames, utf-8-sig/BOM handling)
│   ├── supabase/
│   │   ├── client.rs           (reqwest pool, headers, error semantics)
│   │   ├── read.rs             (_sb_read port: filters, col_map, pagination)
│   │   ├── write.rs            (upserts/delete with on_conflict + column-drop retry)
│   │   └── auth.rs             (GoTrue admin + password sign-in ports)
│   ├── nhl/
│   │   ├── client.rs           (shared get_json with timeouts/redirect policy)
│   │   ├── gamecenter.rs       (boxscore, right-rail, play-by-play, landing highlights)
│   │   ├── stats_rest.rs       (cayenneExp builders: summary/bios/team/summaryshooting/standings)
│   │   ├── edge.rs             (4 edge metric endpoints + url substitution)
│   │   ├── schedule.rs         (schedule/{date}, schedule/now, club-schedule-season, partner-game odds)
│   │   ├── roster.rs           (roster current/season)
│   │   ├── standings.rs        (multi-variant standings + REST fallback)
│   │   ├── bracket.rs          (playoff-bracket)
│   │   ├── images.rs           (logo/headshot proxy w/ host whitelist + SVG normalization)
│   │   └── shifts_html.rs      (TV/TH.HTM BeautifulSoup → scraper port)
│   ├── data/
│   │   ├── teams.rs            (Teams.csv/Supabase, TEAM_ROWS, Active filter)
│   │   ├── boxid.rs            (BoxID.csv map, O/D mirroring)
│   │   ├── seasonstats.rs      (season_stats + team aggregates + _build_seasonstats_agg + career matrix)
│   │   ├── rapm.rs             (rapm/context static loaders, rates synthesis, JSONB unpack)
│   │   ├── lineups.rs          (_load_lineups_all + gp_est merge + injuries)
│   │   ├── projections.rs      (V1 sheets fallback, V2 builder, EVPP coefficients, GP weighting)
│   │   ├── player_names.rs     (players table + bios maps + rosters)
│   │   ├── odds.rs             (odds_snapshots reader + payload builder + partner odds)
│   │   └── card_defs.rs        (card_metrics.csv defs)
│   ├── models/
│   │   ├── xg.rs               (UBJ load, feature cache, one-hot vectorizer, predict+avg, ENA logistic)
│   │   └── windows.rs          (window_filenames_for_season port)
│   ├── web/
│   │   ├── router.rs           (route registration, blueprints → modules)
│   │   ├── templates.rs        (minijinja env, url_for shim, request context, auth injection)
│   │   ├── session.rs          (signed cookie session, auth_user record, _auth_state_from_record)
│   │   ├── csrf.rs             (token gen/validate)
│   │   ├── premium.rs          (before_request gating port, GM public allowlist, crawler handling)
│   │   └── middleware.rs
│   ├── routes/
│   │   ├── pages.rs            (18 page routes + robots/sitemap/favicon)
│   │   ├── auth_account.rs     (login/signup/logout/account/*)
│   │   ├── stripe.rs           (checkout/portal/donate/webhook/sync)
│   │   ├── admin.rs            (users mgmt, admin jobs, db-check, lineups upload)
│   │   ├── community.rs        (community posts)
│   │   ├── line_tool.rs        (players/data/wowy/versus/lines)
│   │   ├── projections_api.rs  (series/games/team-season-points[+custom]/simulate[-batch]/custom-lineups-cache)
│   │   ├── gm_mode.rs          (all-teams-custom, V2 builder endpoints, roster proxy)
│   │   ├── skaters.rs          (players/card/table/scatter/edge/shooting/current-projections/trend)
│   │   ├── goalies.rs          (players/card/table/scatter/series/goaltending)
│   │   ├── teams.rs            (card/table/scatter/current-projections)
│   │   ├── rapm_api.rs         (player/scale/career)
│   │   ├── game.rs             (boxscore/right-rail/pbp/shifts + game page)
│   │   ├── pbp.rs              (play-by-play pipeline, the biggest module)
│   │   └── misc_api.rs         (lineups/all, projections/sheets, odds history, seasons, standings, live-games, diag, logos)
│   └── jobs/
│       ├── runner.rs           (admin subprocess jobs + persistence)
│       ├── prestart.rs         (prestart snapshot logger)
│       └── preload.rs          (XG_PRELOAD + GM cache preload)
└── tests/
    ├── parity/                 (recorded Flask fixtures → replay against Rust)
    ├── math.rs                 (CDF/Poisson/coefficients unit tests)
    └── pbp.rs                  (xG parity vs Python outputs)
```

Mapping of `app/routes.py` line ranges to modules (from §2.1):

| routes.py lines | Rust module(s) |
|---|---|
| 140–1375 (auth/session/CSRF/premium/stripe helpers) | `web/*`, `routes/stripe.rs`, `routes/auth_account.rs` |
| 1377–1837 (admin ops/jobs) | `routes/admin.rs`, `jobs/runner.rs` |
| 2233–2361 (cache/disk helpers) | `cache.rs`, `disk_cache.rs` |
| 2382–3393 (pages, community, donation) | `routes/pages.rs`, `routes/community.rs` |
| 3393–4930 (line tool) | `routes/line_tool.rs` |
| 4922–5680 (teams page, card builder, about, odds) | `routes/pages.rs`, `routes/misc_api.rs` |
| 5662–8600 (projections family) | `routes/projections_api.rs`, `routes/gm_mode.rs`, `data/projections.rs` |
| 8600–15210 (skaters/goalies/teams analytics) | `routes/{skaters,goalies,teams,rapm_api}.rs`, `data/seasonstats.rs`, `nhl/edge.rs` |
| 15325–17264 (supabase helpers, col maps, static loaders) | `supabase/*`, `data/*`, `util/*` |
| 17264–19245 (bios/rosters/lineups/odds/projection helpers) | `data/*`, `nhl/*` |
| 19245–19330 (model utils, preload) | `models/*`, `jobs/preload.rs` |
| 19330–end (game + pbp + shifts + shooting) | `routes/game.rs`, `routes/pbp.rs`, `nhl/shifts_html.rs` |

---

## 5. Endpoint port spec (grouped, with parity notes)

### 5.1 Pages + misc (M0–M1)
`/`, `/schedule`, `/live`, `/standings`, `/projections`, `/community`, `/donation`, `/skaters`, `/goalies`, `/line-tool`, `/gm-mode`, `/teams`, `/card-builder`, `/about(+/<slug>)`, `/game/<id>`, `/odds/<id>`, `/robots.txt`, `/sitemap.xml`, `/favicon.png`.
- Render with minijinja + shared context; sitemap via simple XML builder.
- `/about/<section_slug>` glossary content comes from the route (port the glossary dict).

### 5.2 Proxies & simple JSON (M1)
`/api/standings/<season>`, `/api/live-games`, `/api/seasons/<team>`, `/api/schedule/<team>/<season>`, `/api/team/<team>/<season>/schedule`, `/api/roster/<team>/current`, `/api/player/<id>/landing`, `/api/game/<id>/boxscore|right-rail`, `/api/team-logo/<abbr>.svg`, `/api/player-headshot/<id>.png`, `/api/diag/models`, `/api/player-projections/<id>`, `/api/lineups/all`, `/api/odds/history/<id>`, `/api/player-projections/sheets|league`, `/api/skaters/players`, `/api/goalies/players`.
- Straight ports; keep exact fallback chains and cache keys (stale-cache/bios fallbacks in landing).

### 5.3 Line Tool (M2)
`/api/line-tool/players|data|wowy|versus|lines`.
- Core = shifts+PBP join on `(game_id, shift_index)`; masks for WOWY (2ⁿ combos); versus grouping; team vs league combo scopes with `forward_lines`/`defense_pairings` tables and live-combo override.
- Port `_get_lt_shifts/_get_lt_pbp` batching (`LINE_TOOL_PBP_BATCH_SIZE`=50), season-state digits filter (`game_id[4:6]` in `02/03`), strength-state sets, period-5 exclusion.

### 5.4 Analytics (M2)
Cards, tables, scatters, series, edge, RAPM (player/scale/career), context player, shooting/goaltending.
- The heavy shared pieces: `_build_seasonstats_agg`, `_build_goalies_seasonstats_agg`, `_build_goalies_career_season_matrix`, `_build_team_base_stats` (REST fallback), edge URL/rank/percentile helpers, card metric resolution + percentile bisect with lower-is-better inversion.
- RAPM rates synthesis (`Totals→Rates` via context minutes, `_RAPM_VALUE_COLS`), Supabase JSONB (`stddev/zscore/pp_sh`) unpack, scale/career distributions.
- Shooting/goaltending: exact Supabase column selections (avoiding nonexistent `goalie`/`player1` text columns — repo memory), fallback team-season rebuild with parallel game loads (8 workers), coverage validation.

### 5.5 PBP + shifts + xG (M3)
See §6.1–6.3. The largest single module.

### 5.6 Projections + GM Mode (M4)
`/api/projections/series|games|team-season-points(+custom)`, `/api/projections/all-teams-custom`, `/api/projections/simulate-season(-batch)`, `/api/projections/custom-lineups-cache`, `/api/player-projections/v2`, `/api/player-current-projections(+public)`, `/api/skaters/current-projections`, `/api/skaters/player-projection-trend`, `/api/teams/current-projections`.
- Port V2 builder, exact-points Poisson convolution, series Markov DP + bracket propagation, full-season sims with deterministic seeding. All constants in Appendix A.

### 5.7 Auth + billing + admin (M5)
All `/login|signup|account*|admin/users*|admin/*|stripe/webhook|donate` routes + `_sync_auth_users_to_accounts`, `_user_management_rows`, `_backfill_missing_user_accounts`, trial/access computation, Stripe sync flows.
- Premium gating middleware must be byte-compatible: 401 `auth_required`+`loginUrl`, 403 `trial_expired`+`accountUrl`, crawler minimal-bot responses, GM public allowlist.

### 5.8 Community + prestart (M5)
`/api/community/posts` (Supabase-first, HTML scrape fallback), prestart logger thread, `/admin/prestart-snapshots`.

---

## 6. Hard-part deep dives

### 6.1 Play-by-play pipeline (`app/routes.py:19474–20460`)

Stages to port in order (function/line refs):

1. **Cache lookup** — disk JSON (`{XG_CACHE_DIR}/pbp_{id}[_{scope}].json` with `_cachedAt`+`gameState`, live TTL 5s) → memory `_PBP_CACHE` (scope = xgModel or `{model}_lite`).
2. **Fetch** — `/v1/gamecenter/{id}/play-by-play` (25s timeout); `FETCH_BIOS=1` → skater bios `shoots` map; goal highlight URLs from `/v1/gamecenter/{id}/landing` (`summary.scoring[].goals[] → highlightClipSharingUrl`).
3. **Orientation** — sum x-coords of typeCodes 505–508 per (period, team) to orient shots so offensive zone is +x.
4. **Normalization** — `parse_time_to_seconds`; `strength_from_situation` (leading/trailing `0` ⇒ ENF/ENA); `remap_strength_state` (5V4→PP1 etc.); `compute_boxid2` (O-box handedness variants); shot geometry `(89,0)` net, `ShotDistance/ShotAngle`; `ScoreState`/`ScoreState2` (±3 clip); `EventIndex = gameId*10000 + i+1`; corsi/fenwick/shot flags; period 5 (shootout) excluded.
5. **Shift join** (skipped in `lite`) — call shifts endpoint in-process; `slices` sorted by start; `onice_cache[ShiftIndex]` → Home/Away `Forwards_ID` (space-joined, sorted), `Defenders_ID`, `Goalie_ID` + name variants; `bisect_left` boundary rule (faceoff/period-start take the later slice).
6. **Wide row mapping** — player1/2/3 priority order (`scoringPlayerId, shootingPlayerId, playerId, hittingPlayerId, hitteePlayerId, assist1/2PlayerId, blockingPlayerId, losingPlayerId, winningPlayerId, committedByPlayerId, drawnByPlayerId`); BoxID join via `_get_boxid_map` with coordinate clamping to ±100/±42 int grid; `RinkVenue="Season-HomeAbbrev"`; `LastEvent` = Rebound/Quick via 4s Fenwick lookback.
7. **xG** — per requested family: ENA first (xG_S=1.0; fenwick logistic with fixed coefficients); else windowed models (§6.3); numpy-batch equivalent via `DMatrix` predict + sigmoid; average across windows.
8. **Sanitize + cache** — `_safe_val`/`_sanitize_row`; store memory + disk.

### 6.2 Shifts HTML parsing (`app/routes.py:20472–21370`)

- Fetch `TV{suffix}.HTM` + `TH{suffix}.HTM` (boxscore only for roster indices + gameState).
- Port the three parsing strategies (content table, player-header scan, regex fallback), name normalization (`LAST, FIRST` flip, diacritics, suffixes), jersey→playerId resolution.
- Slicing: `gameTime = (Period-1)*1200 + secs`, unique boundaries, `ShiftIndex = gameId*10000 + i+1`.
- `_normalize_strength_state` + `StrengthStateBucket` logic must be ported exactly (ENF/ENA from goalie presence, `(6,5)→5v5`, PP/SH mapping, 4v4/3v3 even states, Other bucket).
- Use `scraper` crate with CSS selectors; keep a recorded-fixture test suite (HTML samples) to lock behavior.

### 6.3 xG model conversion & inference parity

**Conversion script** (`scripts/export_models_for_rust.py`, Python, run once + on any retrain):

```python
for fname in Model/xg*_*.pkl:
    m = joblib.load(fname)
    booster = m.get_booster()
    booster.save_model(f"Model/rust/{fname}.ubj")          # XGBoost UBJSON
    json.dump({
        "features": list(m.feature_names_in_),            # one-hot column names
        "objective": "binary:logistic",
        "base_score": float(m.get_booster().attr('base_score') or 0.5),
        "prefix": prefix, "start": s1, "end": s2,
    }, open(f"Model/rust/{fname}.meta.json", "w"))
```

**Rust inference** (`models/xg.rs`):
- Load `Booster::load_model` per window (lazy + `MODEL_CACHE_MAX_ITEMS`=24 bounded).
- Feature vector: 7 base cols `["Venue","shotType2","ScoreState2","RinkVenue","StrengthState2","BoxID2","LastEvent"]`; each model feature is `base_suffix`; set 1.0 on match (`'missing'` for missing values) — direct port of `_vectorize_row_for_model`.
- Predict batches through `DMatrix` → margins → sigmoid; average across `XG_WINDOWS` (default 1, max 3) in preference order `[1,0,2]`; special season `20252026 → {prefix}_20222023_20242025.pkl`.
- **Parity test**: run conversion script; for ~200 sampled PBP rows, compare Python `predict_proba` vs Rust outputs; assert |Δ| < 1e-6.

### 6.4 SeasonStats aggregation & career matrix

- `_build_seasonstats_agg` (`:16198`): streaming read of Supabase `season_stats` (or `app/static/nhl_seasonstats.csv`, tens of MB) → per-player sums across selected season/state/strength filters, with position group split (F/D) for percentiles.
- `_build_goalies_career_season_matrix` (`:16423`): per-goalie-per-season `{TOI,FA,SA,GA,xGA_S,xGA_F,xGA_F2}` + league `{season: (SA, GA)}`.
- Both cached in-memory (1800s); the Rust versions replace gzip+pickle disk caches with a binary format (e.g. `postcard`+`flate2`) under **new file suffixes** so Python and Rust never read each other's pickles.

### 6.5 V2 projections math (port constants verbatim, Appendix A)

Value formula per strength state + GP weighting (`gp/41` when `0<gp<41`) + `raw_projected_value`; win-prob via Normal CDF; exact expected points via 31×31 Poisson convolution; series DP; full-season sim with B2B situation offsets, OT bonus 0.30 flip, scorer simulation (weighted choice over `ig/a1/a2`, assist probabilities).

### 6.6 Auth/Stripe/Admin flows

Port the exact state machine: trial (14d) → `has_access = is_admin or subscription_status in {active,paid} or trial_active`; `_sync_user_account_from_supabase_user` (DB row wins); Stripe `checkout.session.completed` + `customer.subscription.*` sync with user resolution (metadata → customer/subscription id scan); no event-id dedup (upsert-again semantics is the current behavior — preserve it); CSRF enforcement list exactly as today.

---

## 7. Cache & env-var parity

**Keep every env var name and default identical.** The full list is in the research notes; highlights that gate behavior:

| Env | Default | Purpose |
|---|---|---|
| `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `SUPABASE_ANON_KEY`, `SUPABASE_PUBLISHABLE_KEY` | — | data + auth |
| `FLASK_SECRET_KEY`/`SECRET_KEY` | dev fallback | session signing (reused name for cookie key) |
| `XG_PRELOAD`, `PRELOAD_GM_CACHES`, `PRESTART_LOGGER`, `HTTP_COMPRESS`, `COMPRESS_MIN_SIZE` (512), `COMPRESS_LEVEL` (6) | 1/1/1/1 | startup + compression |
| `XG_DISABLED`, `XG_WINDOWS` (1), `FETCH_BIOS` (0), `MODEL_CACHE_MAX_ITEMS` (24), `FEATURE_COLS_CACHE_MAX_ITEMS` (512) | | models |
| `XG_CACHE_DIR`, `PRESTART_DIR`, `PRESTART_CSV`, `PRESTART_WINDOW_SECONDS` (3600), `PRESTART_GRACE_SECONDS` (300) | OS temp | disk cache + prestart |
| `PBP_CACHE_TTL_SECONDS` (600), `SHIFTS_CACHE_TTL_SECONDS` (600), `BOX_CACHE_TTL_SECONDS` (600) + `_MAX_ITEMS` variants | | game caches |
| `LINE_TOOL_PBP_BATCH_SIZE` (50), `LINE_TOOL_DATA_CACHE_TTL_SECONDS` (1800) | | line tool |
| `PLAYER_PROJECTIONS_CACHE_TTL_SECONDS` (300), `GM_PROJECTIONS_CACHE_TTL_SECONDS` (300), `LINEUPS_SHEET_CACHE_TTL_SECONDS` (300), `CLUB_SCHEDULE_CACHE_TTL_SECONDS` (21600), `ODDS_SNAPSHOTS_CACHE_TTL_SECONDS` (60), `CUSTOM_LINEUPS_CACHE_TTL_SECONDS` (43200) | | projections family |
| `SKATERS_SHOOTING_CACHE_TTL_SECONDS` (180), `GOALIES_GOALTENDING_CACHE_TTL_SECONDS` (180), `TEAM_SEASON_PBP_FALLBACK_WORKERS` (8) | | shooting |
| `EDGE_API_CACHE_TTL_SECONDS` (3600), `SEASONSTATS_AGG_CACHE_TTL_SECONDS` (1800), `RAPM_SCALE_CACHE_TTL_SECONDS` (300), `RAPM_CAREER_CACHE_TTL_SECONDS` (300), `SEASONS_CACHE_TTL_SECONDS` (3600) | | analytics |
| `COMMUNITY_NHL_HUB_ID`, `COMMUNITY_SITE_URL`, `COMMUNITY_FEED_CACHE_TTL_SECONDS` (300) | | community |
| `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_PRICE_MONTHLY_ID`, `STRIPE_PRICE_YEARLY_ID`, `APP_BASE_URL` | | billing |
| `DATABASE_URL(_RO/_RW)`, `DB_*`, `DB_SSL_*` | | admin preflight |
| `APP_INTERNAL_BASE_URL` | **new** | lets `scripts/update_data.py` call the Rust service over HTTP |

---

## 8. Testing & parity strategy

1. **Fixture recorder** (Python, dev-only): hit the Flask test client for every route × representative params; store request + status + headers + body in `nhl_rust/tests/fixtures/` (normalized: timestamps scrubbed).
2. **Parity runner** (Rust tests): replay fixtures against the Rust app with upstream HTTP mocked (recorded NHL/Supabase responses) or with envs pointed at the same live services in a dev profile; assert structural equality + float tolerance (1e-6 for xG, 1e-9 for probabilities).
3. **Unit tests** for math ports (CDF, Poisson, Kelly, EVPP formula, series DP, bracket propagation) against Python-computed goldens.
4. **PBP/xG golden test**: 200 rows, Python `predict_proba` vs Rust, per §6.3.
5. **Smoke suite** port of `tests/test_smoke_app.py` (pages, auth redirects, CSRF, admin validation).
6. **CI** (`.github/workflows/nhl-rust-ci.yml`): `cargo fmt --check`, `clippy -D warnings`, `cargo test`, release build.

---

## 9. Milestones

| Milestone | Scope | Acceptance criteria |
|---|---|---|
| **M0 Scaffold** | Axum skeleton, config, state, error model, cache/disk-cache modules, supabase read client, minijinja + static serving, compression, Dockerfile + render.yaml draft | Binary boots; `/`, `/about` render; static assets + compression served; env parity validated |
| **M1 Pages + proxies** | §5.1 + §5.2 (18 pages + 16 proxy/simple APIs) | Parity fixtures pass for all M1 routes |
| **M2 Analytics** | §5.3 line tool + §5.4 cards/tables/scatter/series/edge/RAPM/shooting | Parity fixtures pass; shooting/goaltending Supabase-first path byte-identical |
| **M3 Game core** | §6.1 PBP pipeline, §6.2 shifts HTML, §6.3 model conversion + xG | xG parity < 1e-6; shifts parsing fixture suite green |
| **M4 Projections + GM** | §5.6 endpoints + simulations | Sim outputs match Python within seed/tolerance contract |
| **M5 Auth + billing + ops** | §5.7 auth/account/stripe/admin + community + prestart logger | ✅ DONE (2026-08-14): full Supabase GoTrue auth + session/CSRF + premium gating, account page + actions, admin user management, raw-HTTP Stripe checkout/portal/donation/webhook with HMAC verification, community posts API (Supabase-first + HTML fallback), prestart logger + /admin/prestart-snapshots. 43/43 tests green. |
| **M6 Hardening + cutover** | Perf (moka tuning), request tracing, load test vs Flask, Render blue-green, update_data HTTP mode, docs | Render service `nhl-rust` live; Flask kept as fallback |

Rough sizing per milestone: M0 ~2–3d, M1 ~3d, M2 ~1.5–2w, M3 ~1.5–2w, M4 ~1w, M5 ~1w, M6 ~1w.

---

## 10. Rollout on Render

1. Add `nhl_rust/render.yaml` as a **second web service** (`nhl-rust`, Docker runtime) with the same env vars + the two cron jobs (lineups 30m, daily update) — cron commands remain Python and run in the runtime image.
2. First cutover option: change root `render.yaml` service to point at `nhl_rust` (after M6); alternatively create a new Render service from the dashboard to keep the Flask service alive.
3. Update `scripts/update_data.py` + `backfill_shifts_season.py` to use `APP_INTERNAL_BASE_URL` (HTTP) with test_client fallback.
4. Keep the Flask service on a "canary" for 1–2 weeks; compare error rates/timing.

---

## 11. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| XGBoost crate build/link issues on Render | Medium | Pure-Rust JSON-dump tree evaluator fallback (kept as feature flag) |
| Numeric drift in sims/probabilities | Medium | Golden fixtures + seeded RNG contract (same algorithm, same seed sequence) |
| `scraper`-based HTML shift parsing subtle mismatches | Medium | Recorded HTML fixture suite from real games |
| Supabase/Python quirks (column-drop upsert retry, None-vs-empty semantics) | Medium | Encapsulated in one module with unit tests |
| Cron/scripts need Python in runtime image | Low | Dockerfile runtime stage includes Python 3.11 + requirements |
| Template drift between two apps during transition | Low | Single template source at `app/templates`, symlinked/copied into Rust build |
| Behavioral parity gaps (error codes, cache headers) | Medium | Parity fixture suite must pass before cutover; keep `Cache-Control` values identical |

---

## 12. Open questions for the user

1. **Model conversion timing** — OK to add `scripts/export_models_for_rust.py` + a `Model/rust/` output folder (new files only, Flask untouched)?
2. **Render topology** — second service `nhl-rust` alongside `nhl-analytics` (blue-green), or replace the service definition in `render.yaml` at cutover?
3. **Cron jobs** — keep the two Python cron jobs attached to whichever service is live; the Rust image includes Python to support this. Confirm that's acceptable vs moving crons to a separate worker service.
4. **Priority order** — default is M0→M6 as listed; if there's a bottleneck you care most about (e.g., Line Tool or Game pages), I can reorder.

---

## Appendix A — Port-critical constants

**V2 projection coefficients (`_EVPP_COEF`, `app/routes.py:5855`)**
```
poss_value_ev  = 1.315740      poss_value_st = 1.952908
xga_ev         = -0.191099     xga_sh        = -0.284162
xgf_pp         = 0.127149      off_the_puck_sh = 0.272929
gax            = 0.232042      gsax          = 0.529976
rookie (5v5):  F -0.018529     D -0.026125   G -0.419905
```
**V2 win/points math:** conservative weight `0.7`; `mu = 0.7*(home_proj - away_proj + sit)`; `lg = _V2_LG_AVG[season]` (20192020→2.9505 … 20252026→3.0724, default 3.0); `gf_home = max(0.5, lg+mu/2)`, `gf_away = max(0.5, lg−mu/2)`; `p_home = Φ(mu/√(gf_home+gf_away))`; goals ~ Poisson; tie → OT/SO winner with `p_home`; 1-goal regulation win flips to OT with p=0.30.

**V2 situation map (`_V2_SITUATION`)**: (homeB2B, awayB2B) → `(0,0)=0.181937`, `(0,1)=0.539297`, `(1,0)=−0.196136`, `(1,1)=0.247761`.

**Game-projection SITUATION logistics (`app/routes.py:6863–6868`)**: Away-B2B-B2B `−0.126602018`; Away-B2B-Rested `−0.400515738`; Away-Rested-B2B `0.174538991`; Away-Rested-Rested `−0.153396566`.

**Empty-net fenwick logistic:** `1/(1+exp(0.013609495·d + 0.023174225·|a| − 1.97392131))`.

**Playoffs:** home pattern `(T,T,F,F,T,F,T)` (series sim uses `[top, top, bottom, bottom, top, bottom, top]`); rested-rested situation `−0.153396566`; 2-2-1-1-1 assumption; bracket slots A–O with `I←A/B, J←C/D, K←E/F, L←G/H, M←I/J, N←K/L, O←M/N`.

**Simulation floors/assists:** `_SIM_FLOOR` F `{ig 0.030, a1 0.050, a2 0.050}`, D `{0.012, 0.025, 0.025}`; assist counts: two 0.65 / one 0.20 / zero 0.15.

**Rookie fallback (`_ROOKIE_FALLBACK`)**: D `−0.031768511`, F `−0.024601581`, G `−0.12`.

**Line tool strength sets:** 5v5→{5v5}; PP→{5v4,5v3,4v3}; SH→{4v5,3v5,3v4}; Other→complement of the 7 special states.
