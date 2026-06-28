# NHL Project Overview

This document is a practical guide to the NHL analytics app in this repository. It is meant to help future chats, new contributors, and anyone trying to find the right place to make changes.

The app is a Flask web application that serves HTML pages and JSON APIs for NHL analytics. It combines live NHL data, stored play-by-play and shift data, xG and projection models, Supabase-backed auth and user data, and a set of front-end tools for exploring player, team, and game information.

## What the App Does

At a high level, the app provides:

- A public home page and navigation hub.
- Schedule, live game, standings, and game detail views.
- Player pages for skaters and goalies with advanced metrics, charts, and tables.
- Team pages with team cards, tables, scatter plots, and projection views.
- A Line Tool for on-ice analysis, WOWY, Versus, forward lines, and defense pairs.
- A projections area for game predictions, playoff series simulations, bracket views, and Stanley Cup probabilities.
- A card builder for creating social-ready stat graphics.
- Auth, billing, free trial, and account management backed by Supabase Auth and Stripe.
- Admin workflows for user management and data jobs.
- A lightweight community page fed from a Supabase-backed community service.

## First Things to Know

If you are new to this repo, these points matter more than anything else:

1. Most of the application logic lives in one file: `app/routes.py`.
2. The HTML pages are mostly shells in `app/templates/` that fetch JSON from `/api/...` routes.
3. `app/__init__.py` creates the Flask app and registers a single blueprint from `app/routes.py`.
4. Production runs through Gunicorn and `wsgi.py`; local development usually runs through `run.py`.
5. This app relies on several data sources at once: NHL APIs, Supabase, local/static files, and model files under `Model/`.

## Entry Points and Runtime

### App startup

- `run.py`
  - Local development entry point.
  - Loads `.env`, calls `create_app()`, and runs Flask in debug mode.

- `wsgi.py`
  - Production entry point.
  - Used by Gunicorn on Render.

- `app/__init__.py`
  - Defines `create_app()`.
  - Configures the Flask secret key and session lifetime.
  - Registers the main blueprint from `app/routes.py`.
  - Optionally preloads common xG models at startup.
  - Optionally starts a background prestart snapshot logger.

### Deployment

- `Procfile`
  - Production process: `gunicorn --bind 0.0.0.0:$PORT wsgi:app`

- `render.yaml`
  - Render deployment configuration.
  - Auto-deploy is enabled.
  - Also defines cron jobs for lineup updates and daily data refresh.

### Important runtime behavior

- The app uses module-level caches in `app/routes.py`.
- Some expensive responses also use an on-disk cache under `%TEMP%/nhl_cache` on Windows or `/tmp/nhl_cache` on Linux unless overridden by `XG_CACHE_DIR`.
- Startup model preload can be disabled with `XG_PRELOAD=0`.
- Scripts that reuse the Flask app typically set `XG_PRELOAD=0` so they do not eagerly load models.

## Main Code Layout

### Core application files

- `app/routes.py`
  - Main HTTP surface.
  - Route handlers, API logic, parsing helpers, caching, model loading, auth flows, billing hooks, and admin jobs.
  - This is the main place to start when debugging behavior.

- `app/supabase_client.py`
  - Supabase helper layer.
  - Used for auth, user accounts, layouts, and community-related data.

- `app/templates/`
  - Jinja templates for each page.
  - Front-end UI logic is often embedded directly in the template.

- `app/static/`
  - Static images, generated JSON, CSV fallbacks, and browser-side JS such as `card_builder.js`.

- `Model/`
  - Serialized xG and projection model artifacts.
  - These are loaded from `app/routes.py` through model-loading helpers.

- `scripts/`
  - Data refresh, exports, backfills, health checks, and model training scripts.

- `supabase/migrations/`
  - Schema definition and evolution for the Supabase/Postgres side.

- `tests/`
  - Smoke tests and a few broader validation scripts.

## User-Facing Features and Where They Live

This section is the quickest way to map a visible feature to the code that drives it.

### Home and site navigation

- Route: `/`
- Template: `app/templates/home.html`
- Route handler: `index_page()` in `app/routes.py`
- Purpose:
  - Landing page.
  - Acts as a hub for Hockey-Statistics properties and external links.

### Schedule

- Route: `/schedule`
- Template: `app/templates/index.html`
- Main route handler: `schedule_page()`
- Related APIs:
  - `/api/team/<team_code>/<season>/schedule`
  - `/api/schedule/<team_code>/<season>`
- Purpose:
  - Browse games, schedules, and completed results.
  - Link into per-game detail pages.

### Live games

- Route: `/live`
- Template: `app/templates/live.html`
- Main API: `/api/live-games`
- Purpose:
  - Show in-progress games and live game state.

### Standings

- Route: `/standings`
- Template: `app/templates/standings.html`
- Main API: `/api/standings/<season>`
- Purpose:
  - Display standings with analytics context.
  - Uses seasons derived from `Last_date.csv` to seed the UI.

### Skaters

- Route: `/skaters`
- Template: `app/templates/skaters.html`
- Key APIs:
  - `/api/skaters/players`
  - `/api/player/<player_id>/landing`
  - `/api/skaters/card`
  - `/api/skaters/table`
  - `/api/skaters/scatter`
  - `/api/skaters/edge`
  - `/api/rapm/player/<player_id>`
  - `/api/context/player/<player_id>`
  - `/api/skaters/current-projections`
  - `/api/skaters/player-projection-trend/<player_id>`
  - `/api/skaters/shooting`
- Purpose:
  - Main skater profile and comparison experience.
  - Supports card views, league tables, scatter plots, RAPM/context views, and current projection overlays.

### Goalies

- Route: `/goalies`
- Template: `app/templates/goalies.html`
- Key APIs:
  - `/api/goalies/players`
  - `/api/goalies/card`
  - `/api/goalies/table`
  - `/api/goalies/scatter`
  - `/api/goalies/series`
  - `/api/goalies/goaltending`
- Purpose:
  - Main goalie analytics surface.
  - Includes GSAx-style views, tables, charts, and goalie-specific rate breakdowns.

### Teams

- Route: `/teams`
- Template: `app/templates/teams.html`
- Key APIs:
  - `/api/teams/card`
  - `/api/teams/table`
  - `/api/teams/scatter`
  - `/api/teams/current-projections`
- Purpose:
  - Team-level card view, comparative tables, charts, and current projection context.

### Line Tool

- Route: `/line-tool`
- Template: `app/templates/line_tool.html`
- Key APIs:
  - `/api/line-tool/players`
  - `/api/line-tool/data`
  - `/api/line-tool/wowy`
  - `/api/line-tool/versus`
  - `/api/line-tool/lines`
- Purpose:
  - On-ice combination analysis.
  - Heat maps.
  - WOWY splits.
  - Versus team/player splits.
  - Forward line and defense pair aggregation.
- Important current behavior:
  - KPI/WOWY work from shifts where selected players are together.
  - Forward Lines and Defense Pairs use exact combo grouping from shifts.
  - Versus constraints now apply across tabs when an against team is selected.

### GM Mode

- Route: `/gm-mode`
- Template: `app/templates/gm_mode.html`
- Access: admin-only via `_require_admin_page()`
- Purpose:
  - Admin workflow for lineup management and roster/depth-chart review.
  - Closely related to projections data and lineup feeds.

### Card Builder

- Route: `/card-builder`
- Template: `app/templates/card_builder.html`
- Browser-side JS: `app/static/card_builder.js`
- Layout APIs:
  - `/api/card-builder/layouts` (GET/POST)
  - `/api/card-builder/layouts/<layout_id>` (DELETE)
- Definition APIs:
  - `/api/skaters/card/defs`
  - `/api/goalies/card/defs`
  - `/api/teams/card/defs`
- Purpose:
  - Build 16:9 social cards using skater, goalie, team, and GM-mode content.
  - Save drafts and account-linked layouts.

### Projections

- Route: `/projections`
- Template: `app/templates/projections.html`
- Key APIs:
  - `/api/projections/games`
  - `/api/projections/series`
  - `/api/projections/team-season-points`
  - `/api/projections/team-season-points-custom`
  - `/api/projections/all-teams-custom`
  - `/api/projections/custom-lineups-cache`
  - `/api/lineups/all`
  - `/api/player-current-projections`
- Purpose:
  - Game-level win probabilities.
  - Playoff series simulations.
  - Bracket and Stanley Cup probability outputs.
  - Custom lineup what-if workflows.
- Important current behavior:
  - Current projections are driven by current-season player projection tables rather than only legacy static CSV data.
  - `app/static/lineups_all.json` is still an important input for lineup-related views.

### Game detail and odds

- Route: `/game/<game_id>`
- Template: `app/templates/game.html`
- Key APIs:
  - `/api/game/<game_id>/boxscore`
  - `/api/game/<game_id>/right-rail`
  - `/api/game/<game_id>/play-by-play`
  - `/api/game/<game_id>/shifts`
- Odds page:
  - `/odds/<game_id>`
  - Template: `app/templates/odds.html`
  - API: `/api/odds/history/<game_id>`
- Purpose:
  - Detailed per-game reporting, lineups, boxscore, play-by-play, and shift-based analysis.

### Community

- Route: `/community`
- Template: `app/templates/community.html`
- API: `/api/community/posts`
- Purpose:
  - Show community posts for the NHL hub.
  - Pulls community data through Supabase helpers.

### About and glossary

- Routes:
  - `/about`
  - `/about/<section_slug>`
- Template: `app/templates/about.html`
- Purpose:
  - Explain metrics, models, glossary terms, and app concepts.

### Donation

- Route: `/donation`
- Template: `app/templates/donation.html`
- Related POST routes:
  - `/donate`
  - `/account/donate`
- Purpose:
  - Support future development through donation flows.

### Auth, account, billing, and admin

- Templates:
  - `app/templates/login.html`
  - `app/templates/signup.html`
  - `app/templates/account.html`
  - `app/templates/user_management.html`
  - `app/templates/update.html`
- Key routes:
  - `/login`
  - `/signup`
  - `/logout`
  - `/account`
  - `/account/plan`
  - `/account/billing`
  - `/account/profile`
  - `/account/password`
  - `/account/delete`
  - `/account/unsubscribe`
  - `/stripe/webhook`
  - `/admin/users`
  - `/admin/run-update-data`
  - `/admin/run-lineups`
  - `/admin/job/<job_id>`
- Purpose:
  - Supabase Auth login and signup.
  - Free-trial and premium access flows.
  - Stripe billing and subscription sync.
  - Admin user management and operational job controls.

## Data Sources and Persistence

The app does not rely on a single storage source. It mixes several systems.

### NHL upstream APIs

- The app calls NHL endpoints directly with `requests`.
- These feeds are used for schedules, games, live data, and upstream game structures.
- Parsing must be tolerant of schema drift because the upstream payloads are not controlled by this repo.

### Supabase

Supabase is used for at least three major concerns:

1. Auth and user account state.
2. Layout/account persistence for card builder and related user data.
3. Community and projection-related tables.

Relevant files:

- `app/supabase_client.py`
- `supabase/migrations/001_create_schema.sql`

### CSV and static fallbacks

These files still matter:

- `Teams.csv`
  - Basic team metadata input.
- `Last_date.csv`
  - Used in standings and date/season-related flows.
- `app/static/lineups_all.json`
  - Generated lineup snapshot used by projections and lineup-related tooling.
- `app/static/player_projections.csv`
  - Legacy fallback; not the primary current projection source anymore.
- `app/static/zones.json` and rink assets
  - Used by shot/zone and heat map features.

### Model artifacts

- `Model/`
  - Pretrained xG, win probability, and projection model outputs.
- Loaded through helpers in `app/routes.py`.

## Scripts and Batch Jobs

The `scripts/` folder is operationally important. It is not just a collection of one-off experiments.

### High-value scripts

- `scripts/update_data.py`
  - Reuses the Flask app through `create_app().test_client()`.
  - Fetches and normalizes game data using the same logic as production routes.
  - Used in production cron-like workflows.

- `scripts/lineups.py`
  - Scrapes Daily Faceoff and writes `app/static/lineups_all.json`.
  - Render runs this periodically.

- `scripts/rapm.py`
  - RAPM/context export and static CSV generation.

- `scripts/export_current_player_projections.py`
- `scripts/export_preseason_updating_player_projections.py`
- `scripts/export_recent60_player_projections.py`
  - Projection export/update workflows.

- `scripts/Game_Projection_Model.py`
  - Trains or refreshes game projection models and related cached artifacts.

- `scripts/xG_F_model.py`
- `scripts/xG_F2_model.py`
- `scripts/xG_S_model.py`
  - xG model training scripts.

- `scripts/backfill_*.py`
  - Targeted repair and backfill utilities.

- `scripts/check_shifts_table_health.py`
- `scripts/inspect_shift_strength_anomalies.py`
- `scripts/normalize_shifts_strengthstate.py`
  - Useful when debugging shift-quality or strength-state issues.

## Key Templates and Their Role

This app keeps a lot of front-end behavior directly in Jinja templates. That means UI bugs are often fixed in the template rather than a separate JS bundle.

Important templates:

- `app/templates/base.html`
  - Global shell, nav, metadata, and shared layout.
- `app/templates/home.html`
  - Landing page.
- `app/templates/index.html`
  - Schedule page.
- `app/templates/live.html`
  - Live games page.
- `app/templates/standings.html`
  - Standings page.
- `app/templates/skaters.html`
  - Large front-end surface for skater analytics.
- `app/templates/goalies.html`
  - Large front-end surface for goalie analytics.
- `app/templates/teams.html`
  - Team analytics UI.
- `app/templates/line_tool.html`
  - One of the most complex front-end templates in the repo.
- `app/templates/projections.html`
  - Multi-tab projections UI.
- `app/templates/game.html`
  - Game report UI.
- `app/templates/card_builder.html`
  - Card builder shell; much of the dynamic logic is in `app/static/card_builder.js`.

## Where to Start When Editing a Feature

This section is intended to save time in future chats.

### If the problem is on a page

Start with:

1. The matching template in `app/templates/`.
2. The page route in `app/routes.py`.
3. The `/api/...` endpoints called by the page.

### If the problem is data/parsing related

Start with:

1. The relevant `/api/...` route in `app/routes.py`.
2. Shared helpers nearby in `app/routes.py`.
3. The corresponding script in `scripts/` if the issue involves exported or cached data.

### If the problem is auth or billing related

Start with:

1. `app/supabase_client.py`
2. Auth/account routes in `app/routes.py`
3. Stripe-related routes such as `/stripe/webhook`
4. Templates: `login.html`, `signup.html`, `account.html`, `user_management.html`

### If the problem is Line Tool behavior

Start with:

1. `app/templates/line_tool.html`
2. `/api/line-tool/data`
3. `/api/line-tool/wowy`
4. `/api/line-tool/versus`
5. `/api/line-tool/lines`

### If the problem is projections

Start with:

1. `app/templates/projections.html`
2. `/api/projections/games`
3. `/api/projections/series`
4. Current projection export scripts in `scripts/`
5. `app/static/lineups_all.json` if lineups are involved

### If the problem is card-builder related

Start with:

1. `app/templates/card_builder.html`
2. `app/static/card_builder.js`
3. `/api/card-builder/layouts`
4. `/api/*/card/defs`

## Auth and Access Model

The current auth stack is built on Supabase Auth.

Important rules:

- Premium gating is route-prefix based in `app/routes.py`.
- The gating only matters when auth is configured in the environment.
- Signup creates a 14-day free trial with no credit card required; trials can be upgraded to paid plans (Pro Monthly / Pro Yearly) but never require payment. When the trial ends, Pro features lock and the account reverts to the Community site experience.
- Stripe billing runs server-side and subscription state is synced back through `/stripe/webhook`.
- Auth routes also contain crawler-hardening logic and safe-next handling to avoid redirect abuse.

## Caching and Performance Notes

There are several caches in this app. Do not remove or bypass them casually.

- Module-level caches in `app/routes.py` reduce repeated expensive work.
- Disk caches persist some game payloads across restarts.
- Model preload reduces first-hit latency.
- Large pages often depend on cached payload shapes, so API changes should be made carefully.

When changing cache-sensitive logic, check whether:

1. The cache key includes the new parameter.
2. The cache needs to be invalidated or versioned.
3. The same logic is reused by scripts that call the app internally.

## Tests and Validation

The test suite is not exhaustive, but it gives a decent starting point.

Relevant files:

- `tests/test_smoke_app.py`
  - Basic page and redirect smoke coverage.
- `tests/test_league_quick.py`
- `tests/test_league_scope.py`
- `tests/test_speed.py`

Practical validation pattern used in this repo:

- Route-level validation through `create_app().test_client()` is common and effective.
- Many scripts also reuse app routes internally, so route correctness matters outside the browser UI.

## Environment Variables Worth Knowing

Common env vars you are likely to encounter:

- `FLASK_SECRET_KEY`
- `SECRET_KEY`
- `XG_PRELOAD`
- `PRESTART_LOGGER`
- `FETCH_BIOS`
- `XG_CACHE_DIR`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_KEY`
- `STRIPE_SECRET_KEY`
- `STRIPE_WEBHOOK_SECRET`
- `STRIPE_PRICE_MONTHLY_ID`
- `STRIPE_PRICE_YEARLY_ID`
- `APP_BASE_URL`
- DB-style variables such as `DATABASE_URL`, `DATABASE_URL_RO`, `DATABASE_URL_RW`, or explicit `DB_HOST` / `DB_USER` / `DB_PASSWORD` / `DB_PORT` / `DB_NAME`

## Known Conventions and Gotchas

These are useful to know before making changes:

1. `app/routes.py` is intentionally monolithic. Prefer localized edits over broad refactors unless the task truly requires structural change.
2. Many templates embed substantial JavaScript directly in the HTML file.
3. Several scripts rely on route logic by spinning up the app internally, so route changes can affect batch jobs.
4. Projection workflows increasingly prefer current projection tables and exports over legacy CSV-only flows.
5. The Line Tool is shift-key driven, so selected-team context usually defines opponent context by matching `(game_id, shift_index)` rather than mirroring strength labels on both sides.
6. The workspace VS Code task path may not exactly match the currently configured interpreter path; check local environment assumptions before debugging execution issues.

## Recommended Reading Order for Future Chats

If a future chat needs to understand the project quickly, read in this order:

1. `docs/project-overview.md` (this file)
2. `.github/copilot-instructions.md`
3. `app/__init__.py`
4. `app/routes.py`
5. The specific template for the page or feature being changed
6. The related script in `scripts/` if the feature depends on exported or refreshed data

## Short Version

If you only remember one thing, remember this:

This repo is a Flask app with a very large, route-centric backend in `app/routes.py`, and most pages are template-driven front ends that call JSON APIs in the same file. When in doubt, find the page template, find the API route it calls, and then inspect nearby helper functions and scripts that reuse the same logic.