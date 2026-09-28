# Manager Game — Plan (Rust, `nhl_rust/`)

> Status: **DRAFT PLAN — for user review. No code written yet.**
> Date: 2026-08-31
> Target: a league-style Manager game as a new feature of the Rust app in `nhl_rust/`, following the real 2026/27 NHL schedule. **No simulation** — scores come from real NHL boxscore/xG data through the existing play-by-play pipeline.

---

## 1. What we learned from GM Mode / projections (and will reuse)

The existing app gives us almost everything the game needs; the new feature is mostly *league state + rules + a scoring formula*, not new data science.

| Existing piece | Location (Rust) | How the Manager game reuses it |
|---|---|---|
| NHL schedule fetching (per-team, per-season, with shift-forward fallback for unpublished seasons) | `data/projections.rs` → `fetch_club_schedule_games` | Manager league schedule = real 2026/27 NHL schedule (`season=20262027`). The 2026/27 schedule was officially released in July 2026, season opens 2026-10-07, so the real schedule is available. |
| V2 player projections (`nhl_current_playerprojections` → per-player evo/evd/pp/sh/gax/gsax + ig/a1/a2 per-game rates, GP-weighted) | `data/projections.rs` → `build_v2_player_projections`, `load_gm_mode_projections_cached`, `proj_value_for_player` | **CPU draft AI** ("computer picks the best available player") + trade bot valuation. |
| Play-by-play pipeline: normalized events, xG_F (Fenwick xG models), shift join (on-ice players), strength states, penalty actors (`committedByPlayerId` / `drawnByPlayerId`) | `routes/pbp.rs`, `models/xg.rs` | **Game score computation** — real G/A1/A2/SOG/PENT/PEND from events, 5v5 on-ice xG± from xG_F + shift join, GSAx = xGA − GA per goalie (same formula the goalies API already uses). |
| Supabase PostgREST read + upsert patterns, GoTrue auth/sessions, premium gating | `supabase/*`, `web/*`, `routes/mod.rs` | League/roster/lineup/trade persistence + auth; new tables in a new migration. |
| Rosters (current NHL rosters), teams list, headshots/logos | `data/rosters.rs`, `data/teams.rs`, `nhl/images.rs` | Player pool for the draft; UI assets. |
| Background tokio jobs (prestart logger pattern) | `jobs/prestart.rs` | Nightly league sweep: schedule sync, postponed-game re-anchoring, finalization, stat recompute. |

Not reused: the **season simulators** (`simulate_series`, `run_single_sim`, Poisson draws…). The Manager game simulates nothing — those stay untouched for the Projections pages.

---

## 2. Game concepts (rules interpretation)

### 2.1 League

- A league has **2–32 teams**, one per NHL franchise. Each manager/user controls exactly one franchise for the whole season; empty franchises are **CPU teams**.
- Two modes, chosen at league creation:
  - **Mode 1 — True rosters:** each team starts with the franchise's real 23-man NHL roster. The manager activates a lineup of **12 F + 6 D + 2 G** per round from that roster (default = latest real NHL lineup).
  - **Mode 2 — Draft:** a **26-round snake draft** over the full NHL player pool (**minimum 12 F, 6 D, 2 G**; the extra 6 are bench depth). Each round the manager fields a **12 F / 6 D / 2 G** active lineup from those 26, with the **backup goalie (G2) counting 50%**.
- Season: **2026/27** (NHL season id `20262027`). Leagues are creatable pre-season; the draft runs before puck drop; the season runs Oct 2026 – Apr 2027.
- Scope: regular season (the 84 rounds described by the user). Playoffs = future extension (see §10).

### 2.2 Rounds and the schedule ("game 4 vs game 3" problem)

This is the exact reading of the spec we'll implement:

- A **round** is per-team: round *k* of a manager team = its franchise's *k*-th NHL game of the season (index 1..84, computed by sorting the franchise's real schedule by date). Rounds are **not** aligned across the league — on the same calendar night one team can play its 4th round while its opponent plays its 3rd. This is normal early in an NHL season (uneven games played). **The 2026/27 NHL season has 84 games per team** (schedule expansion, confirmed — sources: Front Office Sports, NHL.com), so the spec's "rounds 1 to 84" maps directly to the real schedule; the round count is a per-season constant `ROUNDS_PER_TEAM` derived from the fetched schedule, not hardcoded.
- A **manager game** = one real NHL game between the two franchises (home manager vs away manager), exactly as the 2026/27 schedule dictates. So each manager plays exactly as many manager games as its franchise plays NHL games (84).
- **Round window** for scoring round *k* = the calendar period from the franchise's game *k−1* (exclusive) through game *k* (inclusive):
  - **Mode 1:** only the franchise's own game *k* counts (the franchise plays exactly one NHL game inside its own window).
  - **Mode 2 (confirmed):** every active drafted player earns points from **his own NHL team's games** inside the window (a drafted player's team can play 0–3 games in a ~2–4-day window). Players whose teams don't play in the window score 0 for that round.
- **Finalization rule (from the spec):** a manager game is **not final** until the underlying NHL game is fully played and its stats are complete ("all teams have played the full round"). Score is computed for round *k* (home) vs round *m* (away) from the same real game; the result (W/L only — **no overtime, no ties**) posts when the NHL game is final.
- **Postponements:** if an NHL game is postponed, the manager game is marked postponed and re-anchored to the rescheduled date (the round index stays the franchise's game index). The nightly sweep handles this (§7.4).

### 2.3 Lineup locks (from the spec)

- **Mode 1:** a manager can change the active 12F/6D/2G between rounds. Once the franchise's round-*k* NHL game has started (puck drop), that lineup is **locked** for round *k* — no switches until round *k*+1. ("You can't switch a player in the lineup if he has already played in that round.")
- **Mode 2:** the 20 drafted players are always active; the same lock rule applies to **trades**: a player whose team has already played inside the current round window cannot be traded away in that round (he stays through the current round's scoring).

### 2.4 Scoring — one unified rule for both modes

Each manager team's round score = Σ of its **active lineup players'** fantasy points earned in the round window (mode 1: from the franchise's real game; mode 2: from each player's own real NHL games in the window). Higher total wins the manager game. Ties are impossible by rulebook (W/L only) and are broken by a fixed ladder (§3.3).

---

## 3. Game score model — proposal

Simple, explainable, uses only real stats the app already computes. All numbers shown to 1 decimal.

### 3.1 Skaters

| Stat | Points |
|---|---|
| Goal | **+4** |
| Primary assist (A1) | **+3** |
| Secondary assist (A2) | **+2** |
| Shot on goal (SOG) | **+0.5** |
| Penalty drawn (PEND) | **+0.5** |
| Penalty taken (PENT) | **−0.5** |
| 5v5 on-ice xG differential (xGF − xGA, using the app's **xG_F** model) | **+1.0 per xG** |
| Goalie **GSAx** | **+3.0 per GSAx** |
| **Backup goalie (G2) lineup slot** | **× 0.5** (all other slots × 1.0) |

### 3.2 Goalies

| Stat | Points |
|---|---|
| GSAx (xGA − GA on shots against while in net, same definition the goalies API uses) | **+3.0 per GSAx** |

### 3.3 Why these weights (and typical magnitudes)

- **G > A1 > A2** — matches hockey value, familiar from every fantasy system.
- **SOG 0.5** — rewards volume shooters; ~30 SOG/game ≈ 15 pts across a lineup.
- **PENT/PEND ±0.5** — roughly net-neutral per game (≈4 penalties each way), but rewards drawing and punishes taking them; zero-sum per matchup.
- **5v5 on-ice xG± at 1.0** — a top-line forward typically nets ≈ +0.3…+0.6/game → ~0.5 pts; a bad night ≈ −0.5. Captures two-way play beyond the boxscore, directly uses the app's xG_F model.
- **GSAx × 3** — a solid goalie start (+0.5 GSAx ≈ +1.5 pts) to a great one (+2 ≈ +6 pts), so goalies matter but can't be scoreboard-stacking; a bad night (−2 ≈ −6) hurts.
- Typical **mode 1** team night: 3 G (12) + ~4.5 assists (≈11) + 30 SOG (15) + pens ≈ 0 + xG± ≈ +1 + goalie ±1.5 → **≈ 40 pts, ±10 variance** driven mostly by real goals. Typical **mode 2** round (≈1 game per player) lands in the same range.

### 3.4 Tie-break ladder (only wins/losses exist)

1. Total goals by active lineup in the round
2. Total A1
3. Fewest PENT
4. Higher combined GSAx
5. Deterministic coin flip seeded by `(league_id, game_id)` — cannot both lose.

### 3.5 Data flow for a round's score

Per real NHL game, run the existing PBP+xG pipeline (already ported in `routes/pbp.rs`) and aggregate per player:
- G/A1/A2 from goal events; SOG from shot events; PENT/PEND from penalty events;
- 5v5 on-ice xGF/xGA from xG_F on 5v5 shot events where the player is on ice (shift join already exists);
- Goalie GSAx from xGA − GA per goalie in net.

Store one immutable-ish **stat line per player per NHL game** (`manager_game_stats`) + the fantasy points; a manager game's score = Σ of its two lineups' players' points. Recompute is cheap and idempotent; corrections re-run only when a game changes after final.

---

## 4. Draft mode (26 rounds, snake)

- Draft pool: all players on current NHL rosters (`/v1/roster/{team}/current`, already cached in the Rust app) with position and team. Player value for rankings: existing **GM-mode projection value** (`proj_value_for_player`; includes GSAx component for goalies).
- Order: **snake** — round 1: 1→N, round 2: N→1, … (pick 1 in round 1 picks 32nd in round 2, per spec). Draft order = **randomized at league creation** (commissioner can re-randomize until the draft starts).
- **Roster/lineup (updated 2026-08-31):** teams draft **26 players** with a **minimum of 12 F, 6 D, 2 G** (extra 6 are bench depth, which makes trading easier — no position cap beyond the minimum). Each round the manager fields an **active lineup of 12 F / 6 D / 2 G** from those 26, where the **backup goalie (G2) scores at 50%** and every other slot scores at 100%.
- **CPU teams** (all unfilled franchise slots, up to 32 − #users) autopick immediately: **best available by projection value** with a positional-feasibility constraint — always leave enough remaining rounds to fill the mandatory 12F/6D/2G slots (e.g., can't end the draft with 0 G). Greedy best-available is the literal reading of "the computer picks the best available player".
- Every team must finish with exactly 12 F / 6 D / 2 G (draft UI enforces it; the snake + feasibility constraint guarantees it).
- Optional (cheap): per-pick countdown with autopick; pause between rounds; draft log. All state in Postgres so a draft can be paused/resumed and is safe across restarts.

---

## 5. Trades

- **Any manager ↔ any manager**, both modes, pre-season and in-season. Multi-player offers (N for M), always keeping both rosters valid (≥12F/6D/2G minimum; swaps are equal-size so the 26-man roster size is preserved).
- Flow: offer → accept/decline/counter (counter = new offer) → execution is atomic (transaction): swap roster membership, void conflicting lineup slots (mode 1) that the traded players occupied.
- **Lock rule in trades:** a player who has already played inside the current round window can't be traded away until the next round (spec rule applied to trades).
- **CPU trade bot** (confirmed): CPU accepts an offer iff `projected value received ≥ projected value sent × (1 − margin)`, margin ≈ 5–10%, using the same projection values as the draft AI; otherwise auto-declines. Commissioner can disable CPU trading per league.
- Trade log + league feed entry for every trade.

---

## 6. Rust architecture

New modules (all inside `nhl_rust/src`), following the existing conventions (Axum, `serde_json::Value`, `AppState`, moka caches, PostgREST reads/upserts):

```
src/
├── manager/
│   ├── mod.rs               (submodule glue)
│   ├── scoring.rs           (score model: weights, stat-line aggregation, tie-breaks — pure, unit-tested)
│   ├── league.rs            (league/team state, join codes, CPU slots, standings)
│   ├── schedule.rs          (season schedule snapshot + per-team round indices + round windows)
│   ├── draft.rs             (snake order, CPU autopick, feasibility, pick submission)
│   ├── trades.rs            (offers, accept/decline/counter, atomic execution, CPU bot, lock rule)
│   ├── lineups.rs           (mode-1 active lineup set + lock enforcement)
│   └── sweep.rs             (tokio interval job: schedule sync, postponements, finalization, recompute)
├── routes/manager.rs        (page + JSON API: /manager, /api/manager/*)
└── supabase/write.rs        (generalize the existing upsert helpers for the new tables)
```

API sketch (all under `/api/manager/…`): `leagues` (list/create/join), `leagues/<id>` (detail + settings), `schedule`, `games`, `game/<id>` (scoring breakdown), `standings`, `lineup` (get/set, with lock check), `draft` (state, players, pick), `trades` (offers, actions). Page routes: `/manager` (my leagues) and `/manager/<league_id>` (dashboard, tabs: Standings / Schedule / Team / Lineup / Trades / Draft). **Access (confirmed): free for all logged-in users** — `/manager` and `/api/manager/*` are NOT added to the premium path predicate in `web/auth_state.rs`; login is still required for writes.

---

## 7. Persistence (Supabase / Postgres)

New migration `supabase/migrations/0NN_create_manager_league.sql`:

| Table | Purpose |
|---|---|
| `manager_leagues` | id, name, season (`20262027`), mode, status (`setup→drafting→running→finished`), commissioner, settings (tie-breaks fixed, CPU-trades flag, draft order seed) |
| `manager_league_teams` | league_id, franchise (NHL abbrev), user_id (null = CPU), team_name, draft_order |
| `manager_rosters` | league_id, team, player_id, position_slot, acquired_via (`initial|draft|trade`) |
| `manager_lineups` | league_id, team, round, slot (F1..F12/D1..D6/G1/G2), player_id, locked_at |
| `manager_games` | league_id, nhl_game_id, home/away team, home_round/away_round, scheduled date, status (`pending/live/final/postponed`), home_score/away_score, final lock |
| `manager_trade_offers` | id, league_id, from/to, offered/requested player lists, status (`pending/accepted/declined/cancelled`) |
| `manager_game_stats` | nhl_game_id, player_id, G, A1, A2, SOG, PENT, PEND, xgf_5v5, xga_5v5, gsax, fantasy_points, computed_at |

RLS policies: users can read league state they belong to; write only their own team (lineups, trade offers) unless commissioner (league settings, force-advance). CPU teams are written server-side only.

### 7.1 Schedule snapshot & round indices
- At league creation (and re-synced until season lock): fetch the 1312-game 2026/27 schedule via `fetch_club_schedule_games` per franchise, snapshot into the league, and compute per-team round indices (1..N by date).
- Postponements after lock: nightly sweep refreshes schedule data, re-anchors affected manager games, keeps round indices.

### 7.2 Finalization pipeline
1. NHL game `gameState` = Final → run PBP aggregation → write `manager_game_stats` → compute both managers' scores (their round windows' lineups) → mark `manager_games.final`.
2. Idempotent + cached in memory; sweep job re-checks every N minutes (env `MANAGER_SWEEP_INTERVAL_SECONDS`, default 300) with short-TTL live behavior on game nights.

### 7.3 Lock enforcement
- Lock checks are server-side and time-based: lineup switch allowed iff the franchise's round game hasn't started (`startTimeUTC`), or per-player for mode-2 trades (player's team hasn't played inside the window). Enforced in the API, never trusted from the client.

### 7.4 Background sweeps
`MANAGER_SWEEPER=1` (default on in prod) starts a tokio interval task (same pattern as `jobs/prestart.rs`): schedule sync, postponed re-anchor, finalize completed games, recompute corrected stats, CPU trade auto-declines, and (pre-season) CPU draft autopicks for slow human drafts.

---

## 8. UI (existing template pattern)

- `/manager` — list my leagues + create form (name, mode, max teams, options).
- `/manager/<id>` — league dashboard, tabs:
  - **Overview/Standings** — W/L table + points-for/against, league feed (trades, results).
  - **Schedule** — league game grid, round labels per team (e.g. "ANA game 4 vs BOS game 3"), live/final/postponed badges.
  - **Team/Lineup** — mode 1: pick 12F/6D/2G from the franchise roster with lock countdown; mode 2: view the 20 drafted players, stats, per-round scores.
  - **Draft room** — snake board, live pick queue, CPU picks, countdown (pre-season only).
  - **Trades** — offer/counter/accept UI, roster validity preview, trade log.
  - **Game page** — per-game scoring breakdown (who scored what, xG±, GSAx lines).
- Reuses `base.html`, headshots/logos proxies, team branding. One new template family (`manager*.html`) + `routes/manager.rs` serving JSON; same `fetch`-driven pattern as `projections.html`.

---

## 9. Milestones

| # | Scope | Acceptance |
|---|---|---|
| M1 | Schema migration + league CRUD/join (CPU slots) + schedule snapshot + round indices/windows | Create a 32-team league; schedule renders with per-team rounds; join codes work; RLS verified |

### M1 status — ✅ IMPLEMENTED (2026-08-31)

- `supabase/migrations/017_create_manager_league.sql` — all seven tables + indexes + RLS (service-role only). **Deploy step:** apply this migration to the Supabase project before using the feature (the app never runs migrations itself).
- `nhl_rust/src/supabase/write.rs` — generic PostgREST `upsert_rows` / `update_rows` (chunked, merge-duplicates, representation returned).
- `nhl_rust/src/manager/mod.rs` — module constants (season default 20262027, statuses, modes, 12F/6D/2G lineup shape + slot labels).
- `nhl_rust/src/manager/schedule.rs` — `build_league_schedule_rows` (32 schedule fetches → 1344 deduped game rows with per-side round indices), pure `league_rows_from_team_games`, `assign_rounds`, `round_window` (window = previous game date exclusive → current game date inclusive). Unit-tested incl. the "home game 4 vs away game 3" case.
- `nhl_rust/src/manager/league.rs` — `create_league` (join-code, 32 franchise slots with creator claim, shuffled draft slots, schedule snapshot), `list_leagues_for_user`, `league_detail`, `join_league` (code + unclaimed-franchise check, one team per user), `membership`, `is_commissioner`.
- `nhl_rust/src/routes/manager.rs` — `/manager` page + `GET/POST /api/manager/leagues`, `POST /api/manager/leagues/join`, `GET /api/manager/leagues/{id}`, `GET .../schedule` (team/round/status/date filters), `GET .../rounds` (per-team played/next-round/window). Auth + CSRF enforced on writes; free of premium gating.
- `app/templates/manager_home.html` — hub page: my leagues, create (mode + franchise picker), join-by-code.
- Verified: `cargo check` clean, 60/60 tests pass, live smoke test — page renders (200), APIs return 401 (no session) / 400 (bad CSRF) / 503 (Supabase table not yet migrated) exactly as designed.

---

## 9b. Verified implementation anchors (checked against current code)

These were verified in the codebase on 2026-08-31; implementation should build on them directly.

- **Schedule fetch:** `data/projections.rs::fetch_club_schedule_games(state, team, season)` already prefers the real `20262027` schedule (published July 2026) and only falls back to shifting `20252026` forward when empty. Round indices = sort each franchise's `gameType==2` games by `date` → 1..84 (the 2026/27 season has 84 games per team — verified via Front Office Sports / NHL.com). Manager game id = NHL game id.
- **Draft pool:** `data/rosters.rs::all_rosters(caches, http)` merges skater+goalie bios for the current season (`{playerId, name, team, position(F/D/G), positionCode, shoots}`) — exactly the draft-board data. CPU pick value: `data/projections.rs::proj_value_for_player` over `load_gm_mode_projections_cached` (includes GSAx component for goalies).
- **Scoring inputs — the normalized PBP rows** (`routes/pbp.rs`, row keys):
  - `Goal` (1/0), `Shot` (1/0), `typeCode` (505 goal / 506 shot / 509 penalty), `Event`, `reason`, `PEN_duration`;
  - `Player1_ID/Player2_ID/Player3_ID` — candidate priority `scoringPlayerId, shootingPlayerId, playerId, hitting, hittee, assist1, assist2, blocking, losing, winning, committedByPlayerId, drawnByPlayerId`. For penalty events this makes **Player1 = committedBy, Player2 = drawnBy** (penalty plays have no other candidates). To make intent explicit and future-proof, M3 adds two additive columns `CommittedBy_ID` / `DrawnBy_ID` to the row (a small, parity-safe change in `routes/pbp.rs` only).
  - `StrengthState` (raw, e.g. `5v5`) and `StrengthState2` — filter `StrengthState == "5v5"` for on-ice xG±.
  - On-ice per event: `Home/Away_Forwards_ID | Defenders_ID | Goalie_ID` (space-separated ids from the shift join) — used to attribute on-ice xGF/xGA to the 5 skaters and xGA/GA to the goalie in net.
  - `xG_F` (and `xG_S`/`xG_F2`) filled per shot by `compute_xg` via `models/xg.rs`; the Manager scoring engine requests scope `xG_F`.
  - `Goalie_ID` = `goalieInNetId` on every event — GSAx = Σ xG_F(shots against) − GA, same definition as the existing goalies API.
- **In-process reuse:** `routes/pbp.rs` currently builds rows inside the HTTP handler; M3 will extract the build path into a `pub(crate)` helper (e.g. `build_plays(state, game_id, xg_scope, force)` returning the `mapped` rows) so the scoring engine and the sweeper call it without HTTP.
- **Persistence:** PostgREST reads via `supabase/read.rs::SbClient::read`; upsert examples in `supabase/auth.rs` (`upsert_user_account`, `upsert_card_builder_layout`). M1 adds a generic upsert helper for the new tables (PostgREST `on_conflict` + `Prefer: resolution=merge-duplicates`, same as Python `supabase_client.py`).
- **Auth:** route handlers read the session via `routes/auth.rs::auth_user_from_headers` (pattern already used by `routes/community.rs`); league write actions enforce membership/commissioner checks server-side.
- **Background sweep:** same pattern as `jobs/prestart.rs` (tokio::spawn interval loop, toggle env `MANAGER_SWEEPER`).
- **Tests:** this repo uses inline `#[cfg(test)] mod tests` per module (there is no `tests/` directory) — new modules (`manager/scoring.rs`, `manager/draft.rs`, `manager/schedule.rs`) follow that convention.
- **Router:** `routes/mod.rs::build_router` merges per-family `router()` builders — add `manager::router()` there. `/manager` is deliberately NOT added to the premium path predicate in `web/auth_state.rs` (confirmed: free for logged-in users).
- **Season id:** `util/dates.rs::current_season_id` switches to `20262027` on 2026-09-01 (unit-tested) — correct for league creation before the 2026-10-07 opener; leagues store their season explicitly regardless.
| M2 | Draft engine: snake order, CPU autopick (projection value + feasibility), pick API, draft room UI | Full 26-round draft with 1 human + 31 CPU completes; every team ends with a valid 12F/6D/2G-minimum roster |

### M2 status — ✅ IMPLEMENTED (2026-08-31)

- `supabase/migrations/018_manager_draft_columns.sql` — `manager_rosters.picked_at` (pick order for the draft log) + index. **Deploy step:** apply alongside 017.
- `nhl_rust/src/manager/draft.rs` — snake order (`pick_order`, round-1 pick 1 = round-2 pick 32), `current_picker`, 12F/6D/2G `feasible_pick` (caps + "enough picks left to fill mandatory slots"), `cpu_choose`/`cpu_pick_value` (GM-mode `projected_value`, positional fallback F > D > G), `start_draft` (commissioner, clears prior draft rosters, resets counters, runs CPU picks), `submit_pick` (turn/availability/feasibility validation, then CPU autopick to the next human), `draft_state` (order/counts/picker/taken map/recent picks), `available_players` (search + pos/team filters, value-sorted). Draft picks are `manager_rosters` rows (`acquired_via='draft'`); in-process per-league mutex serializes the pick transaction; draft completion uses round-21 sentinel until the M3 season sweep flips `drafting→running`.
- `nhl_rust/src/routes/manager.rs` — `GET /manager/{league_id}` page + `GET .../draft`, `POST .../draft/start`, `GET .../draft/players`, `POST .../draft/pick` (auth + CSRF on writes).
- `app/templates/manager_league.html` — draft room: board with CPU/human badges + current-picker highlight, searchable player list with projection values, pick action, recent-picks log, commissioner start button, 8s polling.
- `app/templates/manager_home.html` — league rows now link to `/manager/{id}`.
- Verified: `cargo check` clean, 65/65 tests pass (11 in `manager::`), live smoke test — league page renders with draft board; draft APIs degrade to 500 pre-migration as designed.
| M3 | Scoring engine: PBP aggregation, stat lines, game score model, tie-breaks, finalization, lineup locks (mode 1) | Play back a finished 2025/26 NHL game: both lineups' scores match hand-computed expectations; lock blocks post-puck-drop switches; finalization gates correctly |

### M3 status — ✅ ENGINE + APIs implemented (2026-08-31); live playback pending

- `src/routes/pbp.rs` — split the PBP route: the big body became a reusable
  `pub(crate) async fn build_plays(state, game_id, xg_scope, lite_mode) -> Result<(plays, game_state), ()>`
  (fetch + orient + normalize + shift-join + xG), and a thin `api_game_pbp`
  wrapper keeps the HTTP cache/disk/response behavior byte-identical. The
  Manager scoring engine calls `build_plays(..., "xG_F", false)` in-process.
  Middle 700 lines untouched; the split is head/tail edits only.
- `src/manager/scoring.rs` — the score model: weights (G +4, A1 +3, A2 +2,
  SOG +0.5, PEND +0.5, PENT −0.5, 5v5 on-ice xG± +1.0, GSAx +3.0), the pure
  `stat_lines_from_plays` aggregator (goals/assists, SOG, penalty actors,
  5v5 on-ice xGF/xGA via the shooter-vs-opponent on-ice split, goalie GSAx =
  ΣxG_F(shots against) − GA), `team_round_score`, and the W/L `decide_winner`
  ladder (score → goals → A1 → fewer PENT → GSAx → deterministic home).
  Unit-tested (9 tests: weights, goal events, penalties, 5v5 split, GSAx,
  team sum, tiebreak).
- `src/manager/games.rs` — `record_game_stats` (build_plays → stat lines →
  upsert `manager_game_stats`), `set_lineup`/`get_lineup` with slot↔position
  validation (12F/6D/2G) and the **round lock** (the team's round-k game is
  live/final/postponed → locked), and `finalize_game` (both lineups'
  `team_round_score` from recorded stat lines → winner → status=final +
  scores). Unit-tested (slot→position mapping).
- `src/routes/manager.rs` — `GET/POST .../lineup` (auth + CSRF + own-team /
  commissioner gate), `POST /api/manager/games/{id}/stats`, and
  `POST .../leagues/{id}/finalize/{nhl_game_id}` (commissioner-gated).
- **Pending (noted, needs deployed migrations + live NHL feed):** the lineup
  *editor UI* (ships with the M5 dashboard), and an end-to-end live playback
  check of a real finished game — the sandbox cannot reach
  `api-web.nhle.com` (TLS egress blocked) and migrations 017/018 aren't
  applied to Supabase yet. The pure aggregator is fully unit-tested instead.
| M4 | Trades (human + CPU bot) + lineup UI for mode 1 + round-window scoring for mode 2 | Trade executes atomically; CPU bot accepts/declines per margin; traded-away-after-puck-drop is rejected |

### M4 status — ✅ IMPLEMENTED (2026-08-31) · **CPU trades removed (per user)**

- `src/manager/trades.rs` — offer lifecycle (pending→accepted/declined/cancelled),
  sender-owns / recipient-owns validation, equal-size swap, **position validity**
  after the swap (mode 2 must stay exactly 12F/6D/2G; mode 1 ≥ 12F/6D/2G), and
  the **round-lock** rule (a team whose current-round game has started can't
  trade players out). `execute_trade` moves `manager_rosters` rows (offered →
  recipient, requested → sender) and voids stale `manager_lineups` slots that
  referenced a traded player. Unit-tested (counts exactness, swap position
  validity).
- **CPU teams excluded from trades (user decision):** `create_offer` now rejects
  any offer where the recipient is a CPU team (`cpu_team_not_tradable`); the
  CPU trade bot (`cpu_accepts`, projected-value decision, `CPU_MARGIN`) was
  removed entirely, so trades are human-to-human only and every offer is created
  `pending` for the recipient to accept/decline.
- `app/templates/manager_trades.html` + `GET /manager/{id}/trades` — the **trade
  center**: human-opponent selector (CPU excluded), check-box send/receive
  roster panels (N-for-N validation, equal count gate), and an inbox/outbox with
  Accept / Decline / Cancel actions. Player names resolved for offers.
- Verified: `cargo check` clean, **79/79 tests pass**; live smoke test — trade
  page renders, offers list returns 200, create-offer 401-unauthenticated.

### M5 follow-ups — ✅ schedule grid + ✅ LIVE end-to-end dry run (2026-08-31)

- **Schedule grid** — the league dashboard now has a **Schedule** card: game
  nights grouped by date, each row showing `away @ home`, both sides' own game
  numbers ("ANA g4 vs BOS g3"), a status chip (pending/live/final/postponed),
  and scores for finals. A team filter narrows the grid; your team's games are
  highlighted.
- **Live end-to-end dry run** (`MANAGER_DRY_RUN=1`; `nhl_rust/src/manager/dry_run.rs`):
  drives the engine in-process against **live Supabase + the real NHL feed** and
  cleaned up after itself. Confirmed end-to-end:
  `league created (1344 games — real 2026/27 schedule) → draft complete (26
  rounds, CPU autopick) → ANA roster 13F/11D/2G (minimums met) → offer ANA→BOS
  (pending) → trade accepted + executed → lineup set (unlocked) → DRY RUN OK →
  cleaned up`. A `MANAGER_DELETE_LEAGUE=<id>` helper removes any leftover test
  league (cascade).
| M5 | Dashboard/standings/schedule UI, league feed, sweeper job, postponed-game handling, unit + integration tests | Full season dry-run on recorded data: standings consistent, postponed game re-anchors, scores idempotent |

### M5 status — ✅ ENGINE + standings/feed/sweeper implemented (2026-08-31)

- `src/manager/standings.rs` — pure `standings_from_games` (W/L only, no OT,
  win% + GF/GA, sorted by win%) with tests; `league_standings` (reads
  `manager_games`), `league_feed` (recent trades + finalized results), and
  `seed_true_rosters` (mode-1: writes each franchise's current roster players
  to `manager_rosters` as `acquired_via='initial'`, idempotent upsert).
- `src/jobs/manager.rs` — the **sweeper** (`MANAGER_SWEEPER=1`, interval
  `MANAGER_SWEEP_INTERVAL_SECONDS`, default 300): season-start transition
  (`setup/drafting → running` once the league's first game date is reached,
  seeding mode-1 rosters at that point), and finalization of recent games —
  records stat lines and only `finalize_game`s when the NHL game's
  `gameState` is FINAL/OFF. Wired into `main.rs` (spawned like the prestart
  logger).
- `src/routes/manager.rs` — `GET .../standings`, `GET .../feed`,
  `POST .../sweep` (commissioner manual trigger), `POST .../seed-rosters`
  (commissioner).
- `app/templates/manager_league.html` — added **Standings** (table, highlights
  your team) and **League feed** (trade/result items) cards; refreshed with the
  league.
- `src/manager/games.rs::get_team_roster` + `GET .../roster?team=` — the
  lineup-selector pool (players with name/position/NHL team + F/D/G counts +
  minimums).
- `app/templates/manager_lineup.html` + `GET /manager/{id}/lineup` — the
  **lineup selector** (GM-Mode-style): click a roster player to fill the next
  empty slot of their position, click a slot to clear it; live F/D/G count
  validation, G2 "50%" / G1 "100%" labels, round lock banner (disabled + hidden
  Save when the round has started), and Save → POST `/lineup`. A "Set lineup"
  link appears on the dashboard when you're a member.
- Verified: `cargo check` clean, **80/80 tests pass**; live smoke test —
  lineup page renders, roster endpoint returns the 12/6/2-minimum shape, lineup
  GET returns 200.
- **Remaining for M5/M6:** the *trade* UI (APIs exist), a *schedule grid* tab,
  and the live end-to-end dry-run (needs the NHL feed, which is TLS-blocked
  from this sandbox).

### UI redesign — shared Manager design system (2026-08-31)

- `app/static/manager.css` — one shared stylesheet for all four manager pages
  (hero cards, gradient panels, chips/badges, buttons, fields, tables, player
  rows, empty states, on-the-clock pulse, jump nav). Fixes the previously
  unstyled sidebar (`hqj-*` styles were only defined by home/community) and the
  mobile "SLICERS" header on manager pages.
- `app/static/manager-theme.js` — franchise-color theming (mirrors base.html
  `setTheme()`): league-scoped pages recolor `--accent/--panel-alt/--value-text`
  and expose `--mgr-team` once the user's team is known.
- Templates rewritten on that system: hub (hero + league rows with franchise
  logo/status chips), league dashboard (hero + anchor jump nav + restyled draft
  board/room, recent picks, standings table, feed, schedule), lineup editor
  (slot groups, count pills, lock banner), trade center (builder panels, offer
  cards with team logos + status chips). All element ids and JS behavior
  preserved.
- `GET /api/manager/leagues` now returns `my_team`/`my_team_name` per league
  (caller's franchise from `manager_league_teams`) for the lobby cards.
- `base.html`: "Manager Game" added to the desktop nav tabs + mobile menu, and
  manager pages get a `mgr-page` body class.
| M6 | Polish: notifications (toast/email optional), draft countdown, admin tools, docs | Ready for real leagues before 2026-10-07 puck drop |

Sizing (rough): M1 ~2–3 d, M2 ~2 d, M3 ~3–4 d, M4 ~2–3 d, M5 ~2–3 d, M6 ~1–2 d.

---

## 10. Out of scope (future extensions, noted so they're deliberate)

- Manager-game **playoffs** (the spec covers regular-season rounds 1..84 only).
- Free agency / waivers / IR slots / salary cap.
- Draft-pick trading; keeper leagues.
- Any kind of game simulation — per spec, the Manager game scores real games only.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| NHL gamecenter data gaps (rare missing penalty/xG rows) | Stat-line recompute job; zeros not fallbacks (per repo convention); visible "data pending" state instead of a wrong score |
| Schedule changes / postponements mid-season | Snapshot + nightly re-sync + postponed re-anchoring (round indices are index-based, so re-anchoring is a date change) |
| Draft concurrency (two users pick simultaneously) | Single Postgres transaction per pick; optimistic lock on draft state; CPU picks serialize via sweep |
| xG_F model retrains mid-season | Score model uses the same pipeline version the app deploys; stat lines recomputed on version bump |
| Round-window asymmetry in mode 2 (window lengths vary 1–4 days) | Accepted design property (mirrors real schedule) — confirmed decision |

---

## 12. Decisions (resolved with the user, 2026-08-31)

1. **Rounds = 84** — the real 2026/27 NHL schedule has 84 games per team (historic expansion). Round indices come from the actual schedule; no buffer rounds needed.
2. **Draft-mode scoring window** — round-window accumulation (each drafted player's NHL games between your consecutive franchise games).
3. **CPU trades** — enabled, accepted by projected value with a margin; commissioner can disable per league.
4. **Access** — free for all logged-in users (no premium gating on `/manager`).
5. **Draft order** — randomized at league creation; commissioner can re-randomize until the draft starts.
