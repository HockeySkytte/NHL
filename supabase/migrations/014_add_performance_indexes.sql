-- ============================================================
-- 014_add_performance_indexes
--
-- Add indexes to support the most common query patterns.
-- Tables are ordered by data volume (largest first).
-- ============================================================

-- -----------------------------------------------------------
-- pbp  (~252K rows and growing fast)
-- -----------------------------------------------------------

-- Team stats rebuild: GROUP BY season_state, strength_state, event_team
CREATE INDEX IF NOT EXISTS idx_pbp_season_team ON pbp (season, event_team)
    WHERE event_team IS NOT NULL AND TRIM(event_team) <> '';

-- Against stats rebuild: GROUP BY season_state, strength_state, opponent
CREATE INDEX IF NOT EXISTS idx_pbp_season_opp ON pbp (season, opponent)
    WHERE opponent IS NOT NULL AND TRIM(opponent) <> '';

-- Strength-state filtering (most analytics views filter by strength)
CREATE INDEX IF NOT EXISTS idx_pbp_season_strength ON pbp (season, strength_state);

-- Season-state filtering (regular vs playoffs)
CREATE INDEX IF NOT EXISTS idx_pbp_season_state ON pbp (season, season_state);

-- Common composite: season + date for backfill date-range queries
CREATE INDEX IF NOT EXISTS idx_pbp_season_date ON pbp (season, date);

-- Game-level event-type lookups (e.g. "give me all goals in this game")
CREATE INDEX IF NOT EXISTS idx_pbp_game_event ON pbp (game_id, event);

-- Fenwick events (shot attempts + goals) — most common xG target
CREATE INDEX IF NOT EXISTS idx_pbp_season_fenwick ON pbp (season, game_id)
    WHERE fenwick = 1;

-- Goals only — small subset, useful for highlight/score queries
CREATE INDEX IF NOT EXISTS idx_pbp_season_goal ON pbp (season, game_id)
    WHERE goal = 1;

-- -----------------------------------------------------------
-- shifts  (~464K rows, aggregated by team+shift_index)
-- -----------------------------------------------------------

-- TOI rebuild join: shifts.season + shifts.game_id ← pbp.game_id
CREATE INDEX IF NOT EXISTS idx_shifts_season_game ON shifts (season, game_id);

-- Team TOI aggregation: GROUP BY season_state, strength_state, team
CREATE INDEX IF NOT EXISTS idx_shifts_season_team ON shifts (season, team);

-- Line-tool: look up all shifts for a game
-- (idx_shifts_game already exists but doesn't include season)
-- Already covered by idx_shifts_season_game above + idx_shifts_game.

-- -----------------------------------------------------------
-- game_data  (~53K rows)
-- -----------------------------------------------------------

-- Season stats rebuild reads all rows for a season
-- idx_gamedata_season already exists; composite with team helps roster views
CREATE INDEX IF NOT EXISTS idx_gamedata_season_team ON game_data (season, team);

-- Player career lookups across seasons
CREATE INDEX IF NOT EXISTS idx_gamedata_player_season ON game_data (player_id, season);

-- Game-level player lookups (game detail page)
CREATE INDEX IF NOT EXISTS idx_gamedata_game_player ON game_data (game_id, player_id);

-- -----------------------------------------------------------
-- season_stats  (~4K rows)
-- -----------------------------------------------------------

-- Player career queries (most common: "show me a player's stats across seasons")
CREATE INDEX IF NOT EXISTS idx_seasonstats_player_season ON season_stats (player_id, season);

-- Filter by season + season_state (regular/playoffs toggle)
CREATE INDEX IF NOT EXISTS idx_seasonstats_season_state ON season_stats (season, season_state);

-- -----------------------------------------------------------
-- season_stats_teams  (~192 rows — tiny, but used on every /teams load)
-- -----------------------------------------------------------

-- Primary filter: teams page always passes season
CREATE INDEX IF NOT EXISTS idx_ssteams_season ON season_stats_teams (season);

-- Team-specific lookups
CREATE INDEX IF NOT EXISTS idx_ssteams_season_team ON season_stats_teams (season, team);

-- -----------------------------------------------------------
-- rapm / rapm_context  (~few thousand rows each)
-- -----------------------------------------------------------

-- Player RAPM lookups (most common access pattern)
CREATE INDEX IF NOT EXISTS idx_rapm_player_season ON rapm (player_id, season);

-- Context lookups by player
CREATE INDEX IF NOT EXISTS idx_rapmctx_player_season ON rapm_context (player_id, season);

-- -----------------------------------------------------------
-- player_projections  (~few hundred rows)
-- -----------------------------------------------------------

-- Already has PRIMARY KEY on player_id; no additional indexes needed.

-- -----------------------------------------------------------
-- odds_history  (can grow large over time)
-- -----------------------------------------------------------

-- Query by game_id (most common)
-- idx_odds_game already exists.

-- Time-based queries (recent odds)
CREATE INDEX IF NOT EXISTS idx_odds_timestamp ON odds_history (timestamp);

-- -----------------------------------------------------------
-- lineups  (moderate size)
-- -----------------------------------------------------------

-- Team lookups (most common)
-- idx_lineups_team already exists.

-- Timestamp for "latest lineup" queries
CREATE INDEX IF NOT EXISTS idx_lineups_timestamp ON lineups (timestamp);

-- -----------------------------------------------------------
-- rapm_data  (large intermediate table for RAPM fitting)
-- -----------------------------------------------------------

-- idx_rapmdata_season already exists.
-- RAPM fitting queries by game
CREATE INDEX IF NOT EXISTS idx_rapmdata_game ON rapm_data (game_id);
