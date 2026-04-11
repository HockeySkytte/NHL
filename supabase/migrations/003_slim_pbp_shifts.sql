-- ============================================================
-- Slim down pbp and shifts tables
-- Drop columns no longer exported; change shifts.player_id to
-- TEXT (space-separated IDs after aggregation).
-- ============================================================

-- PBP: drop unused columns
ALTER TABLE pbp DROP COLUMN IF EXISTS strength_state2;
ALTER TABLE pbp DROP COLUMN IF EXISTS type_code;
ALTER TABLE pbp DROP COLUMN IF EXISTS shot_type2;
ALTER TABLE pbp DROP COLUMN IF EXISTS secondary_reason;
ALTER TABLE pbp DROP COLUMN IF EXISTS type_code2;
ALTER TABLE pbp DROP COLUMN IF EXISTS goalie;
ALTER TABLE pbp DROP COLUMN IF EXISTS player1;
ALTER TABLE pbp DROP COLUMN IF EXISTS player2;
ALTER TABLE pbp DROP COLUMN IF EXISTS player3;
ALTER TABLE pbp DROP COLUMN IF EXISTS score_state2;
ALTER TABLE pbp DROP COLUMN IF EXISTS home_forwards;
ALTER TABLE pbp DROP COLUMN IF EXISTS home_defenders;
ALTER TABLE pbp DROP COLUMN IF EXISTS home_goalie;
ALTER TABLE pbp DROP COLUMN IF EXISTS away_forwards;
ALTER TABLE pbp DROP COLUMN IF EXISTS away_defenders;
ALTER TABLE pbp DROP COLUMN IF EXISTS away_goalie;
ALTER TABLE pbp DROP COLUMN IF EXISTS box_id_rev;
ALTER TABLE pbp DROP COLUMN IF EXISTS box_id2;
ALTER TABLE pbp DROP COLUMN IF EXISTS rink_venue;
ALTER TABLE pbp DROP COLUMN IF EXISTS last_event;

-- Shifts: drop unused columns
ALTER TABLE shifts DROP COLUMN IF EXISTS date;
ALTER TABLE shifts DROP COLUMN IF EXISTS name;
ALTER TABLE shifts DROP COLUMN IF EXISTS position;
ALTER TABLE shifts DROP COLUMN IF EXISTS period;
ALTER TABLE shifts DROP COLUMN IF EXISTS start_time;
ALTER TABLE shifts DROP COLUMN IF EXISTS end_time;
ALTER TABLE shifts DROP COLUMN IF EXISTS strength_state_raw;
ALTER TABLE shifts DROP COLUMN IF EXISTS strength_state_bucket;
ALTER TABLE shifts DROP COLUMN IF EXISTS skaters_on_ice_for;
ALTER TABLE shifts DROP COLUMN IF EXISTS skaters_on_ice_against;
ALTER TABLE shifts DROP COLUMN IF EXISTS goalies_on_ice_for;
ALTER TABLE shifts DROP COLUMN IF EXISTS goalies_on_ice_against;

-- Shifts: player_id is now space-separated IDs (TEXT), not a single BIGINT
ALTER TABLE shifts ALTER COLUMN player_id TYPE TEXT USING player_id::TEXT;
