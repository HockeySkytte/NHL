-- ============================================================
-- Fix column types for real-world data values
-- EventIndex / ShiftIndex are 14-digit numbers (GameID*10000+idx)
-- Various PBP columns come as floats from the API
-- ============================================================

-- PBP: shift_index holds values like 20240200030001 → needs BIGINT
ALTER TABLE pbp ALTER COLUMN shift_index TYPE BIGINT;

-- PBP: x/y can be floats, box_size can be float
ALTER TABLE pbp ALTER COLUMN x TYPE REAL;
ALTER TABLE pbp ALTER COLUMN y TYPE REAL;
ALTER TABLE pbp ALTER COLUMN box_size TYPE REAL;

-- PBP: player IDs can exceed INT range in some edge cases
ALTER TABLE pbp ALTER COLUMN goalie_id TYPE BIGINT;
ALTER TABLE pbp ALTER COLUMN player1_id TYPE BIGINT;
ALTER TABLE pbp ALTER COLUMN player2_id TYPE BIGINT;
ALTER TABLE pbp ALTER COLUMN player3_id TYPE BIGINT;

-- PBP: score_state can be float
ALTER TABLE pbp ALTER COLUMN score_state TYPE REAL;
ALTER TABLE pbp ALTER COLUMN score_state2 TYPE REAL;

-- PBP: type_code can be large
ALTER TABLE pbp ALTER COLUMN type_code TYPE BIGINT;

-- PBP: game_time can be float
ALTER TABLE pbp ALTER COLUMN game_time TYPE REAL;

-- Shifts: shift_index holds values like 20240200030001 → needs BIGINT  
ALTER TABLE shifts ALTER COLUMN shift_index TYPE BIGINT;

-- Game_data: many INT columns receive float values → use REAL
-- Corsi/Fenwick/Shots/Goals columns
ALTER TABLE game_data ALTER COLUMN cf_all TYPE REAL;
ALTER TABLE game_data ALTER COLUMN cf_ev TYPE REAL;
ALTER TABLE game_data ALTER COLUMN cf_pp TYPE REAL;
ALTER TABLE game_data ALTER COLUMN cf_sh TYPE REAL;
ALTER TABLE game_data ALTER COLUMN ca_all TYPE REAL;
ALTER TABLE game_data ALTER COLUMN ca_ev TYPE REAL;
ALTER TABLE game_data ALTER COLUMN ca_pp TYPE REAL;
ALTER TABLE game_data ALTER COLUMN ca_sh TYPE REAL;
ALTER TABLE game_data ALTER COLUMN ff_all TYPE REAL;
ALTER TABLE game_data ALTER COLUMN ff_ev TYPE REAL;
ALTER TABLE game_data ALTER COLUMN ff_pp TYPE REAL;
ALTER TABLE game_data ALTER COLUMN ff_sh TYPE REAL;
ALTER TABLE game_data ALTER COLUMN fa_all TYPE REAL;
ALTER TABLE game_data ALTER COLUMN fa_ev TYPE REAL;
ALTER TABLE game_data ALTER COLUMN fa_pp TYPE REAL;
ALTER TABLE game_data ALTER COLUMN fa_sh TYPE REAL;
ALTER TABLE game_data ALTER COLUMN sf_all TYPE REAL;
ALTER TABLE game_data ALTER COLUMN sf_ev TYPE REAL;
ALTER TABLE game_data ALTER COLUMN sf_pp TYPE REAL;
ALTER TABLE game_data ALTER COLUMN sf_sh TYPE REAL;
ALTER TABLE game_data ALTER COLUMN sa_all TYPE REAL;
ALTER TABLE game_data ALTER COLUMN sa_ev TYPE REAL;
ALTER TABLE game_data ALTER COLUMN sa_pp TYPE REAL;
ALTER TABLE game_data ALTER COLUMN sa_sh TYPE REAL;
ALTER TABLE game_data ALTER COLUMN gf_all TYPE REAL;
ALTER TABLE game_data ALTER COLUMN gf_ev TYPE REAL;
ALTER TABLE game_data ALTER COLUMN gf_pp TYPE REAL;
ALTER TABLE game_data ALTER COLUMN gf_sh TYPE REAL;
ALTER TABLE game_data ALTER COLUMN ga_all TYPE REAL;
ALTER TABLE game_data ALTER COLUMN ga_ev TYPE REAL;
ALTER TABLE game_data ALTER COLUMN ga_pp TYPE REAL;
ALTER TABLE game_data ALTER COLUMN ga_sh TYPE REAL;
ALTER TABLE game_data ALTER COLUMN ig_all TYPE REAL;
ALTER TABLE game_data ALTER COLUMN ig_ev TYPE REAL;
ALTER TABLE game_data ALTER COLUMN ig_pp TYPE REAL;
ALTER TABLE game_data ALTER COLUMN ig_sh TYPE REAL;
ALTER TABLE game_data ALTER COLUMN a1_all TYPE REAL;
ALTER TABLE game_data ALTER COLUMN a1_ev TYPE REAL;
ALTER TABLE game_data ALTER COLUMN a1_pp TYPE REAL;
ALTER TABLE game_data ALTER COLUMN a1_sh TYPE REAL;
ALTER TABLE game_data ALTER COLUMN a2_all TYPE REAL;
ALTER TABLE game_data ALTER COLUMN a2_ev TYPE REAL;
ALTER TABLE game_data ALTER COLUMN a2_pp TYPE REAL;
ALTER TABLE game_data ALTER COLUMN a2_sh TYPE REAL;
