-- ============================================================
-- NHL App  –  Supabase (PostgreSQL) schema
-- Replaces MySQL season-specific tables + Google Sheets
-- ============================================================
-- Run once in the Supabase SQL editor (or via supabase db push).
-- All season-specific MySQL tables are unified into single tables
-- with a "season" column and appropriate indexes.
-- ============================================================

-- -----------------------------------------------------------
-- 1. teams  (was: Teams.csv)
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS teams (
    team        TEXT        PRIMARY KEY,          -- 3-letter abbreviation (e.g. 'BOS')
    team_id     INT         NOT NULL UNIQUE,
    name        TEXT        NOT NULL,
    logo        TEXT,
    color       TEXT,
    active      BOOLEAN     NOT NULL DEFAULT TRUE
);

-- -----------------------------------------------------------
-- 2. players  (was: nhl_players_{season})
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS players (
    player_id       BIGINT      NOT NULL,
    season          INT         NOT NULL,         -- e.g. 20252026
    player          TEXT        NOT NULL,         -- full name
    position        VARCHAR(2),                   -- C / L / R / D / G
    shoots_catches  VARCHAR(2),                   -- L / R
    birthday        DATE,
    nationality     VARCHAR(3),
    height          INT,                          -- cm
    weight          INT,                          -- lbs
    draft_position  INT,
    draft_year      INT,
    PRIMARY KEY (player_id, season)
);
CREATE INDEX IF NOT EXISTS idx_players_season ON players (season);

-- -----------------------------------------------------------
-- 3. pbp  (was: nhl_{season}_pbp)
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS pbp (
    event_index         BIGINT      NOT NULL,     -- GameID*10000 + play_index
    game_id             INT         NOT NULL,
    season              INT         NOT NULL,
    season_state        TEXT,                      -- 'regular' / 'playoffs'
    date                DATE,
    venue               TEXT,                      -- 'Home' / 'Away'
    period              SMALLINT,
    game_time           INT,                       -- seconds from game start
    strength_state      TEXT,                      -- '5v5', '5v4', 'ENF', etc.
    strength_state2     TEXT,                      -- 'PP1', 'PP2', 'SH', or raw
    type_code           INT,                       -- NHL event type code
    event               TEXT,                      -- 'shot-on-goal', 'goal', etc.
    x                   INT,
    y                   INT,
    zone                VARCHAR(1),                -- O / D / N
    reason              TEXT,
    shot_type           TEXT,
    shot_type2          TEXT,                      -- normalised
    secondary_reason    TEXT,
    type_code2          TEXT,
    pen_duration        REAL,
    event_team          TEXT,
    opponent            TEXT,
    goalie_id           INT,
    goalie              TEXT,
    player1_id          INT,
    player1             TEXT,
    player2_id          INT,
    player2             TEXT,
    player3_id          INT,
    player3             TEXT,
    corsi               SMALLINT,                  -- 0 / 1
    fenwick             SMALLINT,
    shot                SMALLINT,
    goal                SMALLINT,
    shift_index         INT,
    score_state         INT,
    score_state2        INT,                       -- bounded [-3, 3]
    home_forwards_id    TEXT,
    home_forwards       TEXT,
    home_defenders_id   TEXT,
    home_defenders      TEXT,
    home_goalie_id      TEXT,
    home_goalie         TEXT,
    away_forwards_id    TEXT,
    away_forwards       TEXT,
    away_defenders_id   TEXT,
    away_defenders      TEXT,
    away_goalie_id      TEXT,
    away_goalie         TEXT,
    box_id              TEXT,
    box_id_rev          TEXT,
    box_size            INT,
    box_id2             TEXT,
    shot_distance       REAL,
    shot_angle          REAL,
    position            VARCHAR(2),
    shoots              VARCHAR(2),
    rink_venue          TEXT,
    last_event          TEXT,
    xg_f                REAL,
    xg_s                REAL,
    xg_f2               REAL,
    PRIMARY KEY (event_index, season)
);
CREATE INDEX IF NOT EXISTS idx_pbp_game    ON pbp (game_id);
CREATE INDEX IF NOT EXISTS idx_pbp_season  ON pbp (season);
CREATE INDEX IF NOT EXISTS idx_pbp_date    ON pbp (date);
CREATE INDEX IF NOT EXISTS idx_pbp_game_season ON pbp (game_id, season);

-- -----------------------------------------------------------
-- 4. shifts  (was: nhl_{season}_shifts)
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS shifts (
    id                          BIGSERIAL   PRIMARY KEY,
    shift_index                 INT         NOT NULL,
    game_id                     INT         NOT NULL,
    season                      INT         NOT NULL,
    date                        DATE,
    player_id                   BIGINT      NOT NULL,
    name                        TEXT,
    position                    VARCHAR(2),
    team                        TEXT,
    period                      SMALLINT,
    start_time                  INT,        -- seconds
    end_time                    INT,
    duration                    INT,
    strength_state              TEXT,
    strength_state_raw          VARCHAR(16),
    strength_state_bucket       VARCHAR(16),  -- '5v5','PP','SH','Other'
    skaters_on_ice_for          SMALLINT,
    skaters_on_ice_against      SMALLINT,
    goalies_on_ice_for          SMALLINT,
    goalies_on_ice_against      SMALLINT
);
CREATE INDEX IF NOT EXISTS idx_shifts_game      ON shifts (game_id);
CREATE INDEX IF NOT EXISTS idx_shifts_season    ON shifts (season);
CREATE INDEX IF NOT EXISTS idx_shifts_player    ON shifts (player_id);
CREATE INDEX IF NOT EXISTS idx_shifts_game_si   ON shifts (game_id, shift_index);

-- -----------------------------------------------------------
-- 5. game_data  (was: nhl_{season}_gamedata)
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS game_data (
    game_id     INT         NOT NULL,
    season      INT         NOT NULL,
    date        DATE,
    player_id   INT         NOT NULL,
    player      TEXT,
    position    VARCHAR(2),
    team        TEXT,
    birthday    DATE,
    -- Time on ice (minutes)
    toi_all     REAL, toi_ev  REAL, toi_pp  REAL, toi_sh  REAL,
    -- Corsi
    cf_all      INT,  cf_ev   INT,  cf_pp   INT,  cf_sh   INT,
    ca_all      INT,  ca_ev   INT,  ca_pp   INT,  ca_sh   INT,
    -- Fenwick
    ff_all      INT,  ff_ev   INT,  ff_pp   INT,  ff_sh   INT,
    fa_all      INT,  fa_ev   INT,  fa_pp   INT,  fa_sh   INT,
    -- Shots
    sf_all      INT,  sf_ev   INT,  sf_pp   INT,  sf_sh   INT,
    sa_all      INT,  sa_ev   INT,  sa_pp   INT,  sa_sh   INT,
    -- Goals
    gf_all      INT,  gf_ev   INT,  gf_pp   INT,  gf_sh   INT,
    ga_all      INT,  ga_ev   INT,  ga_pp   INT,  ga_sh   INT,
    -- xG (Fenwick model)
    xgf_f_all   REAL, xgf_f_ev REAL, xgf_f_pp REAL, xgf_f_sh REAL,
    xga_f_all   REAL, xga_f_ev REAL, xga_f_pp REAL, xga_f_sh REAL,
    -- xG (Fenwick v2 model)
    xgf_f2_all  REAL, xgf_f2_ev REAL, xgf_f2_pp REAL, xgf_f2_sh REAL,
    xga_f2_all  REAL, xga_f2_ev REAL, xga_f2_pp REAL, xga_f2_sh REAL,
    -- xG (Shot model)
    xgf_s_all   REAL, xgf_s_ev REAL, xgf_s_pp REAL, xgf_s_sh REAL,
    xga_s_all   REAL, xga_s_ev REAL, xga_s_pp REAL, xga_s_sh REAL,
    -- Individual stats
    ig_all      INT,  ig_ev   INT,  ig_pp   INT,  ig_sh   INT,
    a1_all      INT,  a1_ev   INT,  a1_pp   INT,  a1_sh   INT,
    a2_all      INT,  a2_ev   INT,  a2_pp   INT,  a2_sh   INT,
    pen_taken   REAL,
    pen_drawn   REAL,
    PRIMARY KEY (game_id, player_id, season)
);
CREATE INDEX IF NOT EXISTS idx_gamedata_season ON game_data (season);
CREATE INDEX IF NOT EXISTS idx_gamedata_player ON game_data (player_id);

-- -----------------------------------------------------------
-- 6. season_stats  (was: nhl_seasonstats / Sheets6)
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS season_stats (
    season          INT         NOT NULL,
    season_state    TEXT        NOT NULL,      -- 'regular' / 'playoffs'
    strength_state  TEXT        NOT NULL,      -- '5v5', 'PP', 'SH', 'All'
    player_id       INT         NOT NULL,
    position        VARCHAR(2),
    gp              INT,
    plus_minus      INT,
    blocked_shots   INT,
    toi             REAL,
    i_goals         INT,
    assists1        INT,
    assists2        INT,
    i_corsi         INT,
    i_fenwick       INT,
    i_shots         INT,
    ixg_f           REAL,
    ixg_s           REAL,
    ixg_f2          REAL,
    pim_taken       REAL,
    pim_drawn       REAL,
    hits            INT,
    takeaways       INT,
    giveaways       INT,
    so_goal         INT,
    so_attempt      INT,
    ca              INT, cf INT,
    fa              INT, ff INT,
    sa              INT, sf INT,
    ga              INT, gf INT,
    xga_f           REAL, xgf_f  REAL,
    xga_s           REAL, xgf_s  REAL,
    xga_f2          REAL, xgf_f2 REAL,
    pim_for         REAL,
    pim_against     REAL,
    PRIMARY KEY (season, season_state, strength_state, player_id)
);
CREATE INDEX IF NOT EXISTS idx_seasonstats_player ON season_stats (player_id);

-- -----------------------------------------------------------
-- 7. season_stats_teams  (was: nhl_seasonstats_teams / Sheets7)
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS season_stats_teams (
    season          INT         NOT NULL,
    season_state    TEXT        NOT NULL,
    strength_state  TEXT        NOT NULL,
    team            TEXT        NOT NULL,
    gp              INT,
    toi             REAL,
    cf INT, ca INT, ff INT, fa INT, sf INT, sa INT, gf INT, ga INT,
    xgf_f REAL, xga_f REAL,
    xgf_s REAL, xga_s REAL,
    xgf_f2 REAL, xga_f2 REAL,
    pim_for REAL, pim_against REAL,
    PRIMARY KEY (season, season_state, strength_state, team)
);

-- -----------------------------------------------------------
-- 8. player_projections  (was: Sheets3 / nhl_player_projections)
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS player_projections (
    player_id   INT         PRIMARY KEY,
    position    VARCHAR(2),                   -- 'F' / 'D' / 'G'
    game_no     INT,
    age         REAL,
    rookie      REAL,
    evo         REAL,
    evd         REAL,
    pp          REAL,
    sh          REAL,
    gsax        REAL
);

-- -----------------------------------------------------------
-- 9. rapm  (was: rapm2_all / Sheets4)
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS rapm (
    player_id           INT     NOT NULL,
    season              INT     NOT NULL,
    strength_state      TEXT    NOT NULL,
    rates_totals        TEXT,
    -- core RAPM coefficients
    cf REAL, ca REAL, gf REAL, ga REAL,
    xgf REAL, xga REAL,
    pen_taken REAL, pen_drawn REAL,
    -- plus-minus composites
    c_plusminus REAL, g_plusminus REAL, xg_plusminus REAL, pen_plusminus REAL,
    -- ridge alphas used
    alpha_cf REAL, alpha_gf REAL, alpha_xgf REAL, alpha_pen REAL,
    -- stddev / z-scores (stored as JSONB for flexibility – many columns)
    stddev  JSONB,
    zscore  JSONB,
    -- PP / SH sub-splits (stored as JSONB)
    pp_sh   JSONB,
    PRIMARY KEY (player_id, season, strength_state)
);

-- -----------------------------------------------------------
-- 10. rapm_context  (was: context_all / Sheets5)
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS rapm_context (
    player_id               INT     NOT NULL,
    season                  INT     NOT NULL,
    strength_state          TEXT    NOT NULL,
    minutes                 REAL,
    qot_blend_xg67_g33     REAL,
    qoc_blend_xg67_g33     REAL,
    zs_difficulty           REAL,
    PRIMARY KEY (player_id, season, strength_state)
);

-- -----------------------------------------------------------
-- 11. rapm_data  (was: rapm_data_{season})
--     Intermediate shift-segment dataset for RAPM fitting
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS rapm_data (
    id                  BIGSERIAL PRIMARY KEY,
    season              INT     NOT NULL,
    shift_index         INT,
    game_id             INT,
    date                DATE,
    period              SMALLINT,
    duration            INT,
    home_team           TEXT,
    away_team           TEXT,
    home_skaters        TEXT,       -- space-separated player IDs
    away_skaters        TEXT,
    home_goalie         TEXT,
    away_goalie         TEXT,
    home_strength_state TEXT,
    away_strength_state TEXT,
    home_corsi          INT,
    away_corsi          INT,
    home_goal           INT,
    away_goal           INT,
    home_xg             REAL,
    away_xg             REAL,
    home_pen            REAL,
    away_pen            REAL,
    zone                VARCHAR(1),
    score_state         INT
);
CREATE INDEX IF NOT EXISTS idx_rapmdata_season ON rapm_data (season);

-- -----------------------------------------------------------
-- 12. odds_history  (was: Google Sheet1)
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS odds_history (
    id          BIGSERIAL   PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    game_id     INT         NOT NULL,
    team        TEXT        NOT NULL,
    ml          REAL
);
CREATE INDEX IF NOT EXISTS idx_odds_game ON odds_history (game_id);

-- -----------------------------------------------------------
-- 13. lineups  (was: Google Sheets2 / lineups_all.json)
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS lineups (
    id          BIGSERIAL   PRIMARY KEY,
    team        TEXT        NOT NULL,
    unit        TEXT,                          -- e.g. 'Line 1', 'Pair 1'
    pos         TEXT,                          -- position within unit
    player_name TEXT        NOT NULL,
    player_id   INT,
    timestamp   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_lineups_team ON lineups (team);

-- -----------------------------------------------------------
-- 14. started_overrides  (was: Google started_overrides sheet)
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS started_overrides (
    id          BIGSERIAL   PRIMARY KEY,
    game_id     INT         NOT NULL,
    team        TEXT,
    ml          REAL
);
CREATE INDEX IF NOT EXISTS idx_overrides_game ON started_overrides (game_id);

-- -----------------------------------------------------------
-- 15. last_dates  (was: Last_date.csv)
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS last_dates (
    season      INT         PRIMARY KEY,
    last_date   DATE        NOT NULL
);

-- -----------------------------------------------------------
-- 16. box_ids  (was: BoxID.csv)
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS box_ids (
    x           INT         NOT NULL,
    y           INT         NOT NULL,
    box_id      TEXT        NOT NULL,
    box_id_rev  TEXT        NOT NULL,
    box_size    INT,
    PRIMARY KEY (x, y)
);

-- -----------------------------------------------------------
-- 17. model_fenwick  (training data for xG Fenwick models)
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_fenwick (
    id              BIGSERIAL PRIMARY KEY,
    venue           TEXT,
    shot_type2      TEXT,
    score_state2    INT,
    rink_venue      TEXT,
    strength_state2 TEXT,
    box_id2         TEXT,
    season          INT,
    goal            SMALLINT
);

-- -----------------------------------------------------------
-- 18. model_shot  (training data for xG Shot model)
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_shot (
    id              BIGSERIAL PRIMARY KEY,
    venue           TEXT,
    shot_type2      TEXT,
    score_state2    INT,
    rink_venue      TEXT,
    strength_state2 TEXT,
    box_id2         TEXT,
    last_event      TEXT,
    season          INT,
    goal            SMALLINT
);

-- -----------------------------------------------------------
-- 19. game_model_preseason  (training data for game projections)
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS game_model_preseason (
    id      BIGSERIAL PRIMARY KEY,
    data    JSONB NOT NULL          -- flexible schema; columns TBD from MySQL
);

-- -----------------------------------------------------------
-- Row Level Security  (enable RLS, then allow anon read)
-- Adjust policies to your needs once auth is set up.
-- -----------------------------------------------------------
DO $$
DECLARE
    t TEXT;
BEGIN
    FOR t IN
        SELECT unnest(ARRAY[
            'teams','players','pbp','shifts','game_data',
            'season_stats','season_stats_teams','player_projections',
            'rapm','rapm_context','rapm_data',
            'odds_history','lineups','started_overrides',
            'last_dates','box_ids',
            'model_fenwick','model_shot','game_model_preseason'
        ])
    LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
        -- Allow the service_role key (used by backend) full access
        EXECUTE format(
            'CREATE POLICY %I ON %I FOR ALL USING (true) WITH CHECK (true)',
            'service_role_all_' || t, t
        );
    END LOOP;
END $$;
