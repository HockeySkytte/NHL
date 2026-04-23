alter table if exists public.player_game_projections
    add column if not exists raw_poss_value double precision,
    add column if not exists raw_off_the_puck double precision,
    add column if not exists raw_gax double precision,
    add column if not exists raw_goalie_gsax double precision,
    add column if not exists raw_rookie_f double precision,
    add column if not exists raw_rookie_d double precision,
    add column if not exists raw_rookie_g double precision,
    add column if not exists raw_projected_value double precision;