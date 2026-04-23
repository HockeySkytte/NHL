create table if not exists public.player_game_projections (
    season integer not null,
    game_id bigint not null,
    source_game_id bigint not null,
    game_date date not null,
    player_id bigint not null,
    source_player_id bigint not null,
    player text not null,
    source_player_name text,
    team text not null,
    opponent text not null,
    position text not null,
    side text not null,
    model_key text not null,
    window_games integer not null,
    weighting text not null,
    games_in_window integer not null,
    rookie_factor double precision not null,
    poss_value double precision not null,
    off_the_puck double precision not null,
    gax double precision not null,
    goalie_gsax double precision not null,
    rookie_f double precision not null,
    rookie_d double precision not null,
    rookie_g double precision not null,
    raw_poss_value double precision,
    raw_off_the_puck double precision,
    raw_gax double precision,
    raw_goalie_gsax double precision,
    raw_rookie_f double precision,
    raw_rookie_d double precision,
    raw_rookie_g double precision,
    raw_projected_value double precision,
    projected_value double precision not null,
    match_type text not null,
    generated_at timestamptz not null default now(),
    primary key (season, game_id, player_id, model_key)
);

create index if not exists idx_player_game_projections_player
    on public.player_game_projections (player_id, season);

create index if not exists idx_player_game_projections_source_player
    on public.player_game_projections (source_player_id, season);

create index if not exists idx_player_game_projections_game
    on public.player_game_projections (game_id, season);