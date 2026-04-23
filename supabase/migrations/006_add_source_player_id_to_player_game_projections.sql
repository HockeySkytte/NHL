alter table if exists public.player_game_projections
    add column if not exists source_player_id bigint;

create index if not exists idx_player_game_projections_source_player
    on public.player_game_projections (source_player_id, season);