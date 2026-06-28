-- Migration: Create nhl_player_metrics and nhl_current_playerprojections tables

create table if not exists public.nhl_player_metrics (
    season             text        not null,
    league             integer     not null default 1,
    playerid           bigint      not null,
    gameid             bigint      not null,
    team               text        not null,
    position           text        not null,
    strengthstate      text        not null,
    seasonstage        text,
    prior_games        integer     not null default 0,
    faceoffs           double precision not null default 0,
    defensive          double precision not null default 0,
    passes             double precision not null default 0,
    carries            double precision not null default 0,
    dump_ins_outs      double precision not null default 0,
    off_the_puck       double precision not null default 0,
    gax                double precision not null default 0,
    gsax               double precision not null default 0,
    xgf                double precision not null default 0,
    xga                double precision not null default 0,
    ig                 double precision not null default 0,
    a1                 double precision not null default 0,
    a2                 double precision not null default 0,
    ishots             double precision not null default 0,
    rookie_f           double precision not null default 0,
    rookie_d           double precision not null default 0,
    rookie_g           double precision not null default 0,
    gs_faceoffs        double precision not null default 0,
    gs_defensive       double precision not null default 0,
    gs_passes          double precision not null default 0,
    gs_carries         double precision not null default 0,
    gs_dump_ins_outs   double precision not null default 0,
    gs_off_the_puck    double precision not null default 0,
    gs_gax             double precision not null default 0,
    gs_gsax            double precision not null default 0,
    gs_xgf             double precision not null default 0,
    gs_xga             double precision not null default 0,
    gs_ig              double precision not null default 0,
    gs_a1              double precision not null default 0,
    gs_a2              double precision not null default 0,
    gs_ishots          double precision not null default 0,
    nhl_api_player_id  bigint,
    nhl_player_name    text,
    primary key (season, playerid, gameid, strengthstate)
);

create index if not exists idx_nhl_player_metrics_player
    on public.nhl_player_metrics (nhl_api_player_id, season);

create index if not exists idx_nhl_player_metrics_game
    on public.nhl_player_metrics (gameid, season);


create table if not exists public.nhl_current_playerprojections (
    season             text        not null,
    league             integer     not null default 1,
    playerid           bigint      not null,
    gameid             bigint      not null,
    team               text        not null,
    position           text        not null,
    strengthstate      text        not null,
    nhl_api_player_id  bigint,
    nhl_player_name    text,
    gp                 integer     not null default 0,
    faceoffs           double precision not null default 0,
    defensive          double precision not null default 0,
    passes             double precision not null default 0,
    carries            double precision not null default 0,
    dump_ins_outs      double precision not null default 0,
    off_the_puck       double precision not null default 0,
    gax                double precision not null default 0,
    gsax               double precision not null default 0,
    xgf                double precision not null default 0,
    xga                double precision not null default 0,
    ig                 double precision not null default 0,
    a1                 double precision not null default 0,
    a2                 double precision not null default 0,
    ishots             double precision not null default 0,
    rookie             double precision not null default 0,
    primary key (season, playerid, strengthstate)
);

create index if not exists idx_nhl_current_playerprojections_player
    on public.nhl_current_playerprojections (nhl_api_player_id, season);
