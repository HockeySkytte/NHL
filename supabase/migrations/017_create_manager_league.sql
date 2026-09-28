-- ============================================================
-- Manager Game (M1): leagues, teams, schedule snapshot, rosters,
-- lineups, trades, and per-game stat lines.
--
-- The Rust app writes these tables with the service-role key
-- (service_role bypasses RLS), so no anon/authenticated policies
-- are granted; RLS stays enabled as defense in depth.
-- ============================================================

-- ── Leagues ──────────────────────────────────────────────────────────
create table if not exists public.manager_leagues (
    id uuid primary key default gen_random_uuid(),
    name text not null,
    season integer not null default 20262027,
    mode text not null,
    status text not null default 'setup',
    commissioner_user_id text,
    join_code text not null unique,
    cpu_trades boolean not null default true,
    draft_round integer not null default 0,
    draft_pick integer not null default 0,
    draft_paused boolean not null default false,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint manager_leagues_mode_check check (mode in ('true_rosters', 'draft')),
    constraint manager_leagues_status_check check (status in ('setup', 'drafting', 'running', 'finished'))
);

create index if not exists idx_manager_leagues_commissioner
    on public.manager_leagues (commissioner_user_id);

-- ── League teams (one row per NHL franchise per league) ──────────────
-- user_id is the GoTrue user uuid; NULL = CPU-controlled team.
create table if not exists public.manager_league_teams (
    league_id uuid not null references public.manager_leagues (id) on delete cascade,
    team_abbrev text not null,
    user_id text,
    team_name text not null default '',
    draft_slot integer not null default 0,
    created_at timestamptz not null default now(),
    primary key (league_id, team_abbrev)
);

create index if not exists idx_manager_league_teams_user
    on public.manager_league_teams (user_id);

-- ── Manager games (schedule snapshot, one row per NHL game) ──────────
-- home_round / away_round are each franchise's own game index (1..84),
-- so a game can be "home game 4 vs away game 3".
create table if not exists public.manager_games (
    league_id uuid not null references public.manager_leagues (id) on delete cascade,
    nhl_game_id bigint not null,
    date text not null,
    home_abbrev text not null,
    away_abbrev text not null,
    home_round integer not null,
    away_round integer not null,
    status text not null default 'pending',
    home_score numeric(10, 2),
    away_score numeric(10, 2),
    finalized_at timestamptz,
    primary key (league_id, nhl_game_id),
    constraint manager_games_status_check check (status in ('pending', 'live', 'final', 'postponed'))
);

create index if not exists idx_manager_games_date
    on public.manager_games (league_id, date);
create index if not exists idx_manager_games_status
    on public.manager_games (league_id, status);

-- ── Rosters (drafted / initial / traded players per team) ────────────
create table if not exists public.manager_rosters (
    league_id uuid not null references public.manager_leagues (id) on delete cascade,
    team_abbrev text not null,
    player_id bigint not null,
    position text not null,
    acquired_via text not null default 'initial',
    primary key (league_id, team_abbrev, player_id),
    constraint manager_rosters_position_check check (position in ('F', 'D', 'G')),
    constraint manager_rosters_acquired_check check (acquired_via in ('initial', 'draft', 'trade'))
);

create index if not exists idx_manager_rosters_player
    on public.manager_rosters (league_id, player_id);

-- ── Active lineups per team per round (12 F / 6 D / 2 G) ─────────────
create table if not exists public.manager_lineups (
    league_id uuid not null references public.manager_leagues (id) on delete cascade,
    team_abbrev text not null,
    round integer not null,
    slot text not null,
    player_id bigint,
    locked_at timestamptz,
    primary key (league_id, team_abbrev, round, slot),
    constraint manager_lineups_slot_check
        check (slot ~ '^(F([1-9]|1[0-2])|D([1-6])|G[12])$')
);

-- ── Trade offers ─────────────────────────────────────────────────────
create table if not exists public.manager_trade_offers (
    id uuid primary key default gen_random_uuid(),
    league_id uuid not null references public.manager_leagues (id) on delete cascade,
    from_team text not null,
    to_team text not null,
    offered_player_ids bigint[] not null default '{}',
    requested_player_ids bigint[] not null default '{}',
    status text not null default 'pending',
    created_at timestamptz not null default now(),
    responded_at timestamptz,
    constraint manager_trade_offers_status_check
        check (status in ('pending', 'accepted', 'declined', 'cancelled'))
);

create index if not exists idx_manager_trade_offers_to
    on public.manager_trade_offers (league_id, to_team, status);

-- ── Per-game stat lines (score model inputs + fantasy points) ────────
create table if not exists public.manager_game_stats (
    nhl_game_id bigint not null,
    player_id bigint not null,
    goals integer not null default 0,
    a1 integer not null default 0,
    a2 integer not null default 0,
    sog integer not null default 0,
    pent integer not null default 0,
    pend integer not null default 0,
    xgf_5v5 numeric(10, 4) not null default 0,
    xga_5v5 numeric(10, 4) not null default 0,
    gsax numeric(10, 4) not null default 0,
    fantasy_points numeric(10, 2) not null default 0,
    computed_at timestamptz not null default now(),
    primary key (nhl_game_id, player_id)
);

create index if not exists idx_manager_game_stats_player
    on public.manager_game_stats (player_id);

-- ── RLS: enabled, no grants (service role only) ──────────────────────
alter table public.manager_leagues enable row level security;
alter table public.manager_league_teams enable row level security;
alter table public.manager_games enable row level security;
alter table public.manager_rosters enable row level security;
alter table public.manager_lineups enable row level security;
alter table public.manager_trade_offers enable row level security;
alter table public.manager_game_stats enable row level security;
