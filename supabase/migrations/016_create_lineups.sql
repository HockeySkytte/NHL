-- 016_create_lineups.sql
-- Store Daily Faceoff lineups + estimated GP per player.
-- Each row is one (team, player) slot in a projected lineup.
-- PK: (team, player_id) so re-scraping the same team overwrites old rows.

create table if not exists public.lineups (
    team           text        not null,
    player_id      integer     not null,
    player_name    text        not null default '',
    position       text        not null default 'F',   -- F | D | G
    line_unit      text        not null default 'EXT',  -- LW1, C1, RW1, LD1, RD1, G1, EXT, ...
    starter        smallint    not null default 0,      -- 1 = in starting lineup (12F+6D+1G), 0 = scratch/extra
    estimated_gp   integer     not null default 0,      -- projected games played for the season
    gp_note        text        not null default '',     -- how the GP was estimated (e.g. 'wtd-avg last 3')

    -- Injury fields (nullable; set only for injured players)
    is_injured     smallint    not null default 0,
    injury_start   date,
    injury_end     date,
    replacement_id  integer,                           -- player_id of the call-up / replacement
    replacement_name text     not null default '',

    season         text        not null default '20262027',  -- season code this lineup belongs to
    source         text        not null default 'dailyfaceoff', -- source of the lineup data
    created_at     timestamptz not null default timezone('utc', now()),
    updated_at     timestamptz not null default timezone('utc', now()),

    constraint lineups_pkey primary key (team, player_id, season)
);

-- Indexes for common queries
create index if not exists idx_lineups_team_season on public.lineups (team, season);
create index if not exists idx_lineups_season on public.lineups (season);
create index if not exists idx_lineups_player on public.lineups (player_id);
create index if not exists idx_lineups_injured on public.lineups (is_injured) where is_injured = 1;

-- Enable RLS — service_role has full access; anon is read-only
alter table public.lineups enable row level security;

drop policy if exists service_role_manage_lineups on public.lineups;
create policy service_role_manage_lineups
    on public.lineups
    for all
    to service_role
    using (true)
    with check (true);

drop policy if exists anon_read_lineups on public.lineups;
create policy anon_read_lineups
    on public.lineups
    for select
    to anon
    using (true);

-- updated_at trigger (reuse pattern from user_accounts)
drop trigger if exists trg_lineups_updated_at on public.lineups;
create or replace function public.set_updated_at()
returns trigger as $$
begin
    new.updated_at = timezone('utc', now());
    return new;
end;
$$ language plpgsql;

create trigger trg_lineups_updated_at
    before update on public.lineups
    for each row
    execute function public.set_updated_at();