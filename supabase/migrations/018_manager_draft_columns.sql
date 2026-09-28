-- ============================================================
-- Manager Game (M2): draft support on top of M1.
-- Draft picks are manager_rosters rows (acquired_via='draft');
-- picked_at gives pick order for the draft log.
-- ============================================================

alter table public.manager_rosters
    add column if not exists picked_at timestamptz not null default now();

create index if not exists idx_manager_rosters_picked
    on public.manager_rosters (league_id, picked_at);
