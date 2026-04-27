create table if not exists public.card_builder_layouts (
    id uuid primary key default gen_random_uuid(),
    auth_user_id uuid not null references auth.users (id) on delete cascade,
    name text not null default 'Untitled card',
    card_type text not null,
    config_json jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now())
);

create index if not exists idx_card_builder_layouts_auth_user_updated
    on public.card_builder_layouts (auth_user_id, updated_at desc);

do $$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'card_builder_layouts_card_type_check'
          and conrelid = 'public.card_builder_layouts'::regclass
    ) then
        alter table public.card_builder_layouts
            add constraint card_builder_layouts_card_type_check check (
                card_type in ('skater', 'goalie', 'team')
            );
    end if;
end $$;

alter table public.card_builder_layouts enable row level security;

drop policy if exists service_role_manage_card_builder_layouts on public.card_builder_layouts;
drop policy if exists authenticated_select_own_card_builder_layouts on public.card_builder_layouts;
drop policy if exists authenticated_insert_own_card_builder_layouts on public.card_builder_layouts;
drop policy if exists authenticated_update_own_card_builder_layouts on public.card_builder_layouts;
drop policy if exists authenticated_delete_own_card_builder_layouts on public.card_builder_layouts;

create policy service_role_manage_card_builder_layouts
    on public.card_builder_layouts
    for all
    to service_role
    using (true)
    with check (true);

create policy authenticated_select_own_card_builder_layouts
    on public.card_builder_layouts
    for select
    to authenticated
    using (auth.uid() = auth_user_id);

create policy authenticated_insert_own_card_builder_layouts
    on public.card_builder_layouts
    for insert
    to authenticated
    with check (auth.uid() = auth_user_id);

create policy authenticated_update_own_card_builder_layouts
    on public.card_builder_layouts
    for update
    to authenticated
    using (auth.uid() = auth_user_id)
    with check (auth.uid() = auth_user_id);

create policy authenticated_delete_own_card_builder_layouts
    on public.card_builder_layouts
    for delete
    to authenticated
    using (auth.uid() = auth_user_id);