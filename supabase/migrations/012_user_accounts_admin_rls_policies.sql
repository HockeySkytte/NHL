alter table public.user_accounts enable row level security;

drop policy if exists service_role_all_user_accounts on public.user_accounts;
drop policy if exists service_role_manage_user_accounts on public.user_accounts;
drop policy if exists authenticated_admin_select_user_accounts on public.user_accounts;
drop policy if exists authenticated_admin_insert_user_accounts on public.user_accounts;
drop policy if exists authenticated_admin_update_user_accounts on public.user_accounts;
drop policy if exists authenticated_admin_delete_user_accounts on public.user_accounts;

create policy service_role_manage_user_accounts
    on public.user_accounts
    for all
    to service_role
    using (true)
    with check (true);

create policy authenticated_admin_select_user_accounts
    on public.user_accounts
    for select
    to authenticated
    using (
        exists (
            select 1
            from public.user_accounts me
            where me.auth_user_id = auth.uid()
              and me.is_admin = true
        )
    );

create policy authenticated_admin_insert_user_accounts
    on public.user_accounts
    for insert
    to authenticated
    with check (
        exists (
            select 1
            from public.user_accounts me
            where me.auth_user_id = auth.uid()
              and me.is_admin = true
        )
    );

create policy authenticated_admin_update_user_accounts
    on public.user_accounts
    for update
    to authenticated
    using (
        exists (
            select 1
            from public.user_accounts me
            where me.auth_user_id = auth.uid()
              and me.is_admin = true
        )
    )
    with check (
        exists (
            select 1
            from public.user_accounts me
            where me.auth_user_id = auth.uid()
              and me.is_admin = true
        )
    );

create policy authenticated_admin_delete_user_accounts
    on public.user_accounts
    for delete
    to authenticated
    using (
        exists (
            select 1
            from public.user_accounts me
            where me.auth_user_id = auth.uid()
              and me.is_admin = true
        )
    );
