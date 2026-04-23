create table if not exists public.user_accounts (
    auth_user_id uuid primary key references auth.users (id) on delete cascade,
    email text not null unique,
    username text,
    display_name text not null,
    is_admin boolean not null default false,
    subscription_status text not null default 'trialing',
    subscription_plan text not null default 'trial',
    billing_interval text,
    trial_started_at timestamptz,
    trial_expires_at timestamptz,
    subscription_started_at timestamptz,
    subscription_ends_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint user_accounts_subscription_status_check check (
        subscription_status in ('inactive', 'trialing', 'active', 'paid', 'canceled', 'past_due', 'expired')
    ),
    constraint user_accounts_billing_interval_check check (
        billing_interval is null or billing_interval in ('monthly', 'yearly')
    )
);

create index if not exists idx_user_accounts_email
    on public.user_accounts (email);

create index if not exists idx_user_accounts_is_admin
    on public.user_accounts (is_admin);

alter table public.user_accounts enable row level security;

do $$
begin
    if not exists (
        select 1
        from pg_policies
        where schemaname = 'public'
          and tablename = 'user_accounts'
          and policyname = 'service_role_all_user_accounts'
    ) then
        create policy service_role_all_user_accounts
            on public.user_accounts
            for all
            using (true)
            with check (true);
    end if;
end $$;