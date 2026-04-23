alter table public.user_accounts
    add column if not exists subscription_source text,
    add column if not exists stripe_customer_id text,
    add column if not exists stripe_subscription_id text,
    add column if not exists stripe_price_id text,
    add column if not exists stripe_current_period_end timestamptz;

create index if not exists idx_user_accounts_stripe_customer_id
    on public.user_accounts (stripe_customer_id);

create index if not exists idx_user_accounts_stripe_subscription_id
    on public.user_accounts (stripe_subscription_id);

do $$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'user_accounts_subscription_source_check'
          and conrelid = 'public.user_accounts'::regclass
    ) then
        alter table public.user_accounts
            add constraint user_accounts_subscription_source_check check (
                subscription_source is null or subscription_source in ('manual', 'stripe')
            );
    end if;
end $$;