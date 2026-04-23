create unique index if not exists idx_user_accounts_email_lower_unique
    on public.user_accounts (lower(email));

create unique index if not exists idx_user_accounts_username_lower_unique
    on public.user_accounts (lower(username))
    where nullif(btrim(username), '') is not null;

do $$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'user_accounts_username_format_check'
          and conrelid = 'public.user_accounts'::regclass
    ) then
        alter table public.user_accounts
            add constraint user_accounts_username_format_check check (
                username is null
                or username ~ '^[a-z0-9](?:[a-z0-9._-]{1,30}[a-z0-9])?$'
            );
    end if;
end $$;