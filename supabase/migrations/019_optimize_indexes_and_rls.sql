-- ============================================================
-- 019_optimize_indexes_and_rls
--
-- 1. Add missing indexes matched to actual Rust/PostgREST query patterns.
-- 2. Drop 100% duplicate / PK-prefix-redundant indexes to cut write overhead
--    and Postgres storage bloat.
-- 3. Replace sparse boolean / nullable indexes with compact partial indexes.
-- 4. Fix self-referencing infinite-recursion (42P17) RLS policies on
--    public.user_accounts from migration 012 via a SECURITY DEFINER helper.
-- ============================================================

-- -----------------------------------------------------------
-- 1. Missing high-impact indexes
-- -----------------------------------------------------------

-- rapm / rapm_context: Rust loads by season=eq.{s}, but PK starts with player_id
CREATE INDEX IF NOT EXISTS idx_rapm_season_strength
    ON public.rapm (season, strength_state);
CREATE INDEX IF NOT EXISTS idx_rapm_context_season_strength
    ON public.rapm_context (season, strength_state);

-- pbp: line-tool & shooting routes always filter corsi = 1
CREATE INDEX IF NOT EXISTS idx_pbp_season_game_corsi
    ON public.pbp (season, game_id)
    WHERE corsi = 1;
CREATE INDEX IF NOT EXISTS idx_pbp_season_team_corsi
    ON public.pbp (season, event_team, event_index)
    WHERE corsi = 1 AND event_team IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_pbp_season_opp_corsi
    ON public.pbp (season, opponent, event_index)
    WHERE corsi = 1 AND opponent IS NOT NULL;

-- projections tables: PKs put player_id ahead of model_key / strengthstate
CREATE INDEX IF NOT EXISTS idx_player_current_projections_season_model
    ON public.player_current_projections (season, model_key);
CREATE INDEX IF NOT EXISTS idx_nhl_current_playerprojections_season_ss
    ON public.nhl_current_playerprojections (season, strengthstate);
CREATE INDEX IF NOT EXISTS idx_nhl_player_metrics_season_player_ss
    ON public.nhl_player_metrics (season, nhl_api_player_id, strengthstate);

-- manager_leagues: sweeper filters active statuses only
CREATE INDEX IF NOT EXISTS idx_manager_leagues_active_status
    ON public.manager_leagues (status)
    WHERE status IN ('setup', 'drafting', 'running');
CREATE INDEX IF NOT EXISTS idx_manager_leagues_comm_created
    ON public.manager_leagues (commissioner_user_id, created_at DESC);

-- manager_league_teams: draft order + unique human team per league
CREATE INDEX IF NOT EXISTS idx_manager_league_teams_draft_slot
    ON public.manager_league_teams (league_id, draft_slot);
CREATE UNIQUE INDEX IF NOT EXISTS idx_manager_league_teams_unique_user
    ON public.manager_league_teams (league_id, user_id)
    WHERE user_id IS NOT NULL;

-- manager_games: sweeper, standings, feed, and lineup/trade lock lookups
CREATE INDEX IF NOT EXISTS idx_manager_games_league_status_date
    ON public.manager_games (league_id, status, date, nhl_game_id);
CREATE INDEX IF NOT EXISTS idx_manager_games_home_round
    ON public.manager_games (league_id, home_abbrev, home_round);
CREATE INDEX IF NOT EXISTS idx_manager_games_away_round
    ON public.manager_games (league_id, away_abbrev, away_round);

-- manager_lineups: trade execution deletes moved players by (league_id, player_id)
CREATE INDEX IF NOT EXISTS idx_manager_lineups_league_player
    ON public.manager_lineups (league_id, player_id);

-- manager_trade_offers: league_feed & list_offers order by created_at DESC
CREATE INDEX IF NOT EXISTS idx_manager_trade_offers_league_created
    ON public.manager_trade_offers (league_id, created_at DESC);

-- -----------------------------------------------------------
-- 2. Drop 100% duplicate / PK-prefix-redundant indexes
-- -----------------------------------------------------------

-- user_accounts already has UNIQUE(email)
DROP INDEX IF EXISTS public.idx_user_accounts_email;

-- season_stats PK is (season, season_state, strength_state, player_id)
DROP INDEX IF EXISTS public.idx_seasonstats_season_state;

-- season_stats_teams PK is (season, season_state, strength_state, team)
DROP INDEX IF EXISTS public.idx_ssteams_season;

-- game_data PK is (game_id, player_id, season); composite indexes cover prefixes
DROP INDEX IF EXISTS public.idx_gamedata_game_player;
DROP INDEX IF EXISTS public.idx_gamedata_season;
DROP INDEX IF EXISTS public.idx_gamedata_player;

-- rapm / rapm_context PKs are (player_id, season, strength_state)
DROP INDEX IF EXISTS public.idx_rapm_player_season;
DROP INDEX IF EXISTS public.idx_rapmctx_player_season;

-- pbp & shifts single-column indexes covered by composite indexes
DROP INDEX IF EXISTS public.idx_pbp_game;
DROP INDEX IF EXISTS public.idx_pbp_season;
DROP INDEX IF EXISTS public.idx_shifts_game;
DROP INDEX IF EXISTS public.idx_shifts_season;

-- Replaced by (league_id, status, date, nhl_game_id) above
DROP INDEX IF EXISTS public.idx_manager_games_status;

-- -----------------------------------------------------------
-- 3. Convert sparse boolean / nullable indexes to partial indexes
-- -----------------------------------------------------------

DROP INDEX IF EXISTS public.idx_user_accounts_is_admin;
CREATE INDEX IF NOT EXISTS idx_user_accounts_is_admin_true
    ON public.user_accounts (auth_user_id)
    WHERE is_admin = true;

DROP INDEX IF EXISTS public.idx_user_accounts_stripe_customer_id;
CREATE INDEX IF NOT EXISTS idx_user_accounts_stripe_customer_id_nn
    ON public.user_accounts (stripe_customer_id)
    WHERE stripe_customer_id IS NOT NULL;

DROP INDEX IF EXISTS public.idx_user_accounts_stripe_subscription_id;
CREATE INDEX IF NOT EXISTS idx_user_accounts_stripe_subscription_id_nn
    ON public.user_accounts (stripe_subscription_id)
    WHERE stripe_subscription_id IS NOT NULL;

DROP INDEX IF EXISTS public.idx_manager_league_teams_user;
CREATE INDEX IF NOT EXISTS idx_manager_league_teams_user_nn
    ON public.manager_league_teams (user_id, league_id)
    WHERE user_id IS NOT NULL;

-- -----------------------------------------------------------
-- 4. Fix self-referencing RLS recursion on public.user_accounts (migration 012)
-- -----------------------------------------------------------

CREATE OR REPLACE FUNCTION public.is_admin_user()
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT EXISTS (
        SELECT 1
        FROM public.user_accounts
        WHERE auth_user_id = auth.uid()
          AND is_admin = true
    );
$$;

REVOKE ALL ON FUNCTION public.is_admin_user() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.is_admin_user() TO authenticated, service_role;

DROP POLICY IF EXISTS "authenticated_admin_read_user_accounts" ON public.user_accounts;
CREATE POLICY "authenticated_admin_read_user_accounts"
    ON public.user_accounts
    FOR SELECT
    TO authenticated
    USING (public.is_admin_user());

DROP POLICY IF EXISTS "authenticated_admin_insert_user_accounts" ON public.user_accounts;
CREATE POLICY "authenticated_admin_insert_user_accounts"
    ON public.user_accounts
    FOR INSERT
    TO authenticated
    WITH CHECK (public.is_admin_user());

DROP POLICY IF EXISTS "authenticated_admin_update_user_accounts" ON public.user_accounts;
CREATE POLICY "authenticated_admin_update_user_accounts"
    ON public.user_accounts
    FOR UPDATE
    TO authenticated
    USING (public.is_admin_user())
    WITH CHECK (public.is_admin_user());

DROP POLICY IF EXISTS "authenticated_admin_delete_user_accounts" ON public.user_accounts;
CREATE POLICY "authenticated_admin_delete_user_accounts"
    ON public.user_accounts
    FOR DELETE
    TO authenticated
    USING (public.is_admin_user());
