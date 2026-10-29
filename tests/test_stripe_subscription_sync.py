"""Stripe -> user_accounts mapping, and the repair path that backfills it.

Paying customers stayed on "trial ended" because the sync could not persist:
  1. the subscription payload carries no `auth_user_id` (the upsert target) and
     no `email` (NOT NULL on the insert half of PostgREST `merge-duplicates`),
     so it must be merged into a full account row;
  2. `_isoformat_utc(None)` is '', which Postgres rejects for a timestamptz
     column with `22007`;
  3. the live table was missing the Stripe columns entirely (migration 011
     never applied), and the missing-column retry dropped them silently.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('XG_PRELOAD', '0')
os.environ.setdefault('PRESTART_LOGGER', '0')
os.environ.setdefault('PRELOAD_GM_CACHES', '0')

import app.routes as routes  # noqa: E402
from scripts.sync_stripe_subscriptions import (  # noqa: E402
    _account_payload,
    pick_subscription,
)


def _subscription(**overrides):
    sub = {
        'id': 'sub_1',
        'customer': 'cus_1',
        'status': 'active',
        'start_date': 1_700_000_000,
        'current_period_end': 1_700_200_000,
        'cancel_at_period_end': False,
        'items': {'data': [{'price': {'id': 'price_1', 'recurring': {'interval': 'month'}}}]},
        'metadata': {'auth_user_id': 'user-123', 'plan_key': 'monthly'},
    }
    sub.update(overrides)
    return sub


# ── mapping ──────────────────────────────────────────────────────────────────

def test_active_subscription_maps_to_pro_monthly():
    updates = routes._stripe_updates_from_subscription(_subscription())

    assert updates['subscription_plan'] == 'pro'
    assert updates['subscription_status'] == 'active'
    assert updates['billing_interval'] == 'monthly'
    assert updates['subscription_source'] == 'stripe'
    assert updates['stripe_subscription_id'] == 'sub_1'
    assert updates['stripe_customer_id'] == 'cus_1'
    assert updates['stripe_price_id'] == 'price_1'


def test_absent_dates_are_none_not_empty_strings():
    """'' is rejected by a timestamptz column (22007); absent must be NULL."""
    updates = routes._stripe_updates_from_subscription(_subscription())

    assert updates['subscription_ends_at'] is None
    assert updates['subscription_started_at'].endswith('Z')
    assert updates['stripe_current_period_end'].endswith('Z')


def test_missing_dates_are_none_too():
    sub = _subscription()
    sub.pop('start_date')
    sub.pop('current_period_end')

    updates = routes._stripe_updates_from_subscription(sub)

    assert updates['subscription_started_at'] is None
    assert updates['stripe_current_period_end'] is None


def test_cancel_at_period_end_sets_the_end_date():
    updates = routes._stripe_updates_from_subscription(
        _subscription(cancel_at_period_end=True)
    )

    assert updates['subscription_ends_at'] is not None


def test_payload_is_not_a_complete_account_row():
    """It targets a subscription, so it cannot be upserted on its own."""
    updates = routes._stripe_updates_from_subscription(_subscription())

    assert 'auth_user_id' not in updates
    assert 'email' not in updates


# ── repair payload ───────────────────────────────────────────────────────────

def test_account_payload_keeps_the_not_null_columns():
    row = {
        'auth_user_id': 'user-123',
        'email': 'fan@example.com',
        'username': 'fan',
        'display_name': 'Fan',
        'subscription_plan': 'trial',
        'subscription_status': 'trialing',
        'billing_interval': None,
    }
    updates = routes._stripe_updates_from_subscription(_subscription())

    payload = _account_payload('user-123', row, updates, {})

    assert payload['email'] == 'fan@example.com'          # NOT NULL preserved
    assert payload['username'] == 'fan'
    assert payload['subscription_plan'] == 'pro'
    assert payload['subscription_status'] == 'active'
    assert payload['billing_interval'] == 'monthly'
    assert payload['stripe_customer_id'] == 'cus_1'
    assert payload['subscription_ends_at'] is None


def test_account_payload_without_a_row_needs_an_email():
    # No stored row and no GoTrue lookup available -> refuse rather than write
    # a row that violates NOT NULL.
    assert _account_payload('missing-user', None, {'subscription_plan': 'pro'}, {}) is None


# ── choosing between several subscriptions ──────────────────────────────────

def test_live_subscription_beats_a_canceled_one():
    live = {'id': 'sub_live', 'status': 'active', 'created': 1, 'current_period_end': 10}
    dead = {'id': 'sub_dead', 'status': 'canceled', 'created': 99, 'current_period_end': 999}

    assert pick_subscription([dead, live])['id'] == 'sub_live'


def test_newest_wins_among_live_subscriptions():
    older = {'id': 'sub_old', 'status': 'active', 'created': 1, 'current_period_end': 5}
    newer = {'id': 'sub_new', 'status': 'active', 'created': 2, 'current_period_end': 9}

    assert pick_subscription([older, newer])['id'] == 'sub_new'
