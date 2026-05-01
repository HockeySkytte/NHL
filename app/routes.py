from __future__ import annotations

import os
import csv
import re
import math
import bisect
import pickle
import gzip
from datetime import datetime, timedelta, timezone
import threading
import time
from urllib.parse import urlsplit
from typing import Dict, List, Tuple, Optional, Any, Iterable, Iterator

import requests
import joblib       # to load pickled models
from flask import Blueprint, jsonify, render_template, request, current_app, make_response, session, redirect, url_for, flash
import subprocess
import sys
import uuid
import json
import tempfile
import secrets
import pandas as pd
try:
    # Python 3.9+: IANA timezones
    from zoneinfo import ZoneInfo  # type: ignore
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # type: ignore

try:
    import stripe  # type: ignore
except Exception:
    stripe = None  # type: ignore

# ── Supabase (primary data source, fallback to CSV / Sheets) ────
try:
    from app.supabase_client import (
        get_client as _sb_client,
        auth_is_configured as _sb_auth_is_configured,
        auth_admin_create_user as _sb_auth_admin_create_user,
        auth_admin_delete_user as _sb_auth_admin_delete_user,
        auth_admin_get_user as _sb_auth_admin_get_user,
        auth_admin_list_users as _sb_auth_admin_list_users,
        auth_admin_update_user as _sb_auth_admin_update_user,
        auth_sign_in_with_password as _sb_auth_sign_in_with_password,
        delete_card_builder_layout as _sb_delete_card_builder_layout,
        delete_user_account as _sb_delete_user_account,
        get_card_builder_layout as _sb_get_card_builder_layout,
        get_user_account as _sb_get_user_account,
        list_card_builder_layouts as _sb_list_card_builder_layouts,
        list_user_accounts as _sb_list_user_accounts,
        upsert_card_builder_layout as _sb_upsert_card_builder_layout,
        upsert_user_account as _sb_upsert_user_account,
    )
    _SUPABASE_OK = True
except Exception:
    _sb_client = None  # type: ignore
    _sb_auth_is_configured = None  # type: ignore
    _sb_auth_admin_create_user = None  # type: ignore
    _sb_auth_admin_delete_user = None  # type: ignore
    _sb_auth_admin_get_user = None  # type: ignore
    _sb_auth_admin_list_users = None  # type: ignore
    _sb_auth_admin_update_user = None  # type: ignore
    _sb_auth_sign_in_with_password = None  # type: ignore
    _sb_delete_card_builder_layout = None  # type: ignore
    _sb_delete_user_account = None  # type: ignore
    _sb_get_card_builder_layout = None  # type: ignore
    _sb_get_user_account = None  # type: ignore
    _sb_list_card_builder_layouts = None  # type: ignore
    _sb_list_user_accounts = None  # type: ignore
    _sb_upsert_card_builder_layout = None  # type: ignore
    _sb_upsert_user_account = None  # type: ignore
    _SUPABASE_OK = False


main_bp = Blueprint('main', __name__)
_AUTH_TRIAL_DAYS = 7
_AUTH_SESSION_KEY = 'auth_user'
_CSRF_SESSION_KEY = 'csrf_token'
_AUTH_PLAN_OPTIONS = (
    {
        'key': 'monthly',
        'label': 'Pro Monthly',
        'price_label': '$5/month',
        'detail': 'Full access to Projections with monthly billing.',
    },
    {
        'key': 'yearly',
        'label': 'Pro Yearly',
        'price_label': '$40/year',
        'detail': 'Same access with the lowest annual price.',
    },
)
_AUTH_PREMIUM_PAGE_PREFIXES = (
    '/projections',
)
_AUTH_PREMIUM_API_PREFIXES = (
    '/api/projections/',
)
_CARD_BUILDER_CARD_TYPES = {'skater', 'goalie', 'team'}


def _auth_enabled() -> bool:
    try:
        return bool(_sb_auth_is_configured and _sb_auth_is_configured())
    except Exception:
        return False


def _auth_is_premium_path(path: str) -> bool:
    if not path:
        return False
    for prefix in _AUTH_PREMIUM_PAGE_PREFIXES + _AUTH_PREMIUM_API_PREFIXES:
        if path == prefix or path.startswith(prefix):
            return True
    return False


def _safe_next_url(value: Any) -> Optional[str]:
    raw = str(value or '').strip()
    if not raw or not raw.startswith('/') or raw.startswith('//'):
        return None
    parsed = urlsplit(raw)
    if parsed.scheme or parsed.netloc:
        return None
    return raw


def _csrf_token() -> str:
    token = str(session.get(_CSRF_SESSION_KEY) or '').strip()
    if token:
        return token
    token = secrets.token_urlsafe(32)
    session[_CSRF_SESSION_KEY] = token
    return token


def _csrf_validate() -> bool:
    expected = str(session.get(_CSRF_SESSION_KEY) or '').strip()
    provided = str(request.form.get('csrf_token') or request.headers.get('X-CSRF-Token') or '').strip()
    if not expected or not provided:
        return False
    try:
        return secrets.compare_digest(expected, provided)
    except Exception:
        return False


def _require_csrf_form() -> Optional[Any]:
    if _csrf_validate():
        return None
    return make_response('Invalid CSRF token', 400)


def _auth_redirect_target(default: str = '/projections') -> str:
    return _safe_next_url(request.values.get('next')) or default


def _auth_login_target() -> str:
    return _safe_next_url((request.full_path or request.path or '').rstrip('?')) or '/projections'


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    raw = str(value or '').strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace('Z', '+00:00'))
    except Exception:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _isoformat_utc(value: Optional[datetime]) -> str:
    if value is None:
        return ''
    dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def _merge_auth_user_account(base_record: Dict[str, Any], account_record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not account_record:
        return dict(base_record)
    merged = dict(base_record)
    merged.update({
        'user_id': str(account_record.get('auth_user_id') or merged.get('user_id') or '').strip(),
        'email': str(account_record.get('email') or merged.get('email') or '').strip().lower(),
        'username': str(account_record.get('username') or merged.get('username') or '').strip(),
        'display_name': str(account_record.get('display_name') or account_record.get('username') or merged.get('display_name') or 'Account').strip(),
        'trial_started_at': _isoformat_utc(
            _parse_iso_datetime(account_record.get('trial_started_at'))
            or _parse_iso_datetime(merged.get('trial_started_at'))
        ),
        'trial_expires_at': _isoformat_utc(
            _parse_iso_datetime(account_record.get('trial_expires_at'))
            or _parse_iso_datetime(merged.get('trial_expires_at'))
        ),
        'subscription_status': str(account_record.get('subscription_status') or merged.get('subscription_status') or '').strip().lower(),
        'subscription_plan': str(account_record.get('subscription_plan') or merged.get('subscription_plan') or '').strip(),
        'billing_interval': str(account_record.get('billing_interval') or merged.get('billing_interval') or '').strip().lower(),
        'is_admin': bool(account_record.get('is_admin') or merged.get('is_admin')),
        'subscription_source': str(account_record.get('subscription_source') or merged.get('subscription_source') or '').strip(),
        'stripe_customer_id': str(account_record.get('stripe_customer_id') or merged.get('stripe_customer_id') or '').strip(),
        'stripe_subscription_id': str(account_record.get('stripe_subscription_id') or merged.get('stripe_subscription_id') or '').strip(),
        'stripe_price_id': str(account_record.get('stripe_price_id') or merged.get('stripe_price_id') or '').strip(),
        'stripe_current_period_end': _isoformat_utc(
            _parse_iso_datetime(account_record.get('stripe_current_period_end'))
            or _parse_iso_datetime(merged.get('stripe_current_period_end'))
        ),
    })
    return merged


def _auth_record_from_supabase_user(user: Dict[str, Any], account_record: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    user_meta = user.get('user_metadata') or {}
    app_meta = user.get('app_metadata') or {}
    created_at = _parse_iso_datetime(user.get('created_at')) or datetime.now(timezone.utc)
    trial_started = _parse_iso_datetime(user_meta.get('trial_started_at')) or created_at
    trial_expires = _parse_iso_datetime(user_meta.get('trial_expires_at')) or (trial_started + timedelta(days=_AUTH_TRIAL_DAYS))
    base_record = {
        'user_id': str(user.get('id') or '').strip(),
        'email': str(user.get('email') or '').strip(),
        'username': str(user_meta.get('username') or '').strip(),
        'display_name': str(user_meta.get('display_name') or user_meta.get('name') or user.get('email') or 'Account').strip(),
        'created_at': _isoformat_utc(created_at),
        'trial_started_at': _isoformat_utc(trial_started),
        'trial_expires_at': _isoformat_utc(trial_expires),
        'subscription_status': str(app_meta.get('subscription_status') or user_meta.get('subscription_status') or '').strip().lower(),
        'subscription_plan': str(app_meta.get('subscription_plan') or user_meta.get('subscription_plan') or '').strip(),
        'billing_interval': str(app_meta.get('billing_interval') or user_meta.get('billing_interval') or '').strip().lower(),
        'is_admin': bool(app_meta.get('is_admin') or user_meta.get('is_admin') or False),
    }
    return _merge_auth_user_account(base_record, account_record)


def _auth_username_candidate(record: Dict[str, Any], existing_record: Optional[Dict[str, Any]] = None) -> str:
    existing_username = str((existing_record or {}).get('username') or '').strip()
    if existing_username:
        return existing_username
    email_local = str(record.get('email') or '').split('@', 1)[0].strip().lower()
    cleaned_email = re.sub(r'[^a-z0-9._-]+', '', email_local)
    if cleaned_email:
        return cleaned_email[:64]
    cleaned_name = re.sub(r'[^a-z0-9._-]+', '-', str(record.get('display_name') or '').strip().lower()).strip('-')
    if cleaned_name:
        return cleaned_name[:64]
    return ''


def _normalize_username(value: Any) -> str:
    return re.sub(r'[^a-z0-9._-]+', '-', str(value or '').strip().lower()).strip('-')


def _valid_username(value: Any) -> bool:
    username = _normalize_username(value)
    return bool(re.fullmatch(r'[a-z0-9](?:[a-z0-9._-]{1,30}[a-z0-9])?', username))


def _valid_email(value: Any) -> bool:
    return bool(re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', str(value or '').strip().lower()))


def _app_base_url() -> str:
    raw = str(os.getenv('APP_BASE_URL') or '').strip()
    if raw:
        return raw.rstrip('/')
    try:
        return str(request.url_root or '').rstrip('/')
    except Exception:
        return ''


def _absolute_url_for(endpoint: str, **values: Any) -> str:
    path = url_for(endpoint, **values)
    base = _app_base_url()
    return f'{base}{path}' if base else path


def _stripe_secret_key() -> str:
    return str(os.getenv('STRIPE_SECRET_KEY') or '').strip()


def _stripe_webhook_secret() -> str:
    return str(os.getenv('STRIPE_WEBHOOK_SECRET') or '').strip()


def _stripe_price_id(plan_key: str) -> str:
    env_name = {
        'monthly': 'STRIPE_PRICE_MONTHLY_ID',
        'yearly': 'STRIPE_PRICE_YEARLY_ID',
    }.get(str(plan_key or '').strip().lower(), '')
    return str(os.getenv(env_name) or '').strip() if env_name else ''


def _stripe_missing_config(plan_key: Optional[str] = None) -> List[str]:
    missing: List[str] = []
    if stripe is None:
        missing.append('stripe package')
    if not _stripe_secret_key():
        missing.append('STRIPE_SECRET_KEY')
    plans = [plan_key] if plan_key else ['monthly', 'yearly']
    for key in plans:
        env_name = {
            'monthly': 'STRIPE_PRICE_MONTHLY_ID',
            'yearly': 'STRIPE_PRICE_YEARLY_ID',
        }.get(str(key or '').strip().lower())
        if env_name and not str(os.getenv(env_name) or '').strip():
            missing.append(env_name)
    seen: set[str] = set()
    return [item for item in missing if not (item in seen or seen.add(item))]


def _stripe_any_configured() -> bool:
    return any([
        bool(_stripe_secret_key()),
        bool(_stripe_webhook_secret()),
        bool(_stripe_price_id('monthly')),
        bool(_stripe_price_id('yearly')),
    ])


def _stripe_checkout_enabled() -> bool:
    return not _stripe_missing_config()


def _stripe_portal_enabled() -> bool:
    return bool(stripe is not None and _stripe_secret_key())


def _stripe_client() -> Any:
    if stripe is None:
        raise RuntimeError('Stripe SDK is not installed.')
    secret = _stripe_secret_key()
    if not secret:
        raise RuntimeError('STRIPE_SECRET_KEY is not configured.')
    stripe.api_key = secret
    return stripe


def _stripe_datetime(value: Any) -> Optional[datetime]:
    try:
        if value in (None, ''):
            return None
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except Exception:
        return None


def _stripe_status_to_account_status(status: Any) -> str:
    raw = str(status or '').strip().lower()
    if raw in {'active', 'trialing', 'past_due', 'canceled'}:
        return raw
    if raw == 'unpaid':
        return 'past_due'
    if raw == 'incomplete_expired':
        return 'expired'
    return 'inactive'


def _stripe_price_from_subscription(subscription: Dict[str, Any]) -> Dict[str, Any]:
    items = (((subscription or {}).get('items') or {}).get('data') or [])
    if not items:
        return {}
    return dict(items[0].get('price') or {})


def _stripe_interval_from_subscription(subscription: Dict[str, Any]) -> Optional[str]:
    recurring = (_stripe_price_from_subscription(subscription).get('recurring') or {})
    interval = str(recurring.get('interval') or '').strip().lower()
    if interval == 'month':
        return 'monthly'
    if interval == 'year':
        return 'yearly'
    return None


def _account_record_to_auth_record(account_record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not account_record:
        return None
    created_at = _parse_iso_datetime(account_record.get('created_at')) or datetime.now(timezone.utc)
    return {
        'user_id': str(account_record.get('auth_user_id') or '').strip(),
        'email': str(account_record.get('email') or '').strip().lower(),
        'username': str(account_record.get('username') or '').strip(),
        'display_name': str(account_record.get('display_name') or account_record.get('username') or account_record.get('email') or 'Account').strip(),
        'created_at': _isoformat_utc(created_at),
        'trial_started_at': str(account_record.get('trial_started_at') or '').strip(),
        'trial_expires_at': str(account_record.get('trial_expires_at') or '').strip(),
        'subscription_status': str(account_record.get('subscription_status') or '').strip().lower(),
        'subscription_plan': str(account_record.get('subscription_plan') or '').strip(),
        'billing_interval': str(account_record.get('billing_interval') or '').strip().lower(),
        'is_admin': bool(account_record.get('is_admin')),
        'subscription_started_at': str(account_record.get('subscription_started_at') or '').strip(),
        'subscription_ends_at': str(account_record.get('subscription_ends_at') or '').strip(),
        'subscription_source': str(account_record.get('subscription_source') or '').strip(),
        'stripe_customer_id': str(account_record.get('stripe_customer_id') or '').strip(),
        'stripe_subscription_id': str(account_record.get('stripe_subscription_id') or '').strip(),
        'stripe_price_id': str(account_record.get('stripe_price_id') or '').strip(),
        'stripe_current_period_end': str(account_record.get('stripe_current_period_end') or '').strip(),
    }


def _auth_user_for_user_id(auth_user_id: str) -> Optional[Dict[str, Any]]:
    if not auth_user_id:
        return None
    account_record = _sb_get_user_account(auth_user_id) if _sb_get_user_account else None
    if _sb_auth_admin_get_user:
        try:
            auth_row = _sb_auth_admin_get_user(auth_user_id)
        except Exception:
            auth_row = None
        if auth_row:
            return _auth_record_from_supabase_user(auth_row, account_record)
    return _account_record_to_auth_record(account_record)


def _persist_user_account_updates(auth_user_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    auth_user = _auth_user_for_user_id(auth_user_id)
    if not auth_user:
        return None
    payload = _build_account_payload(auth_user, updates)
    return _sb_upsert_user_account(payload) if _sb_upsert_user_account else payload


def _find_user_account_by_stripe_customer_id(customer_id: Any) -> Optional[Dict[str, Any]]:
    wanted = str(customer_id or '').strip()
    if not wanted or not _sb_list_user_accounts:
        return None
    for row in _sb_list_user_accounts() or []:
        if str(row.get('stripe_customer_id') or '').strip() == wanted:
            return row
    return None


def _find_user_account_by_stripe_subscription_id(subscription_id: Any) -> Optional[Dict[str, Any]]:
    wanted = str(subscription_id or '').strip()
    if not wanted or not _sb_list_user_accounts:
        return None
    for row in _sb_list_user_accounts() or []:
        if str(row.get('stripe_subscription_id') or '').strip() == wanted:
            return row
    return None


def _stripe_updates_from_subscription(subscription: Dict[str, Any], *, customer_id: Optional[str] = None) -> Dict[str, Any]:
    price = _stripe_price_from_subscription(subscription)
    price_id = str(price.get('id') or '').strip()
    billing_interval = _stripe_interval_from_subscription(subscription)
    status = _stripe_status_to_account_status(subscription.get('status'))
    started_at = _stripe_datetime(subscription.get('start_date'))
    current_period_end = _stripe_datetime(subscription.get('current_period_end'))
    ended_at = _stripe_datetime(subscription.get('canceled_at') or subscription.get('ended_at'))
    subscription_ends = ended_at or (current_period_end if subscription.get('cancel_at_period_end') else None)
    plan_value = 'pro' if (price_id or billing_interval) else ('canceled' if status in {'canceled', 'expired'} else 'inactive')
    return {
        'subscription_status': status,
        'subscription_plan': plan_value,
        'billing_interval': billing_interval,
        'subscription_started_at': _isoformat_utc(started_at),
        'subscription_ends_at': _isoformat_utc(subscription_ends),
        'subscription_source': 'stripe',
        'stripe_customer_id': str(customer_id or subscription.get('customer') or '').strip() or None,
        'stripe_subscription_id': str(subscription.get('id') or '').strip() or None,
        'stripe_price_id': price_id or None,
        'stripe_current_period_end': _isoformat_utc(current_period_end),
    }


def _sync_stripe_subscription(subscription: Dict[str, Any], *, auth_user_id: Optional[str] = None, customer_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    metadata = subscription.get('metadata') or {}
    resolved_user_id = str(auth_user_id or metadata.get('auth_user_id') or '').strip()
    if not resolved_user_id:
        linked = _find_user_account_by_stripe_subscription_id(subscription.get('id')) or _find_user_account_by_stripe_customer_id(customer_id or subscription.get('customer'))
        resolved_user_id = str((linked or {}).get('auth_user_id') or '').strip()
    if not resolved_user_id:
        return None
    return _persist_user_account_updates(resolved_user_id, _stripe_updates_from_subscription(subscription, customer_id=customer_id))


def _sync_stripe_checkout_session(checkout_session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    metadata = checkout_session.get('metadata') or {}
    auth_user_id = str(metadata.get('auth_user_id') or checkout_session.get('client_reference_id') or '').strip()
    customer_id = str(checkout_session.get('customer') or '').strip()
    subscription_id = str(checkout_session.get('subscription') or '').strip()
    if subscription_id:
        try:
            stripe_client = _stripe_client()
            subscription = stripe_client.Subscription.retrieve(subscription_id)
        except Exception:
            current_app.logger.exception('Stripe checkout sync failed while retrieving subscription.')
            subscription = None
        if subscription:
            return _sync_stripe_subscription(dict(subscription), auth_user_id=auth_user_id, customer_id=customer_id)
    if auth_user_id:
        return _persist_user_account_updates(auth_user_id, {
            'subscription_source': 'stripe',
            'stripe_customer_id': customer_id or None,
            'stripe_subscription_id': subscription_id or None,
        })
    return None


def _stripe_billing_state(auth_user: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    has_customer = bool((auth_user or {}).get('stripe_customer_id'))
    missing = _stripe_missing_config()
    checkout_enabled = not missing
    portal_enabled = bool(_stripe_portal_enabled() and has_customer)
    return {
        'checkout_enabled': checkout_enabled,
        'portal_enabled': portal_enabled,
        'has_customer': has_customer,
        'managed_by_stripe': bool((auth_user or {}).get('subscription_source') == 'stripe' or (auth_user or {}).get('stripe_subscription_id')),
        'partial_config': bool(_stripe_any_configured() and not checkout_enabled),
        'missing_config': missing,
    }


def _stripe_checkout_error_message(exc: Exception) -> str:
    raw = str(exc or '').strip().lower()
    if 'a similar object exists in test mode, but a live mode key was used' in raw:
        return 'Stripe is using a live secret key with test-mode Price IDs. Update STRIPE_PRICE_MONTHLY_ID and STRIPE_PRICE_YEARLY_ID to live prices.'
    if 'a similar object exists in live mode, but a test mode key was used' in raw:
        return 'Stripe is using a test secret key with live-mode Price IDs. Use a matching key/price mode pair.'
    if 'no such price' in raw:
        return 'The configured Stripe Price ID was not found. Verify STRIPE_PRICE_MONTHLY_ID and STRIPE_PRICE_YEARLY_ID in Render.'
    return 'Could not start Stripe checkout right now. Please try again in a moment.'


def _create_stripe_checkout_redirect(auth_user: Dict[str, Any], plan_key: str) -> Any:
    missing = _stripe_missing_config(plan_key)
    if missing:
        flash('Stripe billing is not fully configured yet. Missing: ' + ', '.join(missing) + '.', 'error')
        return redirect(url_for('main.account_page'))
    stripe_client = _stripe_client()
    customer_id = str(auth_user.get('stripe_customer_id') or '').strip() or None
    session_payload: Dict[str, Any] = {
        'mode': 'subscription',
        'line_items': [{'price': _stripe_price_id(plan_key), 'quantity': 1}],
        'success_url': _absolute_url_for('main.account_page') + '?billing=success',
        'cancel_url': _absolute_url_for('main.account_page') + '?billing=canceled',
        'client_reference_id': str(auth_user.get('user_id') or '').strip(),
        'metadata': {
            'auth_user_id': str(auth_user.get('user_id') or '').strip(),
            'plan_key': plan_key,
        },
        'subscription_data': {
            'metadata': {
                'auth_user_id': str(auth_user.get('user_id') or '').strip(),
                'plan_key': plan_key,
            },
        },
        'allow_promotion_codes': True,
    }
    if customer_id:
        session_payload['customer'] = customer_id
    else:
        session_payload['customer_email'] = str(auth_user.get('email') or '').strip().lower()
    try:
        checkout_session = stripe_client.checkout.Session.create(**session_payload)
        checkout_url = str(getattr(checkout_session, 'url', '') or '').strip()
        if not checkout_url:
            raise RuntimeError('Stripe checkout session did not include a redirect URL.')
    except Exception as exc:
        current_app.logger.exception('Stripe checkout creation failed for auth_user_id=%s.', auth_user.get('user_id'))
        flash(_stripe_checkout_error_message(exc), 'error')
        return redirect(url_for('main.account_page'))
    return redirect(checkout_url, code=303)


def _create_stripe_billing_portal_redirect(auth_user: Dict[str, Any]) -> Any:
    if not _stripe_portal_enabled():
        flash('Stripe billing portal is not configured yet.', 'error')
        return redirect(url_for('main.account_page'))
    customer_id = str(auth_user.get('stripe_customer_id') or '').strip()
    if not customer_id:
        flash('No Stripe billing profile exists yet for this account. Start checkout first.', 'error')
        return redirect(url_for('main.account_page'))
    stripe_client = _stripe_client()
    try:
        portal_session = stripe_client.billing_portal.Session.create(
            customer=customer_id,
            return_url=_absolute_url_for('main.account_page'),
        )
        portal_url = str(getattr(portal_session, 'url', '') or '').strip()
        if not portal_url:
            raise RuntimeError('Stripe billing portal session did not include a redirect URL.')
    except Exception:
        current_app.logger.exception('Stripe billing portal creation failed for auth_user_id=%s.', auth_user.get('user_id'))
        flash('Could not open Stripe billing right now. Please try again in a moment.', 'error')
        return redirect(url_for('main.account_page'))
    return redirect(portal_url, code=303)


def _create_stripe_donation_checkout_redirect(auth_user: Optional[Dict[str, Any]], amount_raw: Any, *, guest_email: str = '') -> Any:
    error_redirect_target = url_for('main.account_page') if auth_user else url_for('main.donation_page')
    if not _stripe_portal_enabled():
        flash('Stripe is not configured yet for donations.', 'error')
        return redirect(error_redirect_target)
    try:
        amount = float(str(amount_raw or '').strip())
    except Exception:
        flash('Enter a valid donation amount.', 'error')
        return redirect(error_redirect_target)
    amount_cents = int(round(amount * 100.0))
    if amount_cents < 100:
        flash('Minimum donation is $1.00.', 'error')
        return redirect(error_redirect_target)
    if amount_cents > 500000:
        flash('Maximum donation is $5,000.00 per checkout.', 'error')
        return redirect(error_redirect_target)

    stripe_client = _stripe_client()
    _auth = auth_user or {}
    customer_id = str(_auth.get('stripe_customer_id') or '').strip() or None
    resolved_email = str(_auth.get('email') or '').strip().lower() or guest_email.strip().lower()
    user_id = str(_auth.get('user_id') or '').strip()
    success_base = _absolute_url_for('main.account_page') if auth_user else _absolute_url_for('main.donation_page')
    cancel_base = success_base
    session_payload: Dict[str, Any] = {
        'mode': 'payment',
        'line_items': [{
            'price_data': {
                'currency': 'usd',
                'product_data': {
                    'name': 'NHL Analytics Donation',
                },
                'unit_amount': amount_cents,
            },
            'quantity': 1,
        }],
        'success_url': success_base + '?billing=donation_success',
        'cancel_url': cancel_base + '?billing=donation_canceled',
        'client_reference_id': user_id,
        'metadata': {
            'auth_user_id': user_id,
            'kind': 'donation',
            'amount_cents': str(amount_cents),
        },
    }
    if customer_id:
        session_payload['customer'] = customer_id
    elif resolved_email:
        session_payload['customer_email'] = resolved_email
    try:
        checkout_session = stripe_client.checkout.Session.create(**session_payload)
        checkout_url = str(getattr(checkout_session, 'url', '') or '').strip()
        if not checkout_url:
            raise RuntimeError('Stripe donation checkout session did not include a redirect URL.')
    except Exception:
        current_app.logger.exception('Stripe donation checkout creation failed for auth_user_id=%s.', user_id or 'guest')
        flash('Could not start donation checkout right now. Please try again in a moment.', 'error')
        return redirect(error_redirect_target)
    return redirect(checkout_url, code=303)


def _sync_user_account_from_supabase_user(user: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base_record = _auth_record_from_supabase_user(user)
    auth_user_id = str(base_record.get('user_id') or '').strip()
    if not auth_user_id or not _sb_upsert_user_account:
        return base_record

    existing_record = None
    if _sb_get_user_account:
        try:
            existing_record = _sb_get_user_account(auth_user_id)
        except Exception:
            existing_record = None

    now = datetime.now(timezone.utc)
    override_values = overrides or {}
    trial_started = (
        _parse_iso_datetime(override_values.get('trial_started_at'))
        or _parse_iso_datetime((existing_record or {}).get('trial_started_at'))
        or _parse_iso_datetime(base_record.get('trial_started_at'))
        or now
    )
    trial_expires = (
        _parse_iso_datetime(override_values.get('trial_expires_at'))
        or _parse_iso_datetime((existing_record or {}).get('trial_expires_at'))
        or _parse_iso_datetime(base_record.get('trial_expires_at'))
        or (trial_started + timedelta(days=_AUTH_TRIAL_DAYS))
    )
    payload = {
        'auth_user_id': auth_user_id,
        'email': str(override_values.get('email') or (existing_record or {}).get('email') or base_record.get('email') or '').strip().lower(),
        'username': str(override_values.get('username') or _auth_username_candidate(base_record, existing_record) or '').strip(),
        'display_name': str(override_values.get('display_name') or (existing_record or {}).get('display_name') or base_record.get('display_name') or 'Account').strip(),
        'is_admin': bool(override_values.get('is_admin') if 'is_admin' in override_values else (existing_record or {}).get('is_admin') or False),
        'subscription_status': str(override_values.get('subscription_status') or (existing_record or {}).get('subscription_status') or 'trialing').strip().lower(),
        'subscription_plan': str(override_values.get('subscription_plan') or (existing_record or {}).get('subscription_plan') or 'trial').strip(),
        'billing_interval': str(override_values.get('billing_interval') or (existing_record or {}).get('billing_interval') or '').strip().lower(),
        'trial_started_at': _isoformat_utc(trial_started),
        'trial_expires_at': _isoformat_utc(trial_expires),
        'subscription_started_at': _isoformat_utc(
            _parse_iso_datetime(override_values.get('subscription_started_at'))
            or _parse_iso_datetime((existing_record or {}).get('subscription_started_at'))
        ),
        'subscription_ends_at': _isoformat_utc(
            _parse_iso_datetime(override_values.get('subscription_ends_at'))
            or _parse_iso_datetime((existing_record or {}).get('subscription_ends_at'))
        ),
        'updated_at': _isoformat_utc(now),
    }
    try:
        saved_record = _sb_upsert_user_account(payload)
    except Exception:
        saved_record = None
    return _auth_record_from_supabase_user(user, saved_record or payload)


def _current_auth_record() -> Optional[Dict[str, Any]]:
    raw = session.get(_AUTH_SESSION_KEY)
    if not isinstance(raw, dict) or not raw.get('user_id'):
        return None
    return dict(raw)


def _refresh_current_auth_user() -> Optional[Dict[str, Any]]:
    raw = _current_auth_record()
    if not raw:
        return None
    account_record = None
    if _sb_get_user_account:
        try:
            account_record = _sb_get_user_account(str(raw.get('user_id') or ''))
        except Exception:
            account_record = None
    merged = _merge_auth_user_account(raw, account_record)
    return _set_auth_session(merged)


def _build_account_payload(auth_user: Dict[str, Any], updates: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {
        'auth_user_id': str(auth_user.get('user_id') or '').strip(),
        'email': str(auth_user.get('email') or '').strip().lower(),
        'username': str(auth_user.get('username') or _auth_username_candidate(auth_user) or '').strip(),
        'display_name': str(auth_user.get('display_name') or auth_user.get('username') or auth_user.get('email') or 'Account').strip(),
        'is_admin': bool(auth_user.get('is_admin')),
        'subscription_status': str(auth_user.get('subscription_status') or '').strip().lower() or 'inactive',
        'subscription_plan': str(auth_user.get('subscription_plan') or '').strip() or 'inactive',
        'billing_interval': str(auth_user.get('billing_interval') or '').strip().lower() or None,
        'trial_started_at': auth_user.get('trial_started_at') or None,
        'trial_expires_at': auth_user.get('trial_expires_at') or None,
        'subscription_started_at': auth_user.get('subscription_started_at') or None,
        'subscription_ends_at': auth_user.get('subscription_ends_at') or None,
        'subscription_source': str(auth_user.get('subscription_source') or '').strip() or None,
        'stripe_customer_id': str(auth_user.get('stripe_customer_id') or '').strip() or None,
        'stripe_subscription_id': str(auth_user.get('stripe_subscription_id') or '').strip() or None,
        'stripe_price_id': str(auth_user.get('stripe_price_id') or '').strip() or None,
        'stripe_current_period_end': auth_user.get('stripe_current_period_end') or None,
        'updated_at': _isoformat_utc(datetime.now(timezone.utc)),
    }
    if updates:
        payload.update(updates)
    return payload


def _persist_auth_user_updates(auth_user: Dict[str, Any], updates: Dict[str, Any], *, auth_metadata: Optional[Dict[str, Any]] = None, auth_password: Optional[str] = None) -> Dict[str, Any]:
    payload = _build_account_payload(auth_user, updates)
    saved_record = _sb_upsert_user_account(payload) if _sb_upsert_user_account else payload
    if auth_metadata is not None and _sb_auth_admin_update_user:
        auth_attrs: Dict[str, Any] = {'user_metadata': auth_metadata}
        if auth_password:
            auth_attrs['password'] = auth_password
        _sb_auth_admin_update_user(str(auth_user.get('user_id') or ''), auth_attrs)
    elif auth_password and _sb_auth_admin_update_user:
        _sb_auth_admin_update_user(str(auth_user.get('user_id') or ''), {'password': auth_password})
    return _set_auth_session(_merge_auth_user_account(_current_auth_record() or auth_user, saved_record or payload))


def _subscription_update_for_plan(plan_key: str, *, current_auth_user: Dict[str, Any]) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    if plan_key == 'monthly':
        return {
            'subscription_status': 'active',
            'subscription_plan': 'pro',
            'billing_interval': 'monthly',
            'subscription_started_at': _isoformat_utc(now),
            'subscription_ends_at': None,
        }
    if plan_key == 'yearly':
        return {
            'subscription_status': 'active',
            'subscription_plan': 'pro',
            'billing_interval': 'yearly',
            'subscription_started_at': _isoformat_utc(now),
            'subscription_ends_at': None,
        }
    if plan_key == 'free':
        return {
            'subscription_status': 'active',
            'subscription_plan': 'free',
            'billing_interval': None,
            'subscription_started_at': _isoformat_utc(now),
            'subscription_ends_at': None,
        }
    return {
        'subscription_status': 'canceled',
        'subscription_plan': 'canceled',
        'billing_interval': None,
        'subscription_ends_at': _isoformat_utc(now),
    }


def _require_admin_page() -> Optional[Any]:
    auth_user = _refresh_current_auth_user() or _current_auth_user()
    if not auth_user:
        return redirect(url_for('main.login_page', next=_auth_login_target()))
    if auth_user.get('is_admin'):
        return None
    flash('Admin access required.', 'error')
    return redirect(url_for('main.account_page'))


def _require_admin_api() -> Optional[Any]:
    auth_user = _refresh_current_auth_user() or _current_auth_user()
    if not auth_user:
        return jsonify({'error': 'auth_required', 'loginUrl': url_for('main.login_page', next=_auth_login_target())}), 401
    if auth_user.get('is_admin'):
        return None
    return jsonify({'error': 'admin_required'}), 403


def _find_user_account_by_username(username: Any, *, exclude_user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    wanted = _normalize_username(username)
    if not wanted or not _sb_list_user_accounts:
        return None
    for row in _sb_list_user_accounts() or []:
        user_id = str(row.get('auth_user_id') or '').strip()
        if exclude_user_id and user_id == str(exclude_user_id).strip():
            continue
        if _normalize_username(row.get('username')) == wanted:
            return row
    return None


def _find_user_account_by_email(email: Any, *, exclude_user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    wanted = str(email or '').strip().lower()
    if not wanted or not _sb_list_user_accounts:
        return None
    for row in _sb_list_user_accounts() or []:
        user_id = str(row.get('auth_user_id') or '').strip()
        if exclude_user_id and user_id == str(exclude_user_id).strip():
            continue
        if str(row.get('email') or '').strip().lower() == wanted:
            return row
    return None


def _find_auth_user_by_email(email: Any, *, exclude_user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    wanted = str(email or '').strip().lower()
    if not wanted or not _sb_auth_admin_list_users:
        return None
    for auth_user in _sb_auth_admin_list_users() or []:
        if not isinstance(auth_user, dict):
            continue
        user_id = str(auth_user.get('id') or '').strip()
        if exclude_user_id and user_id == str(exclude_user_id).strip():
            continue
        if str(auth_user.get('email') or '').strip().lower() == wanted:
            return auth_user
    return None


def _user_management_filter_values(source: Optional[Any] = None) -> Dict[str, str]:
    values = source or request.values
    q = str(values.get('filter_q') or values.get('q') or '').strip()
    access = str(values.get('filter_access') or values.get('access') or 'all').strip().lower() or 'all'
    role = str(values.get('filter_role') or values.get('role') or 'all').strip().lower() or 'all'
    if access not in {'all', 'trial', 'free', 'pro', 'inactive'}:
        access = 'all'
    if role not in {'all', 'admin', 'member'}:
        role = 'all'
    return {'q': q, 'access': access, 'role': role}


def _user_management_redirect() -> Any:
    params = _user_management_filter_values()
    query: Dict[str, str] = {}
    if params['q']:
        query['q'] = params['q']
    if params['access'] != 'all':
        query['access'] = params['access']
    if params['role'] != 'all':
        query['role'] = params['role']
    return redirect(url_for('main.user_management_page', **query))


def _ensure_user_account_row(auth_user: Dict[str, Any], existing_account: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    user_id = str((auth_user or {}).get('id') or '').strip()
    if not user_id or not _sb_upsert_user_account:
        return existing_account
    auth_like = _auth_record_from_supabase_user(auth_user, existing_account)
    email = str(auth_like.get('email') or '').strip().lower()
    if not email:
        current_app.logger.warning('Skipping backfill for auth_user_id=%s because email is empty.', user_id)
        return existing_account
    username = _normalize_username(auth_like.get('username'))
    if not _valid_username(username):
        username = ''
    now_iso = _isoformat_utc(datetime.now(timezone.utc))
    payload = {
        'auth_user_id': user_id,
        'email': email,
        'username': username or None,
        'display_name': str(auth_like.get('display_name') or auth_like.get('username') or email).strip(),
        'is_admin': bool(auth_like.get('is_admin')),
        'subscription_status': str(auth_like.get('subscription_status') or '').strip().lower() or 'trialing',
        'subscription_plan': str(auth_like.get('subscription_plan') or '').strip() or 'trial',
        'billing_interval': str(auth_like.get('billing_interval') or '').strip().lower() or None,
        'trial_started_at': auth_like.get('trial_started_at') or None,
        'trial_expires_at': auth_like.get('trial_expires_at') or None,
        'subscription_started_at': auth_like.get('subscription_started_at') or None,
        'subscription_ends_at': auth_like.get('subscription_ends_at') or None,
        'subscription_source': str(auth_like.get('subscription_source') or '').strip() or None,
        'stripe_customer_id': str(auth_like.get('stripe_customer_id') or '').strip() or None,
        'stripe_subscription_id': str(auth_like.get('stripe_subscription_id') or '').strip() or None,
        'stripe_price_id': str(auth_like.get('stripe_price_id') or '').strip() or None,
        'stripe_current_period_end': auth_like.get('stripe_current_period_end') or None,
        'updated_at': now_iso,
    }
    try:
        saved = _sb_upsert_user_account(payload)
    except Exception:
        # Retry without username in case legacy auth metadata creates a uniqueness collision.
        try:
            payload['username'] = None
            saved = _sb_upsert_user_account(payload)
        except Exception:
            current_app.logger.exception('Failed to backfill user_accounts row for auth_user_id=%s.', user_id)
            return existing_account
    if isinstance(saved, dict) and saved.get('auth_user_id'):
        return saved
    if _sb_get_user_account:
        try:
            fetched = _sb_get_user_account(user_id)
        except Exception:
            fetched = None
        if isinstance(fetched, dict) and fetched.get('auth_user_id'):
            return fetched
    current_app.logger.error('Backfill upsert returned no persisted row for auth_user_id=%s.', user_id)
    return None


def _backfill_missing_user_accounts(auth_users: List[Dict[str, Any]], account_by_id: Dict[str, Dict[str, Any]]) -> None:
    for auth_user in auth_users:
        if not isinstance(auth_user, dict):
            continue
        user_id = str(auth_user.get('id') or '').strip()
        if not user_id or user_id in account_by_id:
            continue
        created = _ensure_user_account_row(auth_user)
        if isinstance(created, dict) and created.get('auth_user_id'):
            account_by_id[user_id] = created


def _sync_auth_users_to_accounts(*, only_missing: bool = True) -> Dict[str, Any]:
    auth_users = _sb_auth_admin_list_users() if _sb_auth_admin_list_users else []
    account_by_id = {
        str(row.get('auth_user_id') or ''): row
        for row in (_sb_list_user_accounts() if _sb_list_user_accounts else [])
        if row.get('auth_user_id')
    }
    scanned = 0
    inserted_or_updated = 0
    skipped = 0
    failed = 0
    failed_ids: List[str] = []
    for auth_user in auth_users:
        if not isinstance(auth_user, dict):
            continue
        user_id = str(auth_user.get('id') or '').strip()
        if not user_id:
            continue
        scanned += 1
        existing = account_by_id.get(user_id)
        if only_missing and existing is not None:
            skipped += 1
            continue
        saved = _ensure_user_account_row(auth_user, existing)
        if isinstance(saved, dict) and saved.get('auth_user_id'):
            inserted_or_updated += 1
            account_by_id[user_id] = saved
        else:
            failed += 1
            failed_ids.append(user_id)
    return {
        'scanned': scanned,
        'inserted_or_updated': inserted_or_updated,
        'skipped': skipped,
        'failed': failed,
        'failed_ids': failed_ids[:10],
    }


def _user_management_rows(*, query: str = '', access_filter: str = 'all', role_filter: str = 'all') -> List[Dict[str, Any]]:
    auth_users = _sb_auth_admin_list_users() if _sb_auth_admin_list_users else []
    account_by_id = {
        str(row.get('auth_user_id') or ''): row
        for row in (_sb_list_user_accounts() if _sb_list_user_accounts else [])
        if row.get('auth_user_id')
    }
    _backfill_missing_user_accounts(auth_users, account_by_id)
    rows: List[Dict[str, Any]] = []
    query_value = str(query or '').strip().lower()
    for auth_user in auth_users:
        if not isinstance(auth_user, dict):
            continue
        user_id = str(auth_user.get('id') or '').strip()
        if not user_id:
            continue
        account = account_by_id.get(user_id)
        state = _auth_state_from_record(_auth_record_from_supabase_user(auth_user, account))
        state.update({
            'last_sign_in_at': _isoformat_utc(_parse_iso_datetime(auth_user.get('last_sign_in_at'))),
            'created_at': _isoformat_utc(_parse_iso_datetime(auth_user.get('created_at'))),
            'email_confirmed': bool(auth_user.get('email_confirmed_at') or auth_user.get('confirmed_at')),
        })
        if role_filter == 'admin' and not state.get('is_admin'):
            continue
        if role_filter == 'member' and state.get('is_admin'):
            continue
        if access_filter == 'trial' and state.get('subscription_plan') != 'trial':
            continue
        if access_filter == 'free' and state.get('subscription_plan') != 'free':
            continue
        if access_filter == 'pro' and state.get('billing_interval') not in {'monthly', 'yearly'}:
            continue
        if access_filter == 'inactive' and state.get('has_access'):
            continue
        if query_value:
            haystack = ' '.join([
                str(state.get('email') or ''),
                str(state.get('username') or ''),
                str(state.get('display_name') or ''),
                str(state.get('plan_label') or ''),
                str(state.get('access_label') or ''),
            ]).lower()
            if query_value not in haystack:
                continue
        rows.append(state)
    rows.sort(key=lambda row: (0 if row.get('is_admin') else 1, str(row.get('email') or '').lower()))
    return rows


def _auth_state_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    created_at = _parse_iso_datetime(record.get('created_at')) or datetime.now(timezone.utc)
    trial_started = _parse_iso_datetime(record.get('trial_started_at')) or created_at
    trial_expires = _parse_iso_datetime(record.get('trial_expires_at')) or (trial_started + timedelta(days=_AUTH_TRIAL_DAYS))
    subscription_status = str(record.get('subscription_status') or '').strip().lower()
    now = datetime.now(timezone.utc)
    trial_active = trial_expires > now
    has_subscription = subscription_status in {'active', 'paid'}
    is_admin = bool(record.get('is_admin'))
    has_access = is_admin or has_subscription or trial_active
    remaining_seconds = max(0.0, (trial_expires - now).total_seconds())
    remaining_days = int(math.ceil(remaining_seconds / 86400.0)) if remaining_seconds else 0
    billing_interval = str(record.get('billing_interval') or '').strip().lower()
    subscription_plan = str(record.get('subscription_plan') or '').strip()
    if is_admin:
        access_label = 'Admin access'
    elif subscription_plan == 'free' and has_subscription:
        access_label = 'Free access'
    elif has_subscription:
        access_label = 'Subscription active'
    elif trial_active:
        access_label = f'{remaining_days} day{"s" if remaining_days != 1 else ""} left in trial'
    else:
        access_label = 'Trial ended'
    if is_admin:
        plan_label = 'Admin'
    elif subscription_plan == 'free' and has_subscription:
        plan_label = 'Free access'
    elif has_subscription:
        if billing_interval == 'yearly':
            plan_label = 'Pro yearly'
        elif billing_interval == 'monthly':
            plan_label = 'Pro monthly'
        else:
            plan_label = subscription_plan or 'Pro'
    elif trial_active:
        plan_label = '7-day free trial'
    else:
        plan_label = subscription_plan or 'No active plan'
    out = dict(record)
    out.update({
        'created_at': _isoformat_utc(created_at),
        'trial_started_at': _isoformat_utc(trial_started),
        'trial_expires_at': _isoformat_utc(trial_expires),
        'subscription_status': subscription_status,
        'subscription_plan': subscription_plan,
        'billing_interval': billing_interval,
        'trial_active': trial_active,
        'has_access': has_access,
        'has_subscription': has_subscription,
        'trial_days_remaining': remaining_days,
        'access_label': access_label,
        'plan_label': plan_label,
        'is_authenticated': True,
        'subscription_source': str(record.get('subscription_source') or '').strip(),
        'stripe_customer_id': str(record.get('stripe_customer_id') or '').strip(),
        'stripe_subscription_id': str(record.get('stripe_subscription_id') or '').strip(),
        'stripe_price_id': str(record.get('stripe_price_id') or '').strip(),
        'stripe_current_period_end': _isoformat_utc(_parse_iso_datetime(record.get('stripe_current_period_end'))),
    })
    return out


def _set_auth_session(record: Dict[str, Any]) -> Dict[str, Any]:
    state = _auth_state_from_record(record)
    session[_AUTH_SESSION_KEY] = {
        'user_id': state.get('user_id'),
        'email': state.get('email'),
        'username': state.get('username'),
        'display_name': state.get('display_name'),
        'created_at': state.get('created_at'),
        'trial_started_at': state.get('trial_started_at'),
        'trial_expires_at': state.get('trial_expires_at'),
        'subscription_status': state.get('subscription_status'),
        'subscription_plan': state.get('subscription_plan'),
        'billing_interval': state.get('billing_interval'),
        'is_admin': state.get('is_admin'),
        'subscription_source': state.get('subscription_source'),
        'stripe_customer_id': state.get('stripe_customer_id'),
        'stripe_subscription_id': state.get('stripe_subscription_id'),
        'stripe_price_id': state.get('stripe_price_id'),
        'stripe_current_period_end': state.get('stripe_current_period_end'),
    }
    session.permanent = True
    return state


def _clear_auth_session() -> None:
    session.pop(_AUTH_SESSION_KEY, None)


def _current_auth_user() -> Optional[Dict[str, Any]]:
    raw = session.get(_AUTH_SESSION_KEY)
    if not isinstance(raw, dict) or not raw.get('user_id'):
        return None
    return _auth_state_from_record(raw)


def _auth_error_message(exc: Exception, fallback: str) -> str:
    raw = str(exc or '').strip()
    if not raw:
        return fallback
    lowered = raw.lower()
    if 'already registered' in lowered:
        return 'That email is already registered. Try logging in instead.'
    if 'invalid login credentials' in lowered:
        return 'Invalid email or password.'
    if 'password' in lowered and 'weak' in lowered:
        return 'Choose a stronger password.'
    return raw


def _require_account_page() -> Optional[Any]:
    auth_user = _current_auth_user()
    if auth_user:
        return None
    return redirect(url_for('main.login_page', next=request.path))


def _require_auth_api_user() -> Tuple[Optional[Dict[str, Any]], Optional[Any]]:
    auth_user = _refresh_current_auth_user() or _current_auth_user()
    if auth_user:
        return auth_user, None
    return None, (jsonify({'error': 'auth_required', 'loginUrl': url_for('main.login_page', next=_auth_login_target())}), 401)


def _normalize_card_builder_type(value: Any) -> str:
    raw = str(value or '').strip().lower()
    if raw in _CARD_BUILDER_CARD_TYPES:
        return raw
    return 'skater'


def _normalize_card_builder_layout_id(value: Any) -> str:
    raw = str(value or '').strip()
    if not raw:
        return str(uuid.uuid4())
    try:
        return str(uuid.UUID(raw))
    except Exception as exc:
        raise ValueError('invalid_layout_id') from exc


def _normalize_card_builder_name(value: Any, *, card_type: str) -> str:
    raw = ' '.join(str(value or '').strip().split())
    if not raw:
        raw = f'{card_type.title()} card'
    return raw[:80]


def _normalize_card_builder_config(value: Any, *, card_type: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError('config must be an object')
    try:
        plain = json.loads(json.dumps(value))
    except Exception as exc:
        raise ValueError('config must be JSON serializable') from exc
    if not isinstance(plain, dict):
        raise ValueError('config must be an object')

    raw_grid = plain.get('grid') if isinstance(plain.get('grid'), dict) else {}
    plain['version'] = max(1, min(99, _safe_int(plain.get('version')) or 1))
    plain['cardType'] = _normalize_card_builder_type(plain.get('cardType') or card_type)
    plain['filters'] = plain.get('filters') if isinstance(plain.get('filters'), dict) else {}
    plain['grid'] = {
        'cols': max(8, min(48, _safe_int(raw_grid.get('cols')) or 24)),
        'rows': max(5, min(32, _safe_int(raw_grid.get('rows')) or 13)),
        'show': bool(raw_grid.get('show', True)),
        'snap': bool(raw_grid.get('snap', True)),
    }
    blocks = plain.get('blocks') if isinstance(plain.get('blocks'), list) else []
    plain['blocks'] = [block for block in blocks[:64] if isinstance(block, dict)]

    selected_block_id = str(plain.get('selectedBlockId') or '').strip()
    if selected_block_id:
        plain['selectedBlockId'] = selected_block_id[:64]
    else:
        plain.pop('selectedBlockId', None)

    starter_template = str(plain.get('starterTemplate') or '').strip()
    if starter_template:
        plain['starterTemplate'] = starter_template[:80]
    else:
        plain.pop('starterTemplate', None)

    try:
        payload_size = len(json.dumps(plain, separators=(',', ':')))
    except Exception as exc:
        raise ValueError('config must be JSON serializable') from exc
    if payload_size > 200000:
        raise ValueError('config is too large')

    return plain


def _card_builder_layout_response(row: Dict[str, Any]) -> Dict[str, Any]:
    cfg = row.get('config_json')
    if not isinstance(cfg, dict):
        cfg = {}
    card_type = _normalize_card_builder_type(row.get('card_type') or cfg.get('cardType'))
    return {
        'id': str(row.get('id') or '').strip(),
        'name': _normalize_card_builder_name(row.get('name'), card_type=card_type),
        'cardType': card_type,
        'createdAt': _isoformat_utc(_parse_iso_datetime(row.get('created_at'))),
        'updatedAt': _isoformat_utc(_parse_iso_datetime(row.get('updated_at'))),
        'config': _normalize_card_builder_config(cfg, card_type=card_type),
    }


def _deny_premium_access(auth_user: Optional[Dict[str, Any]]) -> Any:
    next_url = _safe_next_url((request.full_path or request.path or '').rstrip('?')) or request.path or '/projections'
    is_api = (request.path or '').startswith('/api/')
    if not auth_user:
        if is_api:
            return jsonify({'error': 'auth_required', 'loginUrl': url_for('main.login_page', next=next_url)}), 401
        return redirect(url_for('main.login_page', next=next_url))
    if is_api:
        return jsonify({'error': 'trial_expired', 'accountUrl': url_for('main.account_page')}), 403
    return redirect(url_for('main.account_page'))


@main_bp.app_context_processor
def inject_auth_state() -> Dict[str, Any]:
    social_default_title = 'NHL Analytics'
    social_default_description = 'Live games, standings, projections, and deeper NHL analytics.'
    social_default_image = url_for('static', filename='social-preview.png', _external=True)
    social_default_url = request.url
    return {
        'auth_enabled': _auth_enabled(),
        'auth_user': _current_auth_user(),
        'auth_plan_options': _AUTH_PLAN_OPTIONS,
        'auth_login_target': _auth_login_target(),
        'csrf_token': _csrf_token(),
        'social_default_title': social_default_title,
        'social_default_description': social_default_description,
        'social_default_image': social_default_image,
        'social_default_url': social_default_url,
    }


@main_bp.before_app_request
def enforce_auth_for_premium_routes():
    if not _auth_enabled():
        return None
    path = request.path or ''
    if not _auth_is_premium_path(path):
        return None
    auth_user = _current_auth_user()
    if auth_user and auth_user.get('has_access'):
        return None
    return _deny_premium_access(auth_user)

# Update page (no link in app)
@main_bp.route('/admin/update', methods=['GET'])
def update_page():
    guard = _require_admin_page()
    if guard is not None:
        return guard
    return render_template('update.html')

# Optional DB connectivity check for admin use
@main_bp.route('/admin/db-check', methods=['GET'])
def admin_db_check():
    guard = _require_admin_api()
    if guard is not None:
        return guard
    try:
        if _SUPABASE_OK:
            sb = _sb_client()
            # Quick health check: read one row from the teams table
            result = sb.table("teams").select("team").limit(1).execute()
            url = os.getenv('SUPABASE_URL', '(not set)')
            return jsonify({'ok': True, 'backend': 'supabase', 'url': url, 'sample_rows': len(result.data or [])})
        else:
            return jsonify({'ok': False, 'error': 'supabase_not_configured', 'hint': 'Set SUPABASE_URL and SUPABASE_SERVICE_KEY'}), 500
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@main_bp.route('/admin/auth-sync-diagnostics', methods=['GET'])
def admin_auth_sync_diagnostics():
    guard = _require_admin_api()
    if guard is not None:
        return guard
    try:
        auth_users = _sb_auth_admin_list_users() if _sb_auth_admin_list_users else []
    except Exception:
        auth_users = []
    try:
        account_rows = _sb_list_user_accounts() if _sb_list_user_accounts else []
    except Exception:
        account_rows = []
    return jsonify({
        'ok': True,
        'supabase_url': str(os.getenv('SUPABASE_URL') or ''),
        'auth_users_count': len(auth_users or []),
        'user_accounts_count': len(account_rows or []),
        'auth_sample_ids': [str((row or {}).get('id') or '') for row in (auth_users or [])[:5]],
        'account_sample_ids': [str((row or {}).get('auth_user_id') or '') for row in (account_rows or [])[:5]],
    })

# Lightweight in-memory job tracker for admin runs
_ADMIN_JOBS: Dict[str, Dict[str, Any]] = {}

# NHL Edge API cache (per-URL)
_EDGE_API_CACHE: Dict[str, Tuple[float, Any]] = {}

def _jobs_dir() -> str:
    try:
        base = os.getenv('XG_CACHE_DIR') or tempfile.gettempdir()
        d = os.path.join(base, 'nhl_admin_jobs')
        os.makedirs(d, exist_ok=True)
        return d
    except Exception:
        return tempfile.gettempdir()

def _job_status_path(job_id: str) -> str:
    return os.path.join(_jobs_dir(), f'{job_id}.json')

def _persist_job(job_id: str, data: Dict[str, Any]) -> None:
    try:
        with open(_job_status_path(job_id), 'w', encoding='utf-8') as f:
            json.dump(data, f)
    except Exception:
        pass

def _read_job(job_id: str) -> Optional[Dict[str, Any]]:
    # Try memory first
    job = _ADMIN_JOBS.get(job_id)
    if job:
        return job
    # Fallback to disk so other workers can see it
    try:
        p = _job_status_path(job_id)
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        return None
    return None

def _start_admin_job(command: List[str], cwd: str) -> str:
    job_id = str(uuid.uuid4())
    _ADMIN_JOBS[job_id] = {
        'status': 'running',
        'output': '',
        'startedAt': datetime.utcnow().isoformat() + 'Z',
        'command': command,
    }
    _persist_job(job_id, _ADMIN_JOBS[job_id])
    def _runner():
        try:
            res = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
            out = (res.stdout or '') + ("\n" + res.stderr if res.stderr else '')
            _ADMIN_JOBS[job_id]['output'] = out
            _ADMIN_JOBS[job_id]['status'] = 'done' if res.returncode == 0 else 'error'
        except Exception as e:
            _ADMIN_JOBS[job_id]['output'] = str(e)
            _ADMIN_JOBS[job_id]['status'] = 'error'
        finally:
            _ADMIN_JOBS[job_id]['finishedAt'] = datetime.utcnow().isoformat() + 'Z'
            _persist_job(job_id, _ADMIN_JOBS[job_id])
    t = threading.Thread(target=_runner, name=f'admin-job-{job_id}', daemon=True)
    t.start()
    return job_id

@main_bp.route('/admin/job/<job_id>', methods=['GET'])
def get_admin_job(job_id: str):
    guard = _require_admin_api()
    if guard is not None:
        return guard
    if not re.fullmatch(r'[0-9a-fA-F\-]{8,64}', str(job_id or '')):
        return jsonify({'error': 'invalid_job_id'}), 400
    job = _read_job(job_id)
    if not job:
        return jsonify({'error': 'job_not_found'}), 404
    return jsonify({'jobId': job_id, **job})

# Run update_data.py with date (async job)
@main_bp.route('/admin/run-update-data', methods=['POST'])
def run_update_data():
    guard = _require_admin_api()
    if guard is not None:
        return guard
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'Expected JSON body'}), 400
    date = str(data.get('date') or '').strip()
    if date and not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date):
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD.'}), 400
    if not date:
        return jsonify({'error': 'Missing date'}), 400
    try:
        # Resolve project root reliably in both local and Render environments
        project_root = os.path.abspath(os.path.join(current_app.root_path, '..'))
        script_path = os.path.join(project_root, 'scripts', 'update_data.py')
        export_flag = data.get('export', True)
        replace_flag = data.get('replace_date', False)
        season = str(data.get('season') or os.getenv('NHL_SEASON') or '20252026').strip()

        # Projections -> Google Sheets (local-only friendly)
        projections_to_sheets = bool(data.get('projections_to_sheets', True))
        projections_sheet_id = str(
            data.get('projections_sheet_id')
            or os.getenv('PROJECTIONS_SHEET_ID')
            or ''
        ).strip()
        projections_worksheet = str(
            data.get('projections_worksheet')
            or os.getenv('PROJECTIONS_WORKSHEET')
            or 'Sheets3'
        ).strip()

        # Optional: also run RAPM + context refresh (MySQL + Google Sheets)
        run_rapm = bool(data.get('run_rapm', False))
        rebuild_team_seasonstats = bool(data.get('rebuild_team_seasonstats', False))
        rapm_sheet_id = str(
            data.get('rapm_sheet_id')
            or os.getenv('RAPM_SHEET_ID')
            or os.getenv('PROJECTIONS_SHEET_ID')
            or ''
        ).strip()
        rapm_worksheet = str(
            data.get('rapm_worksheet')
            or os.getenv('RAPM_WORKSHEET')
            or 'Sheets4'
        ).strip()
        context_worksheet = str(
            data.get('context_worksheet')
            or os.getenv('CONTEXT_WORKSHEET')
            or 'Sheets5'
        ).strip()

        if projections_to_sheets and not export_flag:
            return jsonify({'error': 'projections_to_sheets requires export=true'}), 400

        # If export was requested, fail fast with a clear message when DB isn't reachable.
        if export_flag:
            try:
                try:
                    from sqlalchemy import create_engine, text  # type: ignore
                except Exception as e:
                    return jsonify({'error': f'sqlalchemy_import_failed: {e}'}), 500
                db_url = os.getenv('DATABASE_URL_RW') or os.getenv('DB_URL_RW') or os.getenv('DATABASE_URL')
                if not db_url:
                    user = os.getenv('DB_USER', 'root')
                    pwd = os.getenv('DB_PASSWORD', '')
                    host = os.getenv('DB_HOST', 'localhost')
                    port = os.getenv('DB_PORT', '3306')
                    name = os.getenv('DB_NAME', 'public')
                    db_url = f"mysql+mysqlconnector://{user}:{pwd}@{host}:{port}/{name}"
                else:
                    host_override = os.getenv('DB_HOST_RW') or os.getenv('DB_HOST')
                    if host_override and '@localhost' in db_url:
                        db_url = db_url.replace('@localhost', f'@{host_override}')

                connect_args: Dict[str, Any] = {'connection_timeout': 3}
                if os.getenv('DB_SSL_CA'):
                    connect_args['ssl_ca'] = str(os.getenv('DB_SSL_CA') or '')
                if os.getenv('DB_SSL_CERT'):
                    connect_args['ssl_cert'] = str(os.getenv('DB_SSL_CERT') or '')
                if os.getenv('DB_SSL_KEY'):
                    connect_args['ssl_key'] = str(os.getenv('DB_SSL_KEY') or '')

                eng = create_engine(db_url, connect_args=connect_args)
                with eng.connect() as conn:
                    conn.execute(text('SELECT 1'))
            except Exception as e:
                # Common on Render when DB_HOST points to a private LAN address.
                return jsonify({
                    'error': (
                        'MySQL is not reachable from this server. '
                        'If you are running on Render, it cannot connect to a LAN/private IP MySQL host. '
                        'Either configure a publicly reachable DB (DATABASE_URL), or uncheck Export to MySQL / projections / RAPM.'
                    ),
                    'details': str(e),
                }), 502

        cmd = [sys.executable, script_path, '--date', date]
        if export_flag:
            cmd.append('--export')
        if replace_flag:
            cmd.append('--replace-date')

        # Ensure we write projections to Google Sheets instead of app/static/player_projections.csv
        if export_flag and projections_to_sheets:
            if not projections_sheet_id:
                return jsonify({'error': 'Missing projections sheet id (set PROJECTIONS_SHEET_ID env var)'}), 400
            cmd.extend(['--projections-sheets-id', projections_sheet_id])
            if projections_worksheet:
                cmd.extend(['--projections-worksheet', projections_worksheet])

        # Explicit season for table names
        if season:
            cmd.extend(['--season', season])

        if run_rapm:
            if not rapm_sheet_id:
                return jsonify({'error': 'Missing RAPM sheet id (set RAPM_SHEET_ID or PROJECTIONS_SHEET_ID env var)'}), 400
            cmd.append('--run-rapm')
            cmd.extend(['--rapm-sheets-id', rapm_sheet_id])
            if rapm_worksheet:
                cmd.extend(['--rapm-worksheet', rapm_worksheet])
            if context_worksheet:
                cmd.extend(['--context-worksheet', context_worksheet])

        if rebuild_team_seasonstats:
            cmd.append('--rebuild-team-seasonstats')

        job_id = _start_admin_job(cmd, cwd=project_root)
        return jsonify({'jobId': job_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run lineups.py for all teams (async job)
@main_bp.route('/admin/run-lineups', methods=['POST'])
def run_lineups():
    guard = _require_admin_api()
    if guard is not None:
        return guard
    try:
        project_root = os.path.abspath(os.path.join(current_app.root_path, '..'))
        script_path = os.path.join(project_root, 'scripts', 'lineups.py')
        cmd = [sys.executable, script_path, '--all']
        job_id = _start_admin_job(cmd, cwd=project_root)
        return jsonify({'jobId': job_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Module-level caches for performance ---
_MODEL_CACHE: Dict[str, Any] = {}
_FEATURE_COLS_CACHE: Dict[str, List[str]] = {}
_PBP_CACHE: Dict[int, Tuple[float, Dict[str, Any]]] = {}

# Player landing cache: {playerId: (timestamp, json)}
_PLAYER_LANDING_CACHE: Dict[int, Tuple[float, Dict[str, Any]]] = {}

# Skaters player-list cache: {(seasonId, teamAbbrev, seasonState): (timestamp, players)}
_SKATERS_PLAYERS_CACHE: Dict[Tuple[int, str, str], Tuple[float, List[Dict[str, Any]]]] = {}

# Goalies player-list cache: {(seasonId, teamAbbrev, seasonState): (timestamp, players)}
_GOALIES_PLAYERS_CACHE: Dict[Tuple[int, str, str], Tuple[float, List[Dict[str, Any]]]] = {}

# Goalie team map (for trend charts): {(playerId, seasonState): (timestamp, {seasonId: teamAbbrev})}
_GOALIES_TEAM_BY_SEASON_MAP_CACHE: Dict[Tuple[int, str], Tuple[float, Dict[int, str]]] = {}

# Static CSV caches
_RAPM_STATIC_CACHE: Optional[Tuple[float, List[Dict[str, Any]]]] = None
_PLAYER_PROJECTIONS_CACHE: Optional[Tuple[float, Dict[int, Dict[str, Any]]]] = None
_CURRENT_PLAYER_PROJECTIONS_CACHE: Optional[Tuple[float, Dict[int, Dict[str, Any]]]] = None
_PLAYER_GAME_PROJECTION_EXPORT_CACHE: Dict[int, Tuple[float, Dict[int, List[Dict[str, Any]]]]] = {}
_PLAYER_GAME_PERFORMANCE_CACHE: Dict[int, Tuple[float, Dict[int, Dict[int, float]]]] = {}
_CONTEXT_STATIC_CACHE: Optional[Tuple[float, List[Dict[str, Any]]]] = None
_SEASONSTATS_STATIC_CACHE: Optional[Tuple[float, List[Dict[str, Any]]]] = None
_TEAMSEASONSTATS_STATIC_CACHE: Optional[Tuple[float, List[Dict[str, Any]]]] = None
_CARD_METRICS_DEF_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_RAPM_PLAYER_STATIC_CACHE: Dict[Tuple[int, Optional[int]], Tuple[float, List[Dict[str, Any]]]] = {}
_CONTEXT_PLAYER_STATIC_CACHE: Dict[Tuple[int, Optional[int]], Tuple[float, List[Dict[str, Any]]]] = {}

_SKATER_PROJECTION_COEFS = {
    'poss_value': 0.098010715415619,
    'off_the_puck': 0.070504541853348,
    'gax': 0.063566677671400,
}
_SEASONSTATS_AGG_CACHE: Dict[Tuple[Any, ...], Tuple[float, Dict[int, Dict[str, Any]], Dict[int, str]]] = {}
_SKATERS_SCATTER_CACHE: Dict[Tuple[Any, ...], Tuple[float, Dict[str, Any]]] = {}
_GOALIES_SCATTER_CACHE: Dict[Tuple[Any, ...], Tuple[float, Dict[str, Any]]] = {}
_GOALIES_GOALTENDING_CACHE: Dict[Tuple[Any, ...], Tuple[float, Dict[str, Any]]] = {}
_PLAYOFF_BRACKET_CACHE: Dict[int, Tuple[float, Dict[str, Any]]] = {}
_TEAM_SEASONS_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}

# Goalies career aggregation helper cache: {(key...): (timestamp, by_pid_season, league_sa_ga)}
_GOALIES_CAREER_MATRIX_CACHE: Dict[
    Tuple[Any, ...],
    Tuple[float, Any, Any],
] = {}
_RAPM_SCALE_CACHE: Dict[Tuple[str, str, str], Tuple[float, Dict[str, Any], Dict[str, Any]]] = {}
_RAPM_CAREER_CACHE: Dict[Tuple[str, str, str], Tuple[float, Dict[str, Any]]] = {}
_SHIFTS_CACHE: Dict[int, Tuple[float, Dict[str, Any]]] = {}
_BOX_CACHE: Dict[int, Tuple[float, Dict[str, Any]]] = {}
_LT_SHIFTS_CACHE: Dict[str, Tuple[float, list]] = {}   # key: "team|season" → (ts, rows)
_LT_PBP_CACHE: Dict[str, Tuple[float, list]] = {}      # key: "season|game_ids_hash|xg_col" → (ts, rows)
_LT_DATA_CACHE: Dict[Tuple[Any, ...], Tuple[float, Dict[str, Any]]] = {}
_LT_SHIFTS_TTL = 300  # 5-minute TTL for line-tool shifts cache
_LT_PBP_TTL = 300

# --- Prestart snapshot config/state ---
_PRESTART_THREAD_STARTED = False
_PRESTART_LOGGED: set[int] = set()  # gameIds captured this process
_PRESTART_CSV_NAME = os.getenv('PRESTART_CSV', 'prestart_snapshots.csv')

def _prestart_csv_path() -> str:
    """Return a writable path for the prestart CSV across environments.
    Priority:
      1) PRESTART_DIR env var
      2) XG_CACHE_DIR (used elsewhere for writable cache on Render)
      3) OS temp dir via _disk_cache_base()
      4) Fallback to project root (may be read-only on some platforms)
    """
    base = os.getenv('PRESTART_DIR') or os.getenv('XG_CACHE_DIR')
    if not base:
        try:
            base = _disk_cache_base()
        except Exception:
            base = None
    if base:
        try:
            os.makedirs(base, exist_ok=True)
        except Exception:
            pass
        return os.path.join(base, _PRESTART_CSV_NAME)
    # Fallbacks
    try:
        return os.path.join(_project_root(), _PRESTART_CSV_NAME)
    except Exception:
        return os.path.join(os.getcwd(), _PRESTART_CSV_NAME)

def _to_decimal_odds(american: Optional[Any]) -> Optional[float]:
    try:
        if american is None:
            return None
        a = float(american)
        if a > 0:
            return 1.0 + (a / 100.0)
        if a < 0:
            return 1.0 + (100.0 / abs(a))
        return None
    except Exception:
        return None

def _bet_fraction_kelly03(prob: Optional[float], american: Optional[Any]) -> Optional[float]:
    try:
        if prob is None:
            return None
        p = float(prob)
        if not (0.0 <= p <= 1.0):
            return None
        dec = _to_decimal_odds(american)
        if dec is None or dec <= 1.0:
            return None
        b = dec - 1.0
        q = 1.0 - p
        f = (b * p - q) / b
        f_scaled = 0.3 * f
        return f_scaled if f_scaled > 0 else 0.0
    except Exception:
        return None

def _append_prestart_row(row: Dict[str, Any]) -> None:
    path = _prestart_csv_path()
    fields = [
        'TimestampUTC','DateET','GameID','StartTimeET',
        'Away','Home',
        'WinAway','WinHome',
        'OddsAway','OddsHome',
        'BetAway','BetHome'
    ]
    try:
        file_exists = os.path.exists(path)
        # Ensure directory exists
        try:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        except Exception:
            pass
        with open(path, 'a', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                w.writeheader()
            # coerce missing keys
            rec = {k: row.get(k) for k in fields}
            w.writerow(rec)
    except Exception:
        # best-effort; do not crash app
        pass

def _build_games_for_date(date_et) -> List[Dict[str, Any]]:
    """Internal helper to construct games with projections and odds for a given ET date (date object)."""
    date_str = str(date_et)
    url = f'https://api-web.nhle.com/v1/schedule/{date_str}'
    try:
        r = requests.get(url, timeout=20)
        js = r.json() if r.status_code == 200 else {}
    except Exception:
        js = {}

    def to_et(iso_utc: Optional[str]) -> Optional[str]:
        if not iso_utc:
            return None
        try:
            s = iso_utc.replace('Z', '+00:00')
            dt = datetime.fromisoformat(s)
            if ZoneInfo is not None:
                et = dt.astimezone(ZoneInfo('America/New_York'))
            else:
                et = dt
            return et.isoformat()
        except Exception:
            return iso_utc

    logo_by_abbrev: Dict[str, str] = {}
    try:
        for tr in TEAM_ROWS:
            ab = (tr.get('Team') or '').upper()
            logo_by_abbrev[ab] = tr.get('Logo') or ''
    except Exception:
        pass

    out: List[Dict[str, Any]] = []
    for wk in (js.get('gameWeek') or []):
        if (wk.get('date') or '')[:10] != date_str:
            continue
        for g in (wk.get('games') or []):
            home = (g.get('homeTeam') or {})
            away = (g.get('awayTeam') or {})
            ha = (home.get('abbrev') or '').upper()
            aa = (away.get('abbrev') or '').upper()
            out.append({
                'id': g.get('id'),
                'season': g.get('season'),
                'gameType': g.get('gameType'),
                'startTimeUTC': g.get('startTimeUTC'),
                'startTimeET': to_et(g.get('startTimeUTC')),
                'gameState': g.get('gameState') or g.get('gameStatus'),
                'venue': g.get('venue'),
                'homeTeam': { 'abbrev': ha, 'score': home.get('score'), 'logo': logo_by_abbrev.get(ha, '') },
                'awayTeam': { 'abbrev': aa, 'score': away.get('score'), 'logo': logo_by_abbrev.get(aa, '') },
                'periodDescriptor': g.get('periodDescriptor'),
            })
    if not out and isinstance(js, dict):
        for g in (js.get('games') or []):
            st = g.get('startTimeUTC') or g.get('gameDate')
            if not isinstance(st, str):
                continue
            if st.replace('Z', '').strip()[:10] != date_str:
                continue
            home = (g.get('homeTeam') or {})
            away = (g.get('awayTeam') or {})
            ha = (home.get('abbrev') or '').upper()
            aa = (away.get('abbrev') or '').upper()
            out.append({
                'id': g.get('id') or g.get('gamePk') or g.get('gameId'),
                'season': g.get('season'),
                'gameType': g.get('gameType') or g.get('gameTypeId'),
                'startTimeUTC': st,
                'startTimeET': to_et(st),
                'gameState': g.get('gameState') or g.get('gameStatus'),
                'venue': g.get('venue'),
                'homeTeam': { 'abbrev': ha, 'score': home.get('score'), 'logo': logo_by_abbrev.get(ha, '') },
                'awayTeam': { 'abbrev': aa, 'score': away.get('score'), 'logo': logo_by_abbrev.get(aa, '') },
                'periodDescriptor': g.get('periodDescriptor'),
            })

    # Compute B2B set for previous day (reused from API)
    prev_date_et = (date_et - timedelta(days=1)).isoformat()
    prev_set: set[str] = set()
    try:
        r2 = requests.get(f'https://api-web.nhle.com/v1/schedule/{prev_date_et}', timeout=20)
        if r2.status_code == 200:
            js2 = r2.json() or {}
            for wk in (js2.get('gameWeek') or []):
                if (wk.get('date') or '')[:10] != prev_date_et:
                    continue
                for g2 in (wk.get('games') or []):
                    home2 = (g2.get('homeTeam') or {})
                    away2 = (g2.get('awayTeam') or {})
                    if home2.get('abbrev'):
                        prev_set.add(str(home2.get('abbrev')).upper())
                    if away2.get('abbrev'):
                        prev_set.add(str(away2.get('abbrev')).upper())
            if not prev_set and isinstance(js2, dict):
                for g2 in (js2.get('games') or []):
                    st2 = g2.get('startTimeUTC') or g2.get('gameDate') or ''
                    if str(st2).replace('Z','').strip()[:10] != prev_date_et:
                        continue
                    home2 = (g2.get('homeTeam') or {})
                    away2 = (g2.get('awayTeam') or {})
                    if home2.get('abbrev'):
                        prev_set.add(str(home2.get('abbrev')).upper())
                    if away2.get('abbrev'):
                        prev_set.add(str(away2.get('abbrev')).upper())
    except Exception:
        prev_set = set()

    # Load lineups and current player projections
    lineups_all = _load_lineups_all()
    proj_map = _load_current_player_projections_cached()
    SITUATION = {
        'Away-B2B-B2B': -0.126602018,
        'Away-B2B-Rested': -0.400515738,
        'Away-Rested-B2B': 0.174538991,
        'Away-Rested-Rested': -0.153396566,
    }
    def situation_for(away_abbrev: str, home_abbrev: str) -> float:
        a_b2b = (away_abbrev.upper() in prev_set)
        h_b2b = (home_abbrev.upper() in prev_set)
        if a_b2b and h_b2b:
            key = 'Away-B2B-B2B'
        elif a_b2b and not h_b2b:
            key = 'Away-B2B-Rested'
        elif (not a_b2b) and h_b2b:
            key = 'Away-Rested-B2B'
        else:
            key = 'Away-Rested-Rested'
        return SITUATION.get(key, 0.0)

    for g in out:
        aa = (g.get('awayTeam') or {}).get('abbrev') or ''
        ha = (g.get('homeTeam') or {}).get('abbrev') or ''
        try:
            proj_away = _team_proj_from_lineup(str(aa), lineups_all, proj_map)
            proj_home = _team_proj_from_lineup(str(ha), lineups_all, proj_map)
            dproj = proj_away - proj_home
            sval = situation_for(str(aa), str(ha))
            win_away = 1.0 / (1.0 + math.exp(-(dproj) - sval))
            win_home = 1.0 - win_away
            g['projections'] = {
                'projAway': round(float(proj_away), 6),
                'projHome': round(float(proj_home), 6),
                'dProj': round(float(dproj), 6),
                'situationValue': round(float(sval), 9),
                'winProbAway': round(float(win_away), 6),
                'winProbHome': round(float(win_home), 6),
            }
        except Exception:
            continue

    game_team_map: Dict[int, Tuple[str, str]] = {}
    for g in out:
        try:
            gid_i = int(str(g.get('id')).strip())
        except Exception:
            continue
        game_team_map[gid_i] = (
            str((g.get('awayTeam') or {}).get('abbrev') or '').strip().upper(),
            str((g.get('homeTeam') or {}).get('abbrev') or '').strip().upper(),
        )
    odds_snapshot_map = _load_latest_odds_snapshot_map(game_team_map)
    try:
        odds_map = _fetch_partner_odds_map(date_str)
    except Exception:
        odds_map = {}
    try:
        from datetime import timezone as _tz
        now_utc = datetime.now(_tz.utc)
        for g in out:
            not_started = False
            try:
                st_raw = g.get('startTimeUTC')
                if isinstance(st_raw, str):
                    se_utc = datetime.fromisoformat(st_raw.replace('Z', '+00:00'))
                    if se_utc.tzinfo is None:
                        se_utc = se_utc.replace(tzinfo=_tz.utc)
                    not_started = now_utc < se_utc
            except Exception:
                not_started = False
            gid = None
            try:
                raw_id = g.get('id')
                if raw_id is not None:
                    gid = int(str(raw_id).strip())
            except Exception:
                gid = None
            if gid is not None:
                snapshot = odds_snapshot_map.get(gid)
                if snapshot and (snapshot.get('oddsAway') is not None or snapshot.get('oddsHome') is not None):
                    if not_started:
                        g['odds'] = {'away': snapshot.get('oddsAway'), 'home': snapshot.get('oddsHome')}
                    else:
                        g['prestart'] = {
                            'oddsAway': snapshot.get('oddsAway'),
                            'oddsHome': snapshot.get('oddsHome'),
                            'winAwayPct': snapshot.get('winAwayPct'),
                            'winHomePct': snapshot.get('winHomePct'),
                            'betAwayPct': snapshot.get('betAwayPct'),
                            'betHomePct': snapshot.get('betHomePct'),
                        }
                elif not_started and gid in odds_map:
                    g['odds'] = odds_map.get(gid)
    except Exception:
        pass

    return out

def _start_prestart_logger_thread_once():
    global _PRESTART_THREAD_STARTED
    if _PRESTART_THREAD_STARTED:
        return
    _PRESTART_THREAD_STARTED = True

    def _runner():
        # Respect optional window seconds (how many seconds before start qualifies)
        # Default prestart window widened to 3600s (1h) to improve chance of capture
        window_secs = 3600
        try:
            window_secs = max(30, int(os.getenv('PRESTART_WINDOW_SECONDS', str(window_secs))))
        except Exception:
            pass
        # Also capture a "grace" period after start to avoid missing games if app boots late.
        # PRESTART_GRACE_SECONDS overrides default of 300 (5 minutes)
        try:
            grace_secs = max(0, int(os.getenv('PRESTART_GRACE_SECONDS', '300')))
        except Exception:
            grace_secs = 300
        while True:
            try:
                # Determine ET date now
                try:
                    if ZoneInfo is None:
                        raise RuntimeError('zoneinfo_unavailable')
                    now_et = datetime.now(ZoneInfo('America/New_York'))
                except Exception:
                    now_et = datetime.utcnow()
                date_et = now_et.date()
                games = _build_games_for_date(date_et)
                # Current time in UTC
                from datetime import timezone as _tz
                now_utc = datetime.now(_tz.utc)
                for g in games:
                    try:
                        raw_id = g.get('id')
                        gid = int(raw_id) if raw_id is not None else None
                    except Exception:
                        gid = None
                    if gid is None or gid in _PRESTART_LOGGED:
                        continue
                    st_raw = g.get('startTimeUTC')
                    if not isinstance(st_raw, str):
                        continue
                    try:
                        se_utc = datetime.fromisoformat(st_raw.replace('Z', '+00:00'))
                        if se_utc.tzinfo is None:
                            se_utc = se_utc.replace(tzinfo=_tz.utc)
                    except Exception:
                        continue
                    # Capture if within prestart window before start OR within grace window after start
                    delta_before = (se_utc - now_utc).total_seconds()
                    delta_after = (now_utc - se_utc).total_seconds()
                    if (0 <= delta_before <= window_secs) or (0 <= delta_after <= grace_secs):
                        # Prepare row
                        away_ab = (g.get('awayTeam') or {}).get('abbrev') or ''
                        home_ab = (g.get('homeTeam') or {}).get('abbrev') or ''
                        win_away = (g.get('projections') or {}).get('winProbAway')
                        win_home = (g.get('projections') or {}).get('winProbHome')
                        odds_away = (g.get('odds') or {}).get('away') if isinstance(g.get('odds'), dict) else None
                        odds_home = (g.get('odds') or {}).get('home') if isinstance(g.get('odds'), dict) else None
                        bet_away = _bet_fraction_kelly03(win_away, odds_away)
                        bet_home = _bet_fraction_kelly03(win_home, odds_home)
                        # Timestamp in UTC ISO
                        ts_utc = datetime.utcnow().isoformat() + 'Z'
                        # DateET and StartTimeET already in record
                        row = {
                            'TimestampUTC': ts_utc,
                            'DateET': str(date_et),
                            'GameID': gid,
                            'StartTimeET': g.get('startTimeET'),
                            'Away': away_ab,
                            'Home': home_ab,
                            'WinAway': round(float(win_away)*100.0, 3) if isinstance(win_away, (int, float)) else None,
                            'WinHome': round(float(win_home)*100.0, 3) if isinstance(win_home, (int, float)) else None,
                            'OddsAway': odds_away,
                            'OddsHome': odds_home,
                            'BetAway': round(float(bet_away)*100.0, 3) if isinstance(bet_away, (int, float)) else None,
                            'BetHome': round(float(bet_home)*100.0, 3) if isinstance(bet_home, (int, float)) else None,
                        }
                        _append_prestart_row(row)
                        _PRESTART_LOGGED.add(gid)
                # Sleep shorter if there are still games not captured; else back off
                remaining = [g for g in games if isinstance(g.get('id'), int) and g.get('id') not in _PRESTART_LOGGED]
                sleep_secs = 20 if remaining else 120
                time.sleep(sleep_secs)
            except Exception:
                # Never crash; sleep and retry
                try:
                    time.sleep(30)
                except Exception:
                    pass

    t = threading.Thread(target=_runner, name='prestart-logger', daemon=True)
    t.start()

def start_prestart_logger():
    """Public entry to start the background prestart logger thread.
    Safe to call multiple times; only starts once per process.
    """
    _start_prestart_logger_thread_once()

def _cache_get(cache: Dict, key, ttl: int) -> Optional[Any]:
    try:
        import time
        ts, val = cache.get(key, (0, None))
        if ts and (time.time() - ts) < ttl:
            return val
    except Exception:
        return None
    return None

def _cache_set(cache: Dict, key, val) -> None:
    try:
        import time
        cache[key] = (time.time(), val)
    except Exception:
        pass


def _cache_prune_ttl_and_size(cache: Dict, *, ttl_s: Optional[float] = None, max_items: Optional[int] = None) -> None:
    """Best-effort pruning for caches storing (timestamp, ...) tuples."""
    try:
        now = time.time()

        if ttl_s is not None:
            try:
                ttl_f = float(ttl_s)
            except Exception:
                ttl_f = None
            if ttl_f and ttl_f > 0:
                expired: List[Any] = []
                for k, v in list(cache.items()):
                    try:
                        ts = float(v[0] or 0.0)
                        if ts <= 0 or (now - ts) >= ttl_f:
                            expired.append(k)
                    except Exception:
                        continue
                for k in expired:
                    try:
                        cache.pop(k, None)
                    except Exception:
                        pass

        if max_items is not None:
            try:
                m = int(max_items)
            except Exception:
                m = 0
            if m > 0 and len(cache) > m:
                try:
                    items = sorted(cache.items(), key=lambda kv: float((kv[1] or (0,))[0] or 0.0))
                    to_drop = max(0, len(items) - m)
                    for i in range(to_drop):
                        try:
                            cache.pop(items[i][0], None)
                        except Exception:
                            pass
                except Exception:
                    while len(cache) > m:
                        try:
                            cache.pop(next(iter(cache)), None)
                        except Exception:
                            break
    except Exception:
        return


def _cache_set_multi_bounded(cache: Dict, key, *vals, ttl_s: Optional[float] = None, max_items: Optional[int] = None) -> None:
    """Set cache entry as (timestamp, *vals) and prune by TTL and size."""
    try:
        cache[key] = (time.time(), *vals)
    except Exception:
        return
    _cache_prune_ttl_and_size(cache, ttl_s=ttl_s, max_items=max_items)


def _dict_set_bounded(cache: Dict, key, val, *, max_items: Optional[int] = None) -> None:
    """Best-effort size bounding for plain dict caches (no timestamp tuples).

    Uses insertion order as a cheap LRU approximation.
    """
    try:
        if key in cache:
            try:
                cache.pop(key, None)
            except Exception:
                pass
        cache[key] = val
        if max_items is None:
            return
        try:
            m = int(max_items)
        except Exception:
            m = 0
        if m > 0:
            while len(cache) > m:
                try:
                    cache.pop(next(iter(cache)), None)
                except Exception:
                    break
    except Exception:
        return
    
# --- Small on-disk cache utilities (persist across restarts) ---
def _disk_cache_base() -> str:
    base = os.getenv('XG_CACHE_DIR')
    if base:
        return base
    try:
        if os.name == 'nt':
            import tempfile
            return os.path.join(tempfile.gettempdir(), 'nhl_cache')
        return '/tmp/nhl_cache'
    except Exception:
        return '/tmp/nhl_cache'

def _disk_cache_path_pbp(game_id: int) -> str:
    d = _disk_cache_base()
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return os.path.join(d, f'pbp_{int(game_id)}.json')

def _disk_cache_path_shifts(game_id: int) -> str:
    d = _disk_cache_base()
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return os.path.join(d, f'shifts_{int(game_id)}.json')
@main_bp.route('/')
def index_page():
    """Landing page."""
    return render_template('home.html', teams=TEAM_ROWS, active_tab='Home')


@main_bp.route('/schedule')
def schedule_page():
    """Schedule view."""
    return render_template('index.html', teams=TEAM_ROWS, active_tab='Schedule', show_season_state=True)


@main_bp.route('/live')
def live_games_page():
    """Live Games page."""
    return render_template('live.html', teams=TEAM_ROWS, active_tab='Live Games', show_season_state=False)


@main_bp.route('/standings')
def standings_page():
    """Standings page."""
    # Provide seasons list for the template (used to seed UI/state)
    seasons = []
    try:
        csv_path = os.path.join(os.getcwd(), 'Last_date.csv')
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                rdr = csv.DictReader(f)
                seasons = sorted({int(row['Season']) for row in rdr if row.get('Season')}, reverse=True)
    except Exception:
        seasons = []
    # Convert to list of objects for template parity
    season_objs = [{ 'season': s } for s in seasons]
    return render_template('standings.html', teams=TEAM_ROWS, seasons=season_objs, active_tab='Standings', show_season_state=False)


@main_bp.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        if not _auth_enabled():
            flash('Auth is not configured in this environment yet.', 'error')
            return render_template('login.html', active_tab=None, show_filters=False, next_url=_auth_redirect_target())
        email = str(request.form.get('email') or '').strip().lower()
        password = str(request.form.get('password') or '')
        next_url = _auth_redirect_target()
        if not email or not password:
            flash('Enter both email and password.', 'error')
            return render_template('login.html', active_tab=None, show_filters=False, next_url=next_url)
        try:
            response = _sb_auth_sign_in_with_password(email, password)
            user = (response or {}).get('user') or {}
            if not isinstance(user, dict) or not user.get('id'):
                raise ValueError('Invalid email or password.')
            auth_user = _set_auth_session(_sync_user_account_from_supabase_user(user))
        except Exception as exc:
            flash(_auth_error_message(exc, 'Unable to log in right now.'), 'error')
            return render_template('login.html', active_tab=None, show_filters=False, next_url=next_url)
        flash('Logged in.', 'success')
        if not auth_user.get('has_access') and _auth_is_premium_path(next_url):
            return redirect(url_for('main.account_page'))
        return redirect(next_url)
    return render_template('login.html', active_tab=None, show_filters=False, next_url=_auth_redirect_target())


@main_bp.route('/signup', methods=['GET', 'POST'])
def signup_page():
    if request.method == 'POST':
        if not _auth_enabled():
            flash('Auth is not configured in this environment yet.', 'error')
            return render_template('signup.html', active_tab=None, show_filters=False, next_url=_auth_redirect_target())
        name = str(request.form.get('name') or '').strip()
        email = str(request.form.get('email') or '').strip().lower()
        password = str(request.form.get('password') or '')
        confirm_password = str(request.form.get('confirm_password') or '')
        next_url = _auth_redirect_target()
        if not name:
            flash('Enter your name.', 'error')
            return render_template('signup.html', active_tab=None, show_filters=False, next_url=next_url)
        if not re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email):
            flash('Enter a valid email address.', 'error')
            return render_template('signup.html', active_tab=None, show_filters=False, next_url=next_url)
        if len(password) < 8:
            flash('Password must be at least 8 characters.', 'error')
            return render_template('signup.html', active_tab=None, show_filters=False, next_url=next_url)
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('signup.html', active_tab=None, show_filters=False, next_url=next_url)
        now = datetime.now(timezone.utc)
        trial_expires = now + timedelta(days=_AUTH_TRIAL_DAYS)
        try:
            created_response = _sb_auth_admin_create_user(
                email,
                password,
                user_metadata={
                    'display_name': name,
                    'trial_started_at': _isoformat_utc(now),
                    'trial_expires_at': _isoformat_utc(trial_expires),
                    'trial_days': _AUTH_TRIAL_DAYS,
                    'subscription_status': 'trialing',
                    'subscription_plan': 'trial',
                },
            )
            response = _sb_auth_sign_in_with_password(email, password)
            user = (response or {}).get('user') or {}
            if not isinstance(user, dict) or not user.get('id'):
                raise ValueError('Account created, but automatic login failed. Please log in.')
            created_user = (created_response or {}).get('user') or {}
            fallback_name = str((created_user or {}).get('email') or email).strip()
            _set_auth_session(
                _sync_user_account_from_supabase_user(
                    user,
                    overrides={
                        'display_name': name or fallback_name,
                        'subscription_status': 'trialing',
                        'subscription_plan': 'trial',
                        'trial_started_at': now,
                        'trial_expires_at': trial_expires,
                    },
                )
            )
        except Exception as exc:
            flash(_auth_error_message(exc, 'Unable to create your account right now.'), 'error')
            return render_template('signup.html', active_tab=None, show_filters=False, next_url=next_url)
        flash(f'Account created. Your {_AUTH_TRIAL_DAYS}-day trial starts now.', 'success')
        return redirect(next_url)
    return render_template('signup.html', active_tab=None, show_filters=False, next_url=_auth_redirect_target())


@main_bp.route('/account')
def account_page():
    guard = _require_account_page()
    if guard is not None:
        return guard
    auth_user = _refresh_current_auth_user() or _current_auth_user()
    billing_status = str(request.args.get('billing') or '').strip().lower()
    billing_banner = None
    if billing_status == 'success':
        billing_banner = {
            'category': 'success' if auth_user.get('has_subscription') else 'info',
            'title': 'Stripe checkout complete',
            'detail': 'Your billing update has been sent to the app. If the plan label has not updated yet, refresh again in a few seconds while the webhook finishes syncing.',
        }
    elif billing_status == 'canceled':
        billing_banner = {
            'category': 'info',
            'title': 'Stripe checkout canceled',
            'detail': 'No billing changes were applied.',
        }
    elif billing_status == 'donation_success':
        billing_banner = {
            'category': 'success',
            'title': 'Thank you for your donation',
            'detail': 'Your support helps keep the app running and improving.',
        }
    elif billing_status == 'donation_canceled':
        billing_banner = {
            'category': 'info',
            'title': 'Donation checkout canceled',
            'detail': 'No donation was processed.',
        }
    return render_template(
        'account.html',
        active_tab='Account',
        show_filters=False,
        plan_options=_AUTH_PLAN_OPTIONS,
        auth_user=auth_user,
        billing=_stripe_billing_state(auth_user),
        billing_banner=billing_banner,
    )


@main_bp.route('/api/card-builder/layouts', methods=['GET'])
def api_card_builder_layouts():
    auth_user, guard = _require_auth_api_user()
    if guard is not None:
        return guard
    auth_user_id = str((auth_user or {}).get('user_id') or '').strip()
    rows = _sb_list_card_builder_layouts(auth_user_id) if _sb_list_card_builder_layouts else []
    layouts = [_card_builder_layout_response(row) for row in rows if isinstance(row, dict)]
    return jsonify({'layouts': layouts, 'storageAvailable': bool(_sb_list_card_builder_layouts)})


@main_bp.route('/api/card-builder/layouts', methods=['POST'])
def api_card_builder_save_layout():
    auth_user, guard = _require_auth_api_user()
    if guard is not None:
        return guard
    if not _csrf_validate():
        return jsonify({'error': 'invalid_csrf'}), 400
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({'error': 'invalid_payload'}), 400
    if not _sb_upsert_card_builder_layout:
        return jsonify({'error': 'storage_unavailable'}), 503

    card_type = _normalize_card_builder_type(body.get('cardType'))
    try:
        layout_id = _normalize_card_builder_layout_id(body.get('id'))
        name = _normalize_card_builder_name(body.get('name'), card_type=card_type)
        config = _normalize_card_builder_config(body.get('config'), card_type=card_type)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    auth_user_id = str((auth_user or {}).get('user_id') or '').strip()
    payload = {
        'id': layout_id,
        'auth_user_id': auth_user_id,
        'name': name,
        'card_type': card_type,
        'config_json': config,
        'updated_at': _isoformat_utc(datetime.now(timezone.utc)),
    }
    saved = _sb_upsert_card_builder_layout(payload)
    if not isinstance(saved, dict):
        return jsonify({'error': 'storage_unavailable'}), 503
    return jsonify({'ok': True, 'layout': _card_builder_layout_response(saved)})


@main_bp.route('/api/card-builder/layouts/<layout_id>', methods=['DELETE'])
def api_card_builder_delete_layout(layout_id: str):
    auth_user, guard = _require_auth_api_user()
    if guard is not None:
        return guard
    if not _csrf_validate():
        return jsonify({'error': 'invalid_csrf'}), 400
    if not _sb_delete_card_builder_layout:
        return jsonify({'error': 'storage_unavailable'}), 503
    try:
        layout_id_norm = _normalize_card_builder_layout_id(layout_id)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    auth_user_id = str((auth_user or {}).get('user_id') or '').strip()
    _sb_delete_card_builder_layout(auth_user_id, layout_id_norm)
    return jsonify({'ok': True, 'id': layout_id_norm})


@main_bp.route('/account/plan', methods=['POST'])
def account_plan_update_page():
    guard = _require_account_page()
    if guard is not None:
        return guard
    csrf_guard = _require_csrf_form()
    if csrf_guard is not None:
        return csrf_guard
    auth_user = _refresh_current_auth_user() or _current_auth_user()
    plan_key = str(request.form.get('plan') or '').strip().lower()
    if plan_key not in {'monthly', 'yearly'}:
        flash('Choose a valid plan.', 'error')
        return redirect(url_for('main.account_page'))
    if str(auth_user.get('subscription_plan') or '').strip() == 'free' and str(auth_user.get('subscription_status') or '').strip().lower() == 'active':
        flash('This account already has free access. No Stripe checkout is needed.', 'info')
        return redirect(url_for('main.account_page'))
    if _stripe_any_configured():
        if auth_user.get('stripe_subscription_id'):
            return _create_stripe_billing_portal_redirect(auth_user)
        return _create_stripe_checkout_redirect(auth_user, plan_key)
    _persist_auth_user_updates(auth_user, _subscription_update_for_plan(plan_key, current_auth_user=auth_user))
    flash(f"Plan updated to {'Pro Monthly' if plan_key == 'monthly' else 'Pro Yearly'}.", 'success')
    return redirect(url_for('main.account_page'))


@main_bp.route('/account/billing', methods=['POST'])
def account_billing_portal_page():
    guard = _require_account_page()
    if guard is not None:
        return guard
    csrf_guard = _require_csrf_form()
    if csrf_guard is not None:
        return csrf_guard
    auth_user = _refresh_current_auth_user() or _current_auth_user()
    return _create_stripe_billing_portal_redirect(auth_user)


@main_bp.route('/account/donate', methods=['POST'])
def account_donate_page():
    guard = _require_account_page()
    if guard is not None:
        return guard
    csrf_guard = _require_csrf_form()
    if csrf_guard is not None:
        return csrf_guard
    auth_user = _refresh_current_auth_user() or _current_auth_user()
    return _create_stripe_donation_checkout_redirect(auth_user, request.form.get('donation_amount'))


@main_bp.route('/donate', methods=['POST'])
def public_donate_page():
    csrf_guard = _require_csrf_form()
    if csrf_guard is not None:
        return csrf_guard
    guest_email = str(request.form.get('guest_email') or '').strip().lower()
    auth_user = _current_auth_user()
    return _create_stripe_donation_checkout_redirect(auth_user or None, request.form.get('donation_amount'), guest_email=guest_email)


@main_bp.route('/account/unsubscribe', methods=['POST'])
def account_unsubscribe_page():
    guard = _require_account_page()
    if guard is not None:
        return guard
    csrf_guard = _require_csrf_form()
    if csrf_guard is not None:
        return csrf_guard
    auth_user = _refresh_current_auth_user() or _current_auth_user()
    if _stripe_any_configured() and auth_user.get('stripe_customer_id'):
        return _create_stripe_billing_portal_redirect(auth_user)
    if str(request.form.get('confirm_unsubscribe') or '').strip() != '1':
        flash('Confirm the unsubscribe action to continue.', 'error')
        return redirect(url_for('main.account_page'))
    _persist_auth_user_updates(auth_user, _subscription_update_for_plan('unsubscribe', current_auth_user=auth_user))
    flash('Subscription canceled. Projections will stay locked until you reactivate a plan.', 'success')
    return redirect(url_for('main.account_page'))


@main_bp.route('/account/profile', methods=['POST'])
def account_profile_update_page():
    guard = _require_account_page()
    if guard is not None:
        return guard
    csrf_guard = _require_csrf_form()
    if csrf_guard is not None:
        return csrf_guard
    auth_user = _refresh_current_auth_user() or _current_auth_user()
    username = _normalize_username(request.form.get('username'))
    if not _valid_username(username):
        flash('Username must be 3-32 characters and use letters, numbers, dots, dashes, or underscores.', 'error')
        return redirect(url_for('main.account_page'))
    current_record = _sb_get_user_account(str(auth_user.get('user_id') or '')) if _sb_get_user_account else None
    if current_record and str(current_record.get('username') or '').strip().lower() == username:
        flash('Username unchanged.', 'info')
        return redirect(url_for('main.account_page'))
    existing_username = _find_user_account_by_username(username, exclude_user_id=str(auth_user.get('user_id') or ''))
    if existing_username:
        flash('That username is already in use. Choose another one.', 'error')
        return redirect(url_for('main.account_page'))
    _persist_auth_user_updates(
        auth_user,
        {'username': username},
        auth_metadata={
            'display_name': auth_user.get('display_name') or username,
            'username': username,
            'is_admin': bool(auth_user.get('is_admin')),
        },
    )
    flash('Username updated.', 'success')
    return redirect(url_for('main.account_page'))


@main_bp.route('/account/password', methods=['POST'])
def account_password_update_page():
    guard = _require_account_page()
    if guard is not None:
        return guard
    csrf_guard = _require_csrf_form()
    if csrf_guard is not None:
        return csrf_guard
    auth_user = _refresh_current_auth_user() or _current_auth_user()
    password = str(request.form.get('password') or '')
    confirm_password = str(request.form.get('confirm_password') or '')
    if len(password) < 8:
        flash('Password must be at least 8 characters.', 'error')
        return redirect(url_for('main.account_page'))
    if password != confirm_password:
        flash('Passwords do not match.', 'error')
        return redirect(url_for('main.account_page'))
    _persist_auth_user_updates(auth_user, {}, auth_password=password)
    flash('Password updated. Use the new password the next time you log in.', 'success')
    return redirect(url_for('main.account_page'))


@main_bp.route('/account/delete', methods=['POST'])
def account_delete_page():
    guard = _require_account_page()
    if guard is not None:
        return guard
    csrf_guard = _require_csrf_form()
    if csrf_guard is not None:
        return csrf_guard
    auth_user = _refresh_current_auth_user() or _current_auth_user()
    confirmation = str(request.form.get('confirmation') or '').strip().upper()
    if confirmation != 'DELETE':
        flash('Type DELETE to confirm account deletion.', 'error')
        return redirect(url_for('main.account_page'))
    if _sb_auth_admin_delete_user:
        _sb_auth_admin_delete_user(str(auth_user.get('user_id') or ''))
    _clear_auth_session()
    flash('Account deleted and access removed.', 'success')
    return redirect(url_for('main.login_page'))


@main_bp.route('/admin/users')
def user_management_page():
    guard = _require_admin_page()
    if guard is not None:
        return guard
    auth_user = _refresh_current_auth_user() or _current_auth_user()
    filters = _user_management_filter_values(request.args)
    users = _user_management_rows(query=filters['q'], access_filter=filters['access'], role_filter=filters['role'])
    counts = {
        'total': len(users),
        'admins': sum(1 for user in users if user.get('is_admin')),
        'trial': sum(1 for user in users if user.get('subscription_plan') == 'trial'),
        'free': sum(1 for user in users if user.get('subscription_plan') == 'free'),
        'pro': sum(1 for user in users if user.get('billing_interval') in {'monthly', 'yearly'}),
    }
    return render_template('user_management.html', active_tab='User Management', show_filters=False, auth_user=auth_user, users=users, plan_options=_AUTH_PLAN_OPTIONS, filters=filters, counts=counts)


@main_bp.route('/admin/users/sync', methods=['POST'])
def user_management_sync_page():
    guard = _require_admin_page()
    if guard is not None:
        return guard
    csrf_guard = _require_csrf_form()
    if csrf_guard is not None:
        return csrf_guard
    results = _sync_auth_users_to_accounts(only_missing=True)
    if results.get('failed'):
        failed_ids = ', '.join(results.get('failed_ids') or [])
        flash(
            'Sync completed with failures. '
            f"scanned={results.get('scanned', 0)}, "
            f"saved={results.get('inserted_or_updated', 0)}, "
            f"skipped={results.get('skipped', 0)}, "
            f"failed={results.get('failed', 0)}"
            + (f". Sample failed ids: {failed_ids}" if failed_ids else ''),
            'error',
        )
    else:
        flash(
            'Sync complete. '
            f"scanned={results.get('scanned', 0)}, "
            f"saved={results.get('inserted_or_updated', 0)}, "
            f"skipped={results.get('skipped', 0)}",
            'success',
        )
    return _user_management_redirect()


@main_bp.route('/admin/users/create', methods=['POST'])
def user_management_create_page():
    guard = _require_admin_page()
    if guard is not None:
        return guard
    csrf_guard = _require_csrf_form()
    if csrf_guard is not None:
        return csrf_guard
    username = _normalize_username(request.form.get('username'))
    email = str(request.form.get('email') or '').strip().lower()
    password = str(request.form.get('password') or '')
    confirm_password = str(request.form.get('confirm_password') or '')
    access = str(request.form.get('access') or 'trial').strip().lower()
    is_admin = str(request.form.get('is_admin') or '').lower() in {'1', 'true', 'on', 'yes'}
    if not _valid_username(username):
        flash('Username must be 3-32 characters and use letters, numbers, dots, dashes, or underscores.', 'error')
        return _user_management_redirect()
    if not _valid_email(email):
        flash('Enter a valid email address.', 'error')
        return _user_management_redirect()
    if _find_user_account_by_username(username) is not None:
        flash('That username is already in use.', 'error')
        return _user_management_redirect()
    if _find_user_account_by_email(email) is not None or _find_auth_user_by_email(email) is not None:
        flash('That email address already has an account.', 'error')
        return _user_management_redirect()
    if len(password) < 8:
        flash('Password must be at least 8 characters.', 'error')
        return _user_management_redirect()
    if password != confirm_password:
        flash('Passwords do not match.', 'error')
        return _user_management_redirect()
    now = datetime.now(timezone.utc)
    trial_expires = now + timedelta(days=_AUTH_TRIAL_DAYS)
    created = _sb_auth_admin_create_user(
        email,
        password,
        user_metadata={
            'display_name': username,
            'username': username,
            'is_admin': is_admin,
            'trial_started_at': _isoformat_utc(now),
            'trial_expires_at': _isoformat_utc(trial_expires),
            'trial_days': _AUTH_TRIAL_DAYS,
        },
    ) if _sb_auth_admin_create_user else {}
    created_user = (created or {}).get('user') or {}
    if not isinstance(created_user, dict) or not created_user.get('id'):
        flash('Unable to create user.', 'error')
        return _user_management_redirect()
    auth_like = _auth_record_from_supabase_user(created_user)
    updates = {
        'username': username,
        'display_name': username,
        'is_admin': is_admin,
        'trial_started_at': _isoformat_utc(now),
        'trial_expires_at': _isoformat_utc(trial_expires),
    }
    if access == 'free':
        updates.update(_subscription_update_for_plan('free', current_auth_user=auth_like))
    elif access in {'monthly', 'yearly'}:
        updates.update(_subscription_update_for_plan(access, current_auth_user=auth_like))
    else:
        updates.update({
            'subscription_status': 'trialing',
            'subscription_plan': 'trial',
            'billing_interval': None,
            'subscription_started_at': None,
            'subscription_ends_at': None,
        })
    saved_row = None
    if _sb_upsert_user_account:
        try:
            saved_row = _sb_upsert_user_account(_build_account_payload(auth_like, updates))
        except Exception:
            current_app.logger.exception('Failed to persist user_accounts row after admin create for user_id=%s.', created_user.get('id'))
            saved_row = None
    if _sb_upsert_user_account and not saved_row:
        flash('User was created in Supabase Auth, but user_accounts sync failed. Verify SUPABASE_URL/project and user_accounts write policy, then run Sync users now.', 'error')
        return _user_management_redirect()
    flash(f"User created for {email} with {'admin' if is_admin else 'member'} access.", 'success')
    return _user_management_redirect()


@main_bp.route('/admin/users/<user_id>/free', methods=['POST'])
def user_management_free_page(user_id: str):
    guard = _require_admin_page()
    if guard is not None:
        return guard
    csrf_guard = _require_csrf_form()
    if csrf_guard is not None:
        return csrf_guard
    auth_row = _sb_auth_admin_get_user(user_id) if _sb_auth_admin_get_user else None
    if not auth_row:
        flash('User not found.', 'error')
        return _user_management_redirect()
    if str(request.form.get('confirm_free') or '').strip() != '1':
        flash('Confirm the free-access change to continue.', 'error')
        return _user_management_redirect()
    auth_like = _auth_record_from_supabase_user(auth_row, _sb_get_user_account(user_id) if _sb_get_user_account else None)
    saved_row = None
    if _sb_upsert_user_account:
        try:
            saved_row = _sb_upsert_user_account(_build_account_payload(auth_like, _subscription_update_for_plan('free', current_auth_user=auth_like)))
        except Exception:
            current_app.logger.exception('Failed to persist free-access update for user_id=%s.', user_id)
            saved_row = None
    if _sb_upsert_user_account and not saved_row:
        flash('Could not persist free-access update in user_accounts. Verify SUPABASE_URL/project and user_accounts write policy, then run Sync users now.', 'error')
        return _user_management_redirect()
    flash(f"{auth_like.get('email') or 'User'} now has free access.", 'success')
    return _user_management_redirect()


@main_bp.route('/admin/users/<user_id>/cancel-free', methods=['POST'])
def user_management_cancel_free_page(user_id: str):
    guard = _require_admin_page()
    if guard is not None:
        return guard
    csrf_guard = _require_csrf_form()
    if csrf_guard is not None:
        return csrf_guard
    auth_row = _sb_auth_admin_get_user(user_id) if _sb_auth_admin_get_user else None
    if not auth_row:
        flash('User not found.', 'error')
        return _user_management_redirect()
    if str(request.form.get('confirm_cancel_free') or '').strip() != '1':
        flash('Confirm the cancel-free action to continue.', 'error')
        return _user_management_redirect()
    auth_like = _auth_record_from_supabase_user(auth_row, _sb_get_user_account(user_id) if _sb_get_user_account else None)
    if str(auth_like.get('subscription_plan') or '').strip() != 'free':
        flash('This user is not currently on free access.', 'info')
        return _user_management_redirect()
    updates = _subscription_update_for_plan('unsubscribe', current_auth_user=auth_like)
    updates['trial_expires_at'] = _isoformat_utc(datetime.now(timezone.utc))
    saved_row = None
    if _sb_upsert_user_account:
        try:
            saved_row = _sb_upsert_user_account(_build_account_payload(auth_like, updates))
        except Exception:
            current_app.logger.exception('Failed to persist cancel-free update for user_id=%s.', user_id)
            saved_row = None
    if _sb_upsert_user_account and not saved_row:
        flash('Could not persist cancel-free update in user_accounts. Verify SUPABASE_URL/project and user_accounts write policy, then run Sync users now.', 'error')
        return _user_management_redirect()
    flash(f"{auth_like.get('email') or 'User'} free access has been canceled.", 'success')
    return _user_management_redirect()


@main_bp.route('/admin/users/<user_id>/password', methods=['POST'])
def user_management_password_page(user_id: str):
    guard = _require_admin_page()
    if guard is not None:
        return guard
    csrf_guard = _require_csrf_form()
    if csrf_guard is not None:
        return csrf_guard
    password = str(request.form.get('password') or '')
    confirm_password = str(request.form.get('confirm_password') or '')
    if len(password) < 8:
        flash('Password must be at least 8 characters.', 'error')
        return _user_management_redirect()
    if password != confirm_password:
        flash('Passwords do not match.', 'error')
        return _user_management_redirect()
    if _sb_auth_admin_update_user:
        _sb_auth_admin_update_user(user_id, {'password': password})
    flash('Password reset completed.', 'success')
    return _user_management_redirect()


@main_bp.route('/admin/users/<user_id>/delete', methods=['POST'])
def user_management_delete_page(user_id: str):
    guard = _require_admin_page()
    if guard is not None:
        return guard
    csrf_guard = _require_csrf_form()
    if csrf_guard is not None:
        return csrf_guard
    current_auth_user = _refresh_current_auth_user() or _current_auth_user()
    if str(current_auth_user.get('user_id') or '') == str(user_id):
        flash('Delete your own account from Account instead.', 'error')
        return _user_management_redirect()
    if str(request.form.get('confirm_delete') or '').strip() != '1':
        flash('Confirm the delete action to continue.', 'error')
        return _user_management_redirect()
    if _sb_auth_admin_delete_user:
        _sb_auth_admin_delete_user(user_id)
    elif _sb_delete_user_account:
        _sb_delete_user_account(user_id)
    flash('User deleted.', 'success')
    return _user_management_redirect()


@main_bp.route('/logout', methods=['POST'])
def logout_page():
    _clear_auth_session()
    flash('Logged out.', 'success')
    return redirect(url_for('main.login_page'))


@main_bp.route('/stripe/webhook', methods=['POST'])
def stripe_webhook_page():
    if not _stripe_portal_enabled() or not _stripe_webhook_secret():
        return jsonify({'error': 'stripe_webhook_not_configured'}), 503
    payload = request.get_data(cache=False, as_text=False)
    signature = request.headers.get('Stripe-Signature', '')
    try:
        stripe_client = _stripe_client()
        event = stripe_client.Webhook.construct_event(payload, signature, _stripe_webhook_secret())
    except Exception:
        current_app.logger.exception('Stripe webhook verification failed.')
        return jsonify({'error': 'invalid_signature'}), 400

    event_type = str(event.get('type') or '').strip()
    data = event.get('data') or {}
    obj = data.get('object') or {}
    if not isinstance(obj, dict):
        obj = dict(obj)

    try:
        if event_type == 'checkout.session.completed':
            _sync_stripe_checkout_session(obj)
        elif event_type in {'customer.subscription.created', 'customer.subscription.updated', 'customer.subscription.deleted'}:
            _sync_stripe_subscription(obj)
    except Exception:
        current_app.logger.exception('Stripe webhook sync failed for %s.', event_type)
        return jsonify({'error': 'sync_failed'}), 500
    return jsonify({'received': True})


@main_bp.route('/projections')
def game_projections_page():
    """Game Projections page showing today's games by Eastern Time, with toggle to yesterday."""
    return render_template('projections.html', teams=TEAM_ROWS, active_tab='Game Projections', show_season_state=False)


@main_bp.route('/donation')
def donation_page():
    return render_template('donation.html', active_tab='Donation', show_filters=False)


@main_bp.route('/skaters')
def skaters_page():
    """Skaters page (player card + bio/metadata)."""
    return render_template(
        'skaters.html',
        teams=TEAM_ROWS,
        active_tab='Skaters',
        show_season_state=False,
        show_include_historic=False,
    )


@main_bp.route('/goalies')
def goalies_page():
    """Goalies page (player card + bio/metadata)."""
    return render_template(
        'goalies.html',
        teams=TEAM_ROWS,
        active_tab='Goalies',
        show_season_state=False,
        show_include_historic=False,
    )


@main_bp.route('/line-tool')
def line_tool_page():
    """Line Tool page using the current lineup feed."""
    return render_template(
        'line_tool.html',
        teams=TEAM_ROWS,
        active_tab='Line Tool',
        show_season_state=False,
        show_include_historic=False,
    )


@main_bp.route('/api/line-tool/players')
def api_line_tool_players():
    """Return all players (skaters + goalies) for a team/season from the players table."""
    team = str(request.args.get('team', '')).strip().upper()
    season = str(request.args.get('season', '')).strip()
    if not team or not season:
        return jsonify({'players': []})

    season_ids = _parse_request_season_ids(season)

    # Step 1: get player IDs who played for this team – game_data is filtered by
    # team + season so this is a small, fast query even for multiple seasons.
    team_pids: set = set()
    for season_id in season_ids:
        gd_rows = _sb_read(
            'game_data',
            columns='player_id',
            filters={'season': f'eq.{int(season_id)}', 'team': f'eq.{team}'},
        )
        if gd_rows:
            team_pids.update(int(r['player_id']) for r in gd_rows if r.get('player_id'))

    if not team_pids:
        return jsonify({'players': []})

    # Step 2: fetch name/position only for the ~20-30 relevant player IDs.
    # This avoids a full-table scan of the players table (which can be 30k+ rows
    # per season) and makes multi-season loads ~100× faster.
    pid_filter = ','.join(str(p) for p in sorted(team_pids))
    player_rows = _sb_read(
        'players',
        columns='player_id,player,position',
        filters={'player_id': f'in.({pid_filter})'},
    )

    pid_info: Dict[str, Dict[str, str]] = {}
    for row in player_rows or []:
        pid = _safe_int(row.get('player_id'))
        if not pid:
            continue
        key = str(pid)
        if key not in pid_info:
            pid_info[key] = {
                'name': str(row.get('player') or '').strip(),
                'position': str(row.get('position') or '').strip().upper(),
            }

    # Fallback: if the players table returned nothing, try the season-based lookup
    # (which may hit the NHL roster API for historical seasons).
    if not pid_info:
        pid_info = _load_player_info_for_seasons(season_ids)

    players_map = {}
    for pid in team_pids:
        info = pid_info.get(str(pid), {})
        players_map[int(pid)] = {
            'id': int(pid),
            'name': info.get('name', ''),
            'position': info.get('position', ''),
        }

    out = sorted(players_map.values(), key=lambda p: (
        0 if p['position'] in ('C', 'L', 'R') else 1 if p['position'] == 'D' else 2,
        p['name']
    ))
    j = jsonify({'players': out})
    j.headers['Cache-Control'] = 'no-store'
    return j


# Game-type codes embedded in NHL game_id: YYYYTTNNNN
_SS_GAME_TYPES = {'regular': {'02'}, 'playoffs': {'03'}}

def _filter_shifts_season_state(shift_rows, ss):
    """Filter shift rows by season state using game_id digit 5-6 (02=reg, 03=playoff)."""
    if not ss or ss == 'all':
        allowed = {'02', '03'}
    else:
        allowed = _SS_GAME_TYPES.get(ss)
        if not allowed:
            return shift_rows
    return [s for s in shift_rows if str(s.get('game_id', ''))[4:6] in allowed]


def _get_lt_shifts(team: str, season: str) -> list:
    """Fetch team+season shifts with caching to avoid redundant pagination."""
    import time as _time
    key = f"{team}|{season}"
    cached = _LT_SHIFTS_CACHE.get(key)
    if cached:
        ts, rows = cached
        if _time.time() - ts < _LT_SHIFTS_TTL:
            return rows
    rows = _sb_read('shifts',
                    columns='shift_index,game_id,player_id,duration,strength_state',
                    filters={'team': f'eq.{team}', 'season': f'eq.{season}'},
                    order='shift_index')
    if rows:
        _LT_SHIFTS_CACHE[key] = (_time.time(), rows)
        # Prune old entries (keep max 40 to support league-wide fetches)
        if len(_LT_SHIFTS_CACHE) > 40:
            oldest_key = min(_LT_SHIFTS_CACHE, key=lambda k: _LT_SHIFTS_CACHE[k][0])
            _LT_SHIFTS_CACHE.pop(oldest_key, None)
    return rows or []


def _get_lt_pbp(season: str, game_ids: list, xg_col: str, extra_cols: str = '') -> list:
    """Fetch PBP corsi events for a set of game IDs with caching."""
    import time as _time
    gid_key = hash(tuple(sorted(game_ids)))
    key = f"{season}|{gid_key}|{xg_col}|{extra_cols}"
    cached = _LT_PBP_CACHE.get(key)
    if cached:
        ts, rows = cached
        if _time.time() - ts < _LT_PBP_TTL:
            return rows
    base_cols = f"game_id,shift_index,event_team,opponent,shot,goal,fenwick,corsi,{xg_col},strength_state,period,season_state"
    if extra_cols:
        base_cols += f",{extra_cols}"
    all_pbp = []
    BATCH = 20
    for i in range(0, len(game_ids), BATCH):
        batch_ids = game_ids[i:i + BATCH]
        gid_filter = ','.join(str(g) for g in batch_ids)
        rows = _sb_read('pbp', columns=base_cols, filters={
            'season': f'eq.{season}',
            'game_id': f'in.({gid_filter})',
            'corsi': 'eq.1',
        })
        if rows:
            all_pbp.extend(rows)
    if all_pbp:
        _LT_PBP_CACHE[key] = (_time.time(), all_pbp)
        if len(_LT_PBP_CACHE) > 10:
            oldest_key = min(_LT_PBP_CACHE, key=lambda k: _LT_PBP_CACHE[k][0])
            _LT_PBP_CACHE.pop(oldest_key, None)
    return all_pbp


def _get_lt_shifts_parallel(team: str, season_ids: Sequence[int]) -> List[Dict[str, Any]]:
    """Fetch shifts for multiple seasons in parallel to avoid sequential I/O latency."""
    if len(season_ids) <= 1:
        rows: List[Dict[str, Any]] = []
        for sid in season_ids:
            rows.extend(_get_lt_shifts(team, str(sid)))
        return rows
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(season_ids)) as ex:
        results = list(ex.map(lambda sid: _get_lt_shifts(team, str(sid)), season_ids))
    return [r for rows in results for r in rows]


def _get_lt_pbp_parallel(
    season_ids: Sequence[int],
    game_list: List[int],
    xg_col: str,
    extra_cols: str = '',
) -> List[Dict[str, Any]]:
    """Fetch PBP for multiple seasons in parallel."""
    def _fetch(sid: int) -> List[Dict[str, Any]]:
        gids = [g for g in game_list if len(str(g)) >= 8 and str(g)[:4] == str(sid)[:4]]
        if not gids:
            return []
        return _get_lt_pbp(str(sid), gids, xg_col, extra_cols)

    if len(season_ids) <= 1:
        return [r for sid in season_ids for r in _fetch(sid)]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(season_ids)) as ex:
        results = list(ex.map(_fetch, season_ids))
    return [r for rows in results for r in rows]


def _compute_line_tool_team_combos(team: str, season: str, ss: str, strength: str,
                                   xg_col: str, line_type: str,
                                   pid_info: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    if line_type == 'def':
        target_positions = {'D'}
        combo_size = 2
    else:
        target_positions = {'C', 'L', 'R'}
        combo_size = 3

    strength_sets = {
        '5v5': {'5v5'},
        'PP': {'5v4', '5v3', '4v3'},
        'SH': {'4v5', '3v5', '3v4'},
    }
    all_special = {'5v5', '5v4', '5v3', '4v3', '4v5', '3v5', '3v4'}

    combo_groups: Dict[tuple, Dict[str, Any]] = {}
    t_rows = _get_lt_shifts(team, season)
    if not t_rows:
        return []
    t_rows = _filter_shifts_season_state(t_rows, ss)
    if strength and strength.lower() != 'all':
        if strength in strength_sets:
            allowed = strength_sets[strength]
            t_rows = [s for s in t_rows if str(s.get('strength_state', '')) in allowed]
        elif strength == 'Other':
            t_rows = [s for s in t_rows if str(s.get('strength_state', '')) not in all_special]

    for s in t_rows:
        pids_on_ice = str(s.get('player_id', '')).split()
        line_pids = sorted(
            pid for pid in pids_on_ice
            if pid_info.get(pid, {}).get('position', '') in target_positions
        )
        if len(line_pids) != combo_size:
            continue
        key = (team, tuple(line_pids))
        grp = combo_groups.get(key)
        if grp is None:
            grp = {'duration': 0, 'game_ids': set(), 'shift_keys': set(), 'team': team}
            combo_groups[key] = grp
        gid = int(s.get('game_id', 0))
        si = int(s.get('shift_index', 0))
        dur = int(s.get('duration', 0) or 0)
        grp['duration'] += dur
        grp['game_ids'].add(gid)
        grp['shift_keys'].add((gid, si))

    if not combo_groups:
        return []

    combo_onice = {}
    for key in combo_groups:
        combo_onice[key] = {'cf': 0, 'ca': 0, 'ff': 0, 'fa': 0,
                            'sf': 0, 'sa': 0, 'gf': 0, 'ga': 0,
                            'xgf': 0.0, 'xga': 0.0}

    all_game_ids = set()
    for grp in combo_groups.values():
        all_game_ids.update(grp['game_ids'])
    all_pbp = _get_lt_pbp(season, sorted(all_game_ids), xg_col)

    from collections import defaultdict
    sk_to_combos = defaultdict(list)
    for key, grp in combo_groups.items():
        for sk in grp['shift_keys']:
            sk_to_combos[sk].append(key)

    for e in all_pbp:
        if int(e.get('period') or 0) == 5:
            continue
        si_raw = e.get('shift_index')
        if si_raw is None:
            continue
        gid = int(e.get('game_id', 0))
        si = int(si_raw)
        ckeys = sk_to_combos.get((gid, si))
        if not ckeys:
            continue
        if ss and ss != 'all' and str(e.get('season_state', '')).lower() != ss:
            continue

        et = str(e.get('event_team', '')).upper()
        opp = str(e.get('opponent', '')).upper()
        is_shot = int(e.get('shot') or 0) == 1
        is_goal = int(e.get('goal') or 0) == 1
        is_fenwick = int(e.get('fenwick') or 0) == 1
        xg_val = float(e.get(xg_col) or 0.0)

        for ckey in ckeys:
            cteam = combo_groups[ckey]['team']
            oi = combo_onice[ckey]
            if et == cteam:
                oi['cf'] += 1
                if is_fenwick:
                    oi['ff'] += 1
                if is_shot:
                    oi['sf'] += 1
                if is_goal:
                    oi['gf'] += 1
                oi['xgf'] += xg_val
            elif opp == cteam:
                oi['ca'] += 1
                if is_fenwick:
                    oi['fa'] += 1
                if is_shot:
                    oi['sa'] += 1
                if is_goal:
                    oi['ga'] += 1
                oi['xga'] += xg_val

    combos: List[Dict[str, Any]] = []
    for key, grp in combo_groups.items():
        toi_min = grp['duration'] / 60.0
        if toi_min < 0.1:
            continue
        team_abbr, pids = key
        oi = combo_onice[key]
        cf = oi['cf']; ca = oi['ca']
        ff = oi['ff']; fa = oi['fa']
        sf = oi['sf']; sa = oi['sa']
        gf = oi['gf']; ga = oi['ga']
        xgf_v = round(oi['xgf'], 2); xga_v = round(oi['xga'], 2)
        combos.append({
            'players': list(pids),
            'team': team_abbr,
            'gp': len(grp['game_ids']),
            'toi': round(toi_min, 1),
            'cf': cf, 'ca': ca, 'cfPct': round(100 * cf / max(cf + ca, 1), 1),
            'ff': ff, 'fa': fa, 'ffPct': round(100 * ff / max(ff + fa, 1), 1),
            'sf': sf, 'sa': sa, 'sfPct': round(100 * sf / max(sf + sa, 1), 1),
            'gf': gf, 'ga': ga, 'gfPct': round(100 * gf / max(gf + ga, 1), 1),
            'xgf': xgf_v, 'xga': xga_v,
            'xgfPct': round(100 * xgf_v / max(xgf_v + xga_v, 0.001), 1),
            'shPct': round(100 * gf / max(sf, 1), 1),
            'svPct': round(100 * (1 - ga / max(sa, 1)), 1),
            'pdo': round(100 * gf / max(sf, 1) + 100 * (1 - ga / max(sa, 1)), 1),
        })
    return combos


@main_bp.route('/api/line-tool/data')
def api_line_tool_data():
    """Return line tool data: KPIs + zone heat map data for selected players on-ice together.

    Query params:
      team – team abbreviation (required)
      season – e.g. 20242025 (required)
      players – comma-separated player IDs (1-5 players, required)
      seasonState – regular (default) / playoffs / all
      strengthState – 5v5 (default) / PP / SH / Other / all
      xgModel – xG_F (default) / xG_S / xG_F2
    """
    team = str(request.args.get('team', '')).strip().upper()
    season = str(request.args.get('season', '')).strip()
    players_raw = str(request.args.get('players', '')).strip()
    if not team or not season:
        return jsonify({'error': 'team and season required'}), 400
    season_ids = _parse_request_season_ids(season)

    try:
        player_ids = [str(int(x)) for x in players_raw.split(',') if x.strip()] if players_raw else []
    except Exception:
        return jsonify({'error': 'invalid player IDs'}), 400
    if len(player_ids) > 5:
        return jsonify({'error': 'maximum 5 players'}), 400

    ss = str(request.args.get('seasonState', 'regular')).lower()
    strength = str(request.args.get('strengthState', '5v5')).strip()
    xg_model = str(request.args.get('xgModel', 'xG_F')).strip()
    xg_col_map = {'xG_F': 'xg_f', 'xG_S': 'xg_s', 'xG_F2': 'xg_f2'}
    xg_col = xg_col_map.get(xg_model, 'xg_f')

    # Optional versus filter
    vs_team_raw = str(request.args.get('vs_team', '')).strip().upper()
    vs_team = vs_team_raw if vs_team_raw else None
    vs_players_raw = str(request.args.get('vs_players', '')).strip()
    try:
        vs_player_ids = [str(int(x)) for x in vs_players_raw.split(',') if x.strip()] if vs_players_raw else []
    except Exception:
        vs_player_ids = []
    if len(vs_player_ids) > 3:
        vs_player_ids = vs_player_ids[:3]

    try:
        lt_data_ttl_s = max(30, int(os.getenv('LINE_TOOL_DATA_CACHE_TTL_SECONDS', '300') or '300'))
    except Exception:
        lt_data_ttl_s = 300
    try:
        lt_data_max_items = max(8, int(os.getenv('LINE_TOOL_DATA_CACHE_MAX_ITEMS', '96') or '96'))
    except Exception:
        lt_data_max_items = 96
    lt_data_cache_key = (
        str(team),
        tuple(_normalize_season_id_list(season_ids)),
        tuple(sorted(player_ids, key=lambda x: int(x))) if player_ids else (),
        str(ss),
        str(strength),
        str(xg_model),
        str(vs_team or ''),
        tuple(sorted(vs_player_ids, key=lambda x: int(x))) if vs_player_ids else (),
    )
    _cache_prune_ttl_and_size(_LT_DATA_CACHE, ttl_s=lt_data_ttl_s, max_items=lt_data_max_items)
    lt_data_cached = _cache_get(_LT_DATA_CACHE, lt_data_cache_key, lt_data_ttl_s)
    if isinstance(lt_data_cached, dict):
        j_cached = jsonify(lt_data_cached)
        j_cached.headers['Cache-Control'] = 'no-store'
        return j_cached

    # ── 1. Fetch shifts for the team + season (parallel) ─────
    shift_rows = _get_lt_shifts_parallel(team, season_ids)
    if not shift_rows:
        return jsonify(_empty_line_tool_response())
    shift_rows = _filter_shifts_season_state(shift_rows, ss)
    shift_rows = _apply_lt_strength_filter(shift_rows, strength)

    # ── 2. Find shifts where ALL selected players are on ice ─
    # When no players are selected, use all team shifts (full team view).
    if player_ids:
        common_shifts = []
        player_id_set = set(player_ids)
        for s in shift_rows:
            pids_on_ice = set(str(s.get('player_id', '')).split())
            if player_id_set.issubset(pids_on_ice):
                common_shifts.append(s)
    else:
        common_shifts = shift_rows

    if not common_shifts:
        return jsonify(_empty_line_tool_response())

    # ── 2b. Intersect with opponent shifts if vs_team / vs_players set ──
    if vs_team:
        # Fetch opponent shifts for all seasons
        opp_shift_rows: list = []
        for _sid in season_ids:
            opp_shift_rows.extend(_get_lt_shifts(vs_team, str(_sid)))
        opp_shift_rows = _filter_shifts_season_state(opp_shift_rows, ss)
        opp_shift_rows = _apply_lt_strength_filter(opp_shift_rows, strength)
        # Build set of shift keys where all requested vs_players are on ice for the opp
        if vs_player_ids:
            vs_id_set = set(vs_player_ids)
            opp_valid_keys: set = set()
            for s in opp_shift_rows:
                pids = set(str(s.get('player_id', '')).split())
                if vs_id_set.issubset(pids):
                    opp_valid_keys.add((int(s.get('game_id', 0)), int(s.get('shift_index', 0))))
        else:
            # Any opp shift → just use all opp shift keys
            opp_valid_keys = {(int(s.get('game_id', 0)), int(s.get('shift_index', 0))) for s in opp_shift_rows}
        # Restrict common_shifts to those where opponent is also on ice
        common_shifts = [
            s for s in common_shifts
            if (int(s.get('game_id', 0)), int(s.get('shift_index', 0))) in opp_valid_keys
        ]
        if not common_shifts:
            return jsonify(_empty_line_tool_response())

    # ── 4. Compute GP, TOI from shifts ───────────────────────
    game_ids = set()
    shift_keys = set()  # (game_id, shift_index)
    total_duration = 0
    for s in common_shifts:
        gid = int(s.get('game_id', 0))
        si = int(s.get('shift_index', 0))
        dur = int(s.get('duration', 0) or 0)
        game_ids.add(gid)
        shift_keys.add((gid, si))
        total_duration += dur

    gp = len(game_ids)
    toi_min = total_duration / 60.0

    # ── 4b. Team-level shift keys + TOI (for Vs. Team / Vs. League) ──
    # Apply same strength filter to ALL team shifts, limited to same games
    team_str_shifts = list(shift_rows)
    team_shift_keys = set()
    team_toi_sec = 0
    for s in team_str_shifts:
        gid = int(s.get('game_id', 0))
        if gid not in game_ids:
            continue  # only games we have PBP for
        si = int(s.get('shift_index', 0))
        team_shift_keys.add((gid, si))
        team_toi_sec += int(s.get('duration', 0) or 0)
    team_toi_min = team_toi_sec / 60.0

    # ── 5. Fetch PBP for the relevant games (parallel) ───────
    game_list = sorted(game_ids)
    all_pbp = _get_lt_pbp_parallel(season_ids, game_list, xg_col, extra_cols='x,y,box_id,highlight_url')

    # ── 6. Filter PBP to matching shift_indexes ──────────────
    events_for = []   # team shooting (event_team = team)
    events_against = []  # opponents shooting at team (opponent = team)
    team_oz_detail = {}   # team-level OZ zone detail (all team shifts)
    team_dz_detail = {}   # team-level DZ zone detail (all team shifts)
    league_zone_detail = {}  # both-teams zone detail (league approx)
    for e in all_pbp:
        if int(e.get('period') or 0) == 5:
            continue  # exclude shootout
        gid = int(e.get('game_id', 0))
        si_raw = e.get('shift_index')
        if si_raw is None:
            continue
        si = int(si_raw)
        # Season state filter
        if ss and ss != 'all':
            if str(e.get('season_state', '')).lower() != ss:
                continue
        et = str(e.get('event_team', '')).upper()
        opp = str(e.get('opponent', '')).upper()
        key = (gid, si)

        # Player-level events
        if key in shift_keys:
            if et == team:
                events_for.append(e)
            elif opp == team:
                events_against.append(e)

        # Team-level and league-level zone data
        if key in team_shift_keys:
            bid = str(e.get('box_id') or '')
            if bid.startswith('O'):
                is_fen = int(e.get('fenwick') or 0) == 1
                is_sh = int(e.get('shot') or 0) == 1
                is_gl = int(e.get('goal') or 0) == 1
                xgv = float(e.get(xg_col) or 0.0)
                # League: all events by O-zone (both teams)
                ld = league_zone_detail.setdefault(bid, {'count': 0, 'fenwick': 0, 'shots': 0, 'goals': 0, 'xg': 0.0})
                ld['count'] += 1
                if is_fen: ld['fenwick'] += 1
                if is_sh: ld['shots'] += 1
                if is_gl: ld['goals'] += 1
                ld['xg'] += xgv
                # Team offense
                if et == team:
                    td = team_oz_detail.setdefault(bid, {'count': 0, 'fenwick': 0, 'shots': 0, 'goals': 0, 'xg': 0.0})
                    td['count'] += 1
                    if is_fen: td['fenwick'] += 1
                    if is_sh: td['shots'] += 1
                    if is_gl: td['goals'] += 1
                    td['xg'] += xgv
                # Team defense (O→D mapping)
                elif opp == team:
                    dz_bid = 'D' + bid[1:]
                    td = team_dz_detail.setdefault(dz_bid, {'count': 0, 'fenwick': 0, 'shots': 0, 'goals': 0, 'xg': 0.0})
                    td['count'] += 1
                    if is_fen: td['fenwick'] += 1
                    if is_sh: td['shots'] += 1
                    if is_gl: td['goals'] += 1
                    td['xg'] += xgv

    # ── 7. Compute KPIs ──────────────────────────────────────
    def _sum_kpis(evts):
        corsi = 0; fenwick = 0; shots = 0; xg_total = 0.0; goals = 0
        for ev in evts:
            corsi += 1
            if int(ev.get('fenwick') or 0) == 1:
                fenwick += 1
            if int(ev.get('shot') or 0) == 1:
                shots += 1
            if int(ev.get('goal') or 0) == 1:
                goals += 1
            xg_total += float(ev.get(xg_col) or 0.0)
        return {'corsi': corsi, 'fenwick': fenwick, 'shots': shots, 'xG': round(xg_total, 2), 'goals': goals}

    kpi_for = _sum_kpis(events_for)
    kpi_against = _sum_kpis(events_against)

    cf = kpi_for['corsi']; ca = kpi_against['corsi']
    ff = kpi_for['fenwick']; fa = kpi_against['fenwick']
    sf = kpi_for['shots']; sa = kpi_against['shots']
    gf = kpi_for['goals']; ga = kpi_against['goals']
    xgf = kpi_for['xG']; xga = kpi_against['xG']

    cf_pct = round(100 * cf / max(cf + ca, 1), 1)
    ff_pct = round(100 * ff / max(ff + fa, 1), 1)
    sf_pct = round(100 * sf / max(sf + sa, 1), 1)
    gf_pct = round(100 * gf / max(gf + ga, 1), 1)
    xgf_pct = round(100 * xgf / max(xgf + xga, 0.001), 1)
    sh_pct = round(100 * gf / max(sf, 1), 1)
    sv_pct = round(100 * (1 - ga / max(sa, 1)), 1)
    pdo = round(sh_pct + sv_pct, 1)

    # ── 8. Zone heat map data ────────────────────────────────
    # Offensive zones: shots FOR (team shooting) – only O-prefixed box_ids
    oz_counts = {}
    oz_detail = {}   # per-zone KPI detail for client-side zone filtering
    for ev in events_for:
        bid = str(ev.get('box_id') or '')
        if not bid.startswith('O'):
            continue
        oz_counts[bid] = oz_counts.get(bid, 0) + 1
        d = oz_detail.setdefault(bid, {'count': 0, 'fenwick': 0, 'shots': 0, 'goals': 0, 'xg': 0.0})
        d['count'] += 1
        if int(ev.get('fenwick') or 0) == 1:
            d['fenwick'] += 1
        if int(ev.get('shot') or 0) == 1:
            d['shots'] += 1
        if int(ev.get('goal') or 0) == 1:
            d['goals'] += 1
        d['xg'] += float(ev.get(xg_col) or 0.0)

    # Defensive zones: shots AGAINST (opp shooting) – convert O→D
    dz_counts = {}
    dz_detail = {}
    for ev in events_against:
        bid = str(ev.get('box_id') or '')
        if not bid.startswith('O'):
            continue
        dz_bid = 'D' + bid[1:]   # O01 → D01
        dz_counts[dz_bid] = dz_counts.get(dz_bid, 0) + 1
        d = dz_detail.setdefault(dz_bid, {'count': 0, 'fenwick': 0, 'shots': 0, 'goals': 0, 'xg': 0.0})
        d['count'] += 1
        if int(ev.get('fenwick') or 0) == 1:
            d['fenwick'] += 1
        if int(ev.get('shot') or 0) == 1:
            d['shots'] += 1
        if int(ev.get('goal') or 0) == 1:
            d['goals'] += 1
        d['xg'] += float(ev.get(xg_col) or 0.0)

    # Round xg values
    for d in oz_detail.values():
        d['xg'] = round(d['xg'], 2)
    for d in dz_detail.values():
        d['xg'] = round(d['xg'], 2)
    for d in team_oz_detail.values():
        d['xg'] = round(d['xg'], 2)
    for d in team_dz_detail.values():
        d['xg'] = round(d['xg'], 2)
    for d in league_zone_detail.values():
        d['xg'] = round(d['xg'], 2)

    # ── 9. Collect goal events with highlight URLs ─────────────
    goal_highlights = []
    for ev in events_for:
        if int(ev.get('goal') or 0) == 1:
            hl = ev.get('highlight_url') or ''
            goal_highlights.append({
                'highlightUrl': hl if hl else None,
                'x': float(ev.get('x') or 0),
                'y': float(ev.get('y') or 0),
                'xG': round(float(ev.get(xg_col) or 0), 4),
                'boxId': str(ev.get('box_id') or ''),
                'direction': 'for',
            })
    for ev in events_against:
        if int(ev.get('goal') or 0) == 1:
            hl = ev.get('highlight_url') or ''
            goal_highlights.append({
                'highlightUrl': hl if hl else None,
                'x': float(ev.get('x') or 0),
                'y': float(ev.get('y') or 0),
                'xG': round(float(ev.get(xg_col) or 0), 4),
                'boxId': str(ev.get('box_id') or ''),
                'direction': 'against',
            })

    result = {
        'gp': gp,
        'toi': round(toi_min, 1),
        'cf': cf, 'ca': ca, 'cfPct': cf_pct,
        'ff': ff, 'fa': fa, 'ffPct': ff_pct,
        'sf': sf, 'sa': sa, 'sfPct': sf_pct,
        'gf': gf, 'ga': ga, 'gfPct': gf_pct,
        'xgf': xgf, 'xga': xga, 'xgfPct': xgf_pct,
        'shPct': sh_pct, 'svPct': sv_pct, 'pdo': pdo,
        'ozZones': oz_counts,
        'dzZones': dz_counts,
        'ozDetail': oz_detail,
        'dzDetail': dz_detail,
        'teamToi': round(team_toi_min, 1),
        'teamOzDetail': team_oz_detail,
        'teamDzDetail': team_dz_detail,
        'leagueToi': round(team_toi_min * 2, 1),
        'leagueZoneDetail': league_zone_detail,
        'goalHighlights': goal_highlights,
    }
    _cache_set_multi_bounded(_LT_DATA_CACHE, lt_data_cache_key, result, ttl_s=lt_data_ttl_s, max_items=lt_data_max_items)
    j = jsonify(result)
    j.headers['Cache-Control'] = 'no-store'
    return j


def _empty_line_tool_response():
    return {
        'gp': 0, 'toi': 0,
        'cf': 0, 'ca': 0, 'cfPct': 0,
        'ff': 0, 'fa': 0, 'ffPct': 0,
        'sf': 0, 'sa': 0, 'sfPct': 0,
        'gf': 0, 'ga': 0, 'gfPct': 0,
        'xgf': 0, 'xga': 0, 'xgfPct': 0,
        'shPct': 0, 'svPct': 0, 'pdo': 0,
        'ozZones': {}, 'dzZones': {},
        'ozDetail': {}, 'dzDetail': {},
    }


@main_bp.route('/api/line-tool/wowy')
def api_line_tool_wowy():
    """Return WOWY (With Or Without You) data for the selected players.

    For each combination of selected players being on/off ice, computes:
      - On-ice stats (TOI, CF, CA, etc.)
      - Individual stats per player (goals, assists, ixG, etc.)

    Query params: same as /api/line-tool/data
    """
    team = str(request.args.get('team', '')).strip().upper()
    season = str(request.args.get('season', '')).strip()
    players_raw = str(request.args.get('players', '')).strip()
    if not team or not season or not players_raw:
        return jsonify({'combos': [], 'players': []}), 400
    season_ids = _parse_request_season_ids(season)

    try:
        player_ids = [str(int(x)) for x in players_raw.split(',') if x.strip()]
    except Exception:
        return jsonify({'combos': [], 'players': []}), 400
    if not player_ids or len(player_ids) > 5:
        return jsonify({'combos': [], 'players': []}), 400

    ss = str(request.args.get('seasonState', 'regular')).lower()
    strength = str(request.args.get('strengthState', '5v5')).strip()
    xg_model = str(request.args.get('xgModel', 'xG_F')).strip()
    xg_col_map = {'xG_F': 'xg_f', 'xG_S': 'xg_s', 'xG_F2': 'xg_f2'}
    xg_col = xg_col_map.get(xg_model, 'xg_f')

    # ── 1. Fetch shifts (parallel across seasons) ─────────────
    shift_rows = _get_lt_shifts_parallel(team, season_ids)
    if not shift_rows:
        return jsonify({'combos': [], 'players': []})
    shift_rows = _filter_shifts_season_state(shift_rows, ss)

    # ── 2. Apply strength state filter ────────────────────────
    if strength and strength.lower() != 'all':
        strength_sets = {
            '5v5': {'5v5'},
            'PP': {'5v4', '5v3', '4v3'},
            'SH': {'4v5', '3v5', '3v4'},
        }
        if strength in strength_sets:
            allowed = strength_sets[strength]
            shift_rows = [s for s in shift_rows if str(s.get('strength_state', '')) in allowed]
        elif strength == 'Other':
            all_special = {'5v5', '5v4', '5v3', '4v3', '4v5', '3v5', '3v4'}
            shift_rows = [s for s in shift_rows if str(s.get('strength_state', '')) not in all_special]

    # ── 3. Group shifts by player mask ────────────────────────
    # mask = tuple of booleans, one per player_id in player_ids order
    mask_groups = {}  # mask -> {'duration': int, 'game_ids': set, 'shift_keys': set}
    for s in shift_rows:
        pids_on_ice = set(str(s.get('player_id', '')).split())
        mask = tuple(pid in pids_on_ice for pid in player_ids)
        # At least one selected player must be on ice OR none (for "without all")
        grp = mask_groups.get(mask)
        if grp is None:
            grp = {'duration': 0, 'game_ids': set(), 'shift_keys': set()}
            mask_groups[mask] = grp
        gid = int(s.get('game_id', 0))
        si = int(s.get('shift_index', 0))
        dur = int(s.get('duration', 0) or 0)
        grp['duration'] += dur
        grp['game_ids'].add(gid)
        grp['shift_keys'].add((gid, si))

    # ── 4. Fetch PBP for all relevant games ───────────────────
    all_game_ids = set()
    for grp in mask_groups.values():
        all_game_ids.update(grp['game_ids'])
    game_list = sorted(all_game_ids)

    all_pbp = _get_lt_pbp_parallel(season_ids, game_list, xg_col,
                                  extra_cols='player1_id,player2_id,player3_id,goalie_id')

    # Build shift_key → mask lookup
    sk_to_mask = {}
    for mask, grp in mask_groups.items():
        for sk in grp['shift_keys']:
            sk_to_mask[sk] = mask

    # ── 5. Assign PBP events to masks ─────────────────────────
    # For each mask, accumulate on-ice stats + individual player events
    mask_onice = {}   # mask -> {cf, ca, ff, fa, sf, sa, gf, ga, xgf, xga}
    mask_indiv = {}   # mask -> {player_id -> {goals, a1, a2, shots, ixg, sa, ga, xga}}

    for mask in mask_groups:
        mask_onice[mask] = {'cf': 0, 'ca': 0, 'ff': 0, 'fa': 0, 'sf': 0, 'sa': 0, 'gf': 0, 'ga': 0, 'xgf': 0.0, 'xga': 0.0}
        mask_indiv[mask] = {pid: {'goals': 0, 'a1': 0, 'a2': 0, 'shots': 0, 'ixg': 0.0,
                                   'fa': 0, 'ga': 0, 'sa': 0, 'xga': 0.0}
                            for pid in player_ids}

    for e in all_pbp:
        if int(e.get('period') or 0) == 5:
            continue
        si_raw = e.get('shift_index')
        if si_raw is None:
            continue
        gid = int(e.get('game_id', 0))
        si = int(si_raw)
        mask = sk_to_mask.get((gid, si))
        if mask is None:
            continue
        # Season state filter
        if ss and ss != 'all':
            if str(e.get('season_state', '')).lower() != ss:
                continue

        et = str(e.get('event_team', '')).upper()
        opp = str(e.get('opponent', '')).upper()
        is_for = (et == team)
        is_against = (opp == team)
        is_shot = int(e.get('shot') or 0) == 1
        is_goal = int(e.get('goal') or 0) == 1
        xg_val = float(e.get(xg_col) or 0.0)

        is_fenwick = int(e.get('fenwick') or 0) == 1

        oi = mask_onice[mask]
        if is_for:
            oi['cf'] += 1
            if is_fenwick:
                oi['ff'] += 1
            if is_shot:
                oi['sf'] += 1
            if is_goal:
                oi['gf'] += 1
            oi['xgf'] += xg_val
        elif is_against:
            oi['ca'] += 1
            if is_fenwick:
                oi['fa'] += 1
            if is_shot:
                oi['sa'] += 1
            if is_goal:
                oi['ga'] += 1
            oi['xga'] += xg_val

        # Individual stats
        p1 = str(e.get('player1_id') or '')
        p2 = str(e.get('player2_id') or '')
        p3 = str(e.get('player3_id') or '')
        gid_goalie = str(e.get('goalie_id') or '')

        for pid in player_ids:
            iv = mask_indiv[mask][pid]
            if is_for:
                if p1 == pid:
                    if is_shot:
                        iv['shots'] += 1
                    iv['ixg'] += xg_val
                    if is_goal:
                        iv['goals'] += 1
                if is_goal and p2 == pid:
                    iv['a1'] += 1
                if is_goal and p3 == pid:
                    iv['a2'] += 1
            elif is_against:
                if gid_goalie == pid:
                    iv['fa'] += 1
                    iv['sa'] += 1 if is_shot else 0
                    if is_goal:
                        iv['ga'] += 1
                    iv['xga'] += xg_val

    # ── 6. Build response ─────────────────────────────────────
    combos = []
    for mask, grp in mask_groups.items():
        toi_min = grp['duration'] / 60.0
        if toi_min < 0.1:
            continue  # skip trivial groups
        oi = mask_onice[mask]
        cf = oi['cf']; ca = oi['ca']
        ff = oi['ff']; fa = oi['fa']
        sf = oi['sf']; sa = oi['sa']
        gf = oi['gf']; ga = oi['ga']
        xgf_v = round(oi['xgf'], 2); xga_v = round(oi['xga'], 2)

        combo = {
            'mask': list(mask),
            'gp': len(grp['game_ids']),
            'toi': round(toi_min, 1),
            'cf': cf, 'ca': ca, 'cfPct': round(100 * cf / max(cf + ca, 1), 1),
            'ff': ff, 'fa': fa, 'ffPct': round(100 * ff / max(ff + fa, 1), 1),
            'sf': sf, 'sa': sa, 'sfPct': round(100 * sf / max(sf + sa, 1), 1),
            'gf': gf, 'ga': ga, 'gfPct': round(100 * gf / max(gf + ga, 1), 1),
            'xgf': xgf_v, 'xga': xga_v,
            'xgfPct': round(100 * xgf_v / max(xgf_v + xga_v, 0.001), 1),
            'shPct': round(100 * gf / max(sf, 1), 1),
            'svPct': round(100 * (1 - ga / max(sa, 1)), 1),
            'pdo': round(100 * gf / max(sf, 1) + 100 * (1 - ga / max(sa, 1)), 1),
            'individual': {},
        }
        for pid in player_ids:
            iv = mask_indiv[mask][pid]
            combo['individual'][pid] = {
                'goals': iv['goals'],
                'a1': iv['a1'],
                'a2': iv['a2'],
                'points': iv['goals'] + iv['a1'] + iv['a2'],
                'shots': iv['shots'],
                'ixg': round(iv['ixg'], 2),
                'gax': round(iv['goals'] - iv['ixg'], 2),
                'shPct': round(100 * iv['goals'] / max(iv['shots'], 1), 1),
                'fa': iv['fa'],
                'ga': iv['ga'],
                'sa': iv['sa'],
                'xga': round(iv['xga'], 2),
                'gsax': round(iv['xga'] - iv['ga'], 2),
                'svPct': round(100 * (1 - iv['ga'] / max(iv['sa'], 1)), 1) if iv['sa'] > 0 else 0.0,
                'xsvPct': round(100 * (1 - iv['xga'] / max(iv['sa'], 1)), 1) if iv['sa'] > 0 else 0.0,
                'dsvPct': round(
                    (100 * (1 - iv['ga'] / max(iv['sa'], 1))) - (100 * (1 - iv['xga'] / max(iv['sa'], 1))),
                    1) if iv['sa'] > 0 else 0.0,
            }
        combos.append(combo)

    # Sort by TOI descending
    combos.sort(key=lambda c: c['toi'], reverse=True)

    # Player name map – targeted lookup for only the selected player IDs (fast).
    pid_info = _load_player_info_targeted([int(pid) for pid in player_ids])

    player_info = []
    for pid in player_ids:
        info = pid_info.get(pid, {})
        player_info.append({
            'id': pid,
            'name': info.get('name', f'#{pid}'),
            'position': info.get('position', ''),
        })

    j = jsonify({'combos': combos, 'players': player_info})
    j.headers['Cache-Control'] = 'no-store'
    return j


@main_bp.route('/api/line-tool/versus')
def api_line_tool_versus():
    """Return Versus split rows (vs team / not-vs team + optional opponent player masks)."""
    team = str(request.args.get('team', '')).strip().upper()
    season = str(request.args.get('season', '')).strip()
    vs_team = str(request.args.get('vs_team', '')).strip().upper()
    players_raw = str(request.args.get('players', '')).strip()
    vs_players_raw = str(request.args.get('vs_players', '')).strip()
    if not team or not season or not vs_team:
        return jsonify({'rows': [], 'vsTeam': vs_team, 'vsPlayers': []}), 400

    season_ids = _parse_request_season_ids(season)
    try:
        player_ids = [str(int(x)) for x in players_raw.split(',') if x.strip()] if players_raw else []
    except Exception:
        return jsonify({'rows': [], 'vsTeam': vs_team, 'vsPlayers': []}), 400
    if len(player_ids) > 5:
        player_ids = player_ids[:5]

    try:
        vs_player_ids = [str(int(x)) for x in vs_players_raw.split(',') if x.strip()] if vs_players_raw else []
    except Exception:
        return jsonify({'rows': [], 'vsTeam': vs_team, 'vsPlayers': []}), 400
    if len(vs_player_ids) > 3:
        vs_player_ids = vs_player_ids[:3]

    ss = str(request.args.get('seasonState', 'regular')).lower()
    strength = str(request.args.get('strengthState', '5v5')).strip()
    xg_model = str(request.args.get('xgModel', 'xG_F')).strip()
    xg_col_map = {'xG_F': 'xg_f', 'xG_S': 'xg_s', 'xG_F2': 'xg_f2'}
    xg_col = xg_col_map.get(xg_model, 'xg_f')

    try:
        lt_data_ttl_s = max(30, int(os.getenv('LINE_TOOL_DATA_CACHE_TTL_SECONDS', '300') or '300'))
    except Exception:
        lt_data_ttl_s = 300
    try:
        lt_data_max_items = max(8, int(os.getenv('LINE_TOOL_DATA_CACHE_MAX_ITEMS', '96') or '96'))
    except Exception:
        lt_data_max_items = 96
    versus_cache_key = (
        'versus',
        str(team),
        tuple(_normalize_season_id_list(season_ids)),
        tuple(sorted(player_ids, key=lambda x: int(x))) if player_ids else (),
        str(vs_team),
        tuple(sorted(vs_player_ids, key=lambda x: int(x))) if vs_player_ids else (),
        str(ss),
        str(strength),
        str(xg_model),
    )
    _cache_prune_ttl_and_size(_LT_DATA_CACHE, ttl_s=lt_data_ttl_s, max_items=lt_data_max_items)
    versus_cached = _cache_get(_LT_DATA_CACHE, versus_cache_key, lt_data_ttl_s)
    if isinstance(versus_cached, dict):
        j_cached = jsonify(versus_cached)
        j_cached.headers['Cache-Control'] = 'no-store'
        return j_cached

    # 1) Team shifts filtered by selected FOR players (or all shifts if no FOR players selected)
    shift_rows: List[Dict[str, Any]] = _get_lt_shifts_parallel(team, season_ids)
    if not shift_rows:
        return jsonify({'rows': [], 'vsTeam': vs_team, 'vsPlayers': []})
    shift_rows = _filter_shifts_season_state(shift_rows, ss)
    shift_rows = _apply_lt_strength_filter(shift_rows, strength)

    if player_ids:
        player_set = set(player_ids)
        base_shifts = []
        for s in shift_rows:
            pids_on_ice = set(str(s.get('player_id', '')).split())
            if player_set.issubset(pids_on_ice):
                base_shifts.append(s)
    else:
        base_shifts = shift_rows
    if not base_shifts:
        return jsonify({'rows': [], 'vsTeam': vs_team, 'vsPlayers': []})

    # 2) Opponent shifts map: (game_id, shift_index) -> set(opponent player ids)
    opp_shift_rows: List[Dict[str, Any]] = _get_lt_shifts_parallel(vs_team, season_ids)
    opp_shift_rows = _filter_shifts_season_state(opp_shift_rows, ss)
    opp_shift_rows = _apply_lt_strength_filter(opp_shift_rows, strength)

    opp_key_to_pids: Dict[Tuple[int, int], set] = {}
    for s in opp_shift_rows:
        gid = int(s.get('game_id', 0))
        si = int(s.get('shift_index', 0))
        pset = set(str(s.get('player_id', '')).split())
        key = (gid, si)
        cur = opp_key_to_pids.get(key)
        if cur is None:
            opp_key_to_pids[key] = pset
        else:
            cur.update(pset)

    # 3) Partition shifts into groups
    # Group key: ('vs', mask_tuple) for vs-team shifts, ('not_vs', None) for all other shifts.
    group_rows: Dict[Tuple[str, Any], Dict[str, Any]] = {}

    # Always expose both context families in the response, even when zero shifts match.
    group_rows[('not_vs', None)] = {'duration': 0, 'game_ids': set(), 'shift_keys': set()}
    if vs_player_ids:
        combo_count = 1 << len(vs_player_ids)
        for m in range(combo_count):
            mask = tuple(bool((m >> i) & 1) for i in range(len(vs_player_ids)))
            group_rows[('vs', mask)] = {'duration': 0, 'game_ids': set(), 'shift_keys': set()}
    else:
        group_rows[('vs', tuple())] = {'duration': 0, 'game_ids': set(), 'shift_keys': set()}

    for s in base_shifts:
        gid = int(s.get('game_id', 0))
        si = int(s.get('shift_index', 0))
        dur = int(s.get('duration', 0) or 0)
        key = (gid, si)
        opp_pids = opp_key_to_pids.get(key)

        if opp_pids is None:
            gkey = ('not_vs', None)
        else:
            if vs_player_ids:
                mask = tuple(pid in opp_pids for pid in vs_player_ids)
            else:
                mask = tuple()
            gkey = ('vs', mask)

        grp = group_rows[gkey]
        grp['duration'] += dur
        grp['game_ids'].add(gid)
        grp['shift_keys'].add(key)

    # 4) Fetch PBP for relevant games and map shift key -> group
    all_game_ids = set()
    for grp in group_rows.values():
        all_game_ids.update(grp['game_ids'])
    game_list = sorted(all_game_ids)

    all_pbp: List[Dict[str, Any]] = _get_lt_pbp_parallel(season_ids, game_list, xg_col)

    sk_to_group: Dict[Tuple[int, int], Tuple[str, Any]] = {}
    for gkey, grp in group_rows.items():
        for sk in grp['shift_keys']:
            sk_to_group[sk] = gkey

    group_stats: Dict[Tuple[str, Any], Dict[str, Any]] = {}
    for gkey in group_rows:
        group_stats[gkey] = {'cf': 0, 'ca': 0, 'ff': 0, 'fa': 0, 'sf': 0, 'sa': 0, 'gf': 0, 'ga': 0, 'xgf': 0.0, 'xga': 0.0}

    for e in all_pbp:
        if int(e.get('period') or 0) == 5:
            continue
        si_raw = e.get('shift_index')
        if si_raw is None:
            continue
        gid = int(e.get('game_id', 0))
        si = int(si_raw)
        gkey = sk_to_group.get((gid, si))
        if gkey is None:
            continue
        if ss and ss != 'all':
            if str(e.get('season_state', '')).lower() != ss:
                continue

        et = str(e.get('event_team', '')).upper()
        opp = str(e.get('opponent', '')).upper()
        is_for = et == team
        is_against = opp == team
        if not is_for and not is_against:
            continue

        is_fenwick = int(e.get('fenwick') or 0) == 1
        is_shot = int(e.get('shot') or 0) == 1
        is_goal = int(e.get('goal') or 0) == 1
        xg_val = float(e.get(xg_col) or 0.0)

        st = group_stats[gkey]
        if is_for:
            st['cf'] += 1
            if is_fenwick:
                st['ff'] += 1
            if is_shot:
                st['sf'] += 1
            if is_goal:
                st['gf'] += 1
            st['xgf'] += xg_val
        elif is_against:
            st['ca'] += 1
            if is_fenwick:
                st['fa'] += 1
            if is_shot:
                st['sa'] += 1
            if is_goal:
                st['ga'] += 1
            st['xga'] += xg_val

    # Targeted lookup – only the selected opponent player IDs are needed for labels.
    pid_info = _load_player_info_targeted([int(pid) for pid in vs_player_ids])
    vs_player_info = []
    for pid in vs_player_ids:
        info = pid_info.get(pid, {})
        vs_player_info.append({
            'id': pid,
            'name': info.get('name', f'#{pid}'),
            'position': info.get('position', ''),
        })

    def _label_for_row(context: str, mask: Tuple[bool, ...]) -> str:
        if context == 'not_vs':
            return f'w/o {vs_team}'
        if not vs_player_info:
            return f'vs {vs_team}'
        parts = [f'vs {vs_team}']
        for i, p in enumerate(vs_player_info):
            ln = str(p.get('name', '')).split(' ')[-1] or f"#{p.get('id', '')}"
            parts.append((f'vs {ln}') if (i < len(mask) and bool(mask[i])) else (f'w/o {ln}'))
        return ' · '.join(parts)

    rows: List[Dict[str, Any]] = []
    for gkey, grp in group_rows.items():
        context, mask = gkey
        st = group_stats[gkey]
        toi_min = grp['duration'] / 60.0
        cf = st['cf']; ca = st['ca']
        gf = st['gf']; ga = st['ga']
        xgf_v = round(st['xgf'], 2); xga_v = round(st['xga'], 2)
        sh_pct = round(100 * gf / max(st['sf'], 1), 1)
        sv_pct = round(100 * (1 - ga / max(st['sa'], 1)), 1)
        row = {
            'context': 'vs' if context == 'vs' else 'not_vs',
            'label': _label_for_row(context, tuple(mask or ())),
            'mask': list(mask) if isinstance(mask, tuple) else [],
            'gp': len(grp['game_ids']),
            'toi': round(toi_min, 1),
            'cf': cf,
            'ca': ca,
            'cfPct': round(100 * cf / max(cf + ca, 1), 1),
            'gf': gf,
            'ga': ga,
            'gfPct': round(100 * gf / max(gf + ga, 1), 1),
            'xgf': xgf_v,
            'xga': xga_v,
            'xgfPct': round(100 * xgf_v / max(xgf_v + xga_v, 0.001), 1),
            'pdo': round(sh_pct + sv_pct, 1),
        }
        rows.append(row)

    # Deterministic order: vs-team rows first (most specific mask first), then n/vs team row.
    def _row_sort_key(r: Dict[str, Any]):
        context_priority = 0 if r.get('context') == 'vs' else 1
        mask = r.get('mask') or []
        on_count = sum(1 for v in mask if v)
        return (context_priority, -on_count, -float(r.get('toi', 0.0)))

    rows.sort(key=_row_sort_key)

    result = {'rows': rows, 'vsTeam': vs_team, 'vsPlayers': vs_player_info}
    _cache_set_multi_bounded(_LT_DATA_CACHE, versus_cache_key, result, ttl_s=lt_data_ttl_s, max_items=lt_data_max_items)
    j = jsonify(result)
    j.headers['Cache-Control'] = 'no-store'
    return j


@main_bp.route('/api/line-tool/lines')
def api_line_tool_lines():
    """Return forward-line and defense-pairing combinations with on-ice stats.

    Query params:
      team – team abbreviation (required)
      season – e.g. 20242025 (required)
      seasonState – regular / playoffs / all
      strengthState – 5v5 / PP / SH / Other / all
      xgModel – xG_F / xG_S / xG_F2
      type – fwd / def
      scope – team (default) / league
    """
    team = str(request.args.get('team', '')).strip().upper()
    season = str(request.args.get('season', '')).strip()
    if not team or not season:
        return jsonify({'combos': [], 'players': {}}), 400
    season_ids = _parse_request_season_ids(season)

    ss = str(request.args.get('seasonState', 'regular')).lower()
    strength = str(request.args.get('strengthState', '5v5')).strip()
    xg_model = str(request.args.get('xgModel', 'xG_F')).strip()
    xg_col_map = {'xG_F': 'xg_f', 'xG_S': 'xg_s', 'xG_F2': 'xg_f2'}
    xg_col = xg_col_map.get(xg_model, 'xg_f')
    line_type = str(request.args.get('type', 'fwd')).strip().lower()
    scope = str(request.args.get('scope', 'team')).strip().lower()

    # Result cache – first cold load for 3 seasons can take 30–60 s; cache the
    # result so subsequent requests are instant.
    try:
        lt_lines_ttl_s = max(30, int(os.getenv('LINE_TOOL_DATA_CACHE_TTL_SECONDS', '300') or '300'))
    except Exception:
        lt_lines_ttl_s = 300
    try:
        lt_lines_max_items = max(8, int(os.getenv('LINE_TOOL_DATA_CACHE_MAX_ITEMS', '96') or '96'))
    except Exception:
        lt_lines_max_items = 96

    lt_lines_cache_key = (
        'lines',
        str(team),
        tuple(_normalize_season_id_list(season_ids)),
        str(ss),
        str(strength),
        str(xg_col),
        str(line_type),
        str(scope),
    )
    _cache_prune_ttl_and_size(_LT_DATA_CACHE, ttl_s=lt_lines_ttl_s, max_items=lt_lines_max_items)
    lt_lines_cached = _cache_get(_LT_DATA_CACHE, lt_lines_cache_key, lt_lines_ttl_s)
    if isinstance(lt_lines_cached, dict):
        j_cached = jsonify(lt_lines_cached)
        j_cached.headers['Cache-Control'] = 'no-store'
        return j_cached

    try:
        result = _compute_api_line_tool_lines(
            team, season_ids, ss, strength, xg_col, line_type, scope)
    except Exception:
        return jsonify({'combos': [], 'players': {}}), 200

    _cache_set_multi_bounded(_LT_DATA_CACHE, lt_lines_cache_key, result,
                             ttl_s=lt_lines_ttl_s, max_items=lt_lines_max_items)
    j = jsonify(result)
    j.headers['Cache-Control'] = 'no-store'
    return j


def _compute_api_line_tool_lines(
    team: str, season_ids: List[int], ss: str, strength: str,
    xg_col: str, line_type: str, scope: str,
) -> Dict[str, Any]:
    """Core logic for api_line_tool_lines, extracted so the route can cache + guard it."""
    # Fast targeted lookup: use game_data (filtered by team+season) to get the
    # ~20-30 relevant player IDs, then fetch only their name/position.
    _team_pids = _get_team_pids_for_seasons(team, season_ids)
    pid_info = _load_player_info_targeted(_team_pids) if _team_pids else {}
    if not pid_info:
        pid_info = _load_player_info_for_seasons(season_ids)

    if scope == 'league':
        tbl = 'forward_lines' if line_type == 'fwd' else 'defense_pairings'
        xg_f_col = {'xg_f': 'xgf', 'xg_s': 'xgf_s', 'xg_f2': 'xgf_f2'}.get(xg_col, 'xgf')
        xg_a_col = {'xg_f': 'xga', 'xg_s': 'xga_s', 'xg_f2': 'xga_f2'}.get(xg_col, 'xga')
        stage_map = {'regular': '2', 'playoffs': '3'}
        combo_acc: Dict[Tuple[str, Tuple[str, ...]], Dict[str, Any]] = {}
        live_keys = set()
        for season_id in season_ids:
            db_filters: Dict[str, str] = {'season': f'eq.{int(season_id)}'}
            if ss in stage_map:
                db_filters['season_stage'] = f'eq.{stage_map[ss]}'
            rows = _sb_read(tbl, filters=db_filters)
            team_live_combos = _compute_line_tool_team_combos(team, str(season_id), ss, strength, xg_col, line_type, pid_info)
            for combo in team_live_combos:
                combo_key = (str(combo.get('team', '')), tuple(combo.get('players', [])))
                live_keys.add(combo_key)
                acc = combo_acc.setdefault(combo_key, {
                    'gp': 0,
                    'toi': 0.0,
                    'cf': 0,
                    'ca': 0,
                    'ff': 0,
                    'fa': 0,
                    'sf': 0,
                    'sa': 0,
                    'gf': 0,
                    'ga': 0,
                    'xgf': 0.0,
                    'xga': 0.0,
                })
                _accumulate_line_tool_combo(acc, combo)

            for r in (rows or []):
                toi = float(r.get('toi') or 0)
                if toi < 0.1:
                    continue
                pids = tuple(str(r.get('player_ids', '')).split())
                combo_key = (str(r.get('team', '')), pids)
                if combo_key in live_keys:
                    continue
                acc = combo_acc.setdefault(combo_key, {
                    'gp': 0,
                    'toi': 0.0,
                    'cf': 0,
                    'ca': 0,
                    'ff': 0,
                    'fa': 0,
                    'sf': 0,
                    'sa': 0,
                    'gf': 0,
                    'ga': 0,
                    'xgf': 0.0,
                    'xga': 0.0,
                })
                _accumulate_line_tool_combo(acc, {
                    'gp': int(r.get('gp') or 0),
                    'toi': toi,
                    'cf': int(r.get('cf') or 0),
                    'ca': int(r.get('ca') or 0),
                    'ff': int(r.get('ff') or 0),
                    'fa': int(r.get('fa') or 0),
                    'sf': int(r.get('sf') or 0),
                    'sa': int(r.get('sa') or 0),
                    'gf': int(r.get('gf') or 0),
                    'ga': int(r.get('ga') or 0),
                    'xgf': float(r.get(xg_f_col) or 0.0),
                    'xga': float(r.get(xg_a_col) or 0.0),
                })

        combos = [
            _finalize_line_tool_combo(team_abbr, pids, acc)
            for (team_abbr, pids), acc in combo_acc.items()
            if float(acc.get('toi') or 0.0) >= 0.1
        ]
        players_out = {}
        for combo in combos:
            for pid in combo['players']:
                if pid not in players_out:
                    info = pid_info.get(pid, {})
                    players_out[pid] = {
                        'name': info.get('name', f'#{pid}'),
                        'position': info.get('position', ''),
                    }
        combos.sort(key=lambda c: c['toi'], reverse=True)
        return {'combos': combos, 'players': players_out}

    combo_acc2: Dict[Tuple[str, Tuple[str, ...]], Dict[str, Any]] = {}
    for season_id in season_ids:
        for combo in _compute_line_tool_team_combos(team, str(season_id), ss, strength, xg_col, line_type, pid_info):
            combo_key = (str(combo.get('team', '')), tuple(combo.get('players', [])))
            acc = combo_acc2.setdefault(combo_key, {
                'gp': 0,
                'toi': 0.0,
                'cf': 0,
                'ca': 0,
                'ff': 0,
                'fa': 0,
                'sf': 0,
                'sa': 0,
                'gf': 0,
                'ga': 0,
                'xgf': 0.0,
                'xga': 0.0,
            })
            _accumulate_line_tool_combo(acc, combo)
    combos = [
        _finalize_line_tool_combo(team_abbr, pids, acc)
        for (team_abbr, pids), acc in combo_acc2.items()
        if float(acc.get('toi') or 0.0) >= 0.1
    ]
    return {'combos': combos, 'players': pid_info}


@main_bp.route('/teams')
def teams_page():
    """Teams page (team card + team table/charts)."""
    return render_template(
        'teams.html',
        teams=TEAM_ROWS,
        active_tab='Teams',
        show_season_state=False,
        show_include_historic=True,
    )


@main_bp.route('/card-builder')
def card_builder_page():
    """Grid-based 16:9 social card builder for skaters, goalies, and teams."""
    return render_template(
        'card_builder.html',
        teams=TEAM_ROWS,
        active_tab='Card Builder',
        show_season_state=False,
        show_include_historic=True,
    )


_ABOUT_GLOSSARY_SECTIONS: List[Dict[str, Any]] = [
    {
        'title': 'Game States',
        'items': [
            {'abbr': '5v5', 'name': 'Five-on-Five', 'description': 'Even-strength play with five skaters aside.'},
            {'abbr': 'PP', 'name': 'Power Play', 'description': 'A team skates with a manpower advantage after an opponent penalty.'},
            {'abbr': 'PK', 'name': 'Penalty Kill', 'description': 'A team defends while shorthanded after taking a penalty.'},
            {'abbr': 'SH', 'name': 'Shorthanded', 'description': 'Play while a team has fewer skaters on the ice than the opponent.'},
            {'abbr': 'EV', 'name': 'Even Strength', 'description': 'Any non-special-teams situation with equal manpower.'},
        ],
    },
    {
        'title': 'Ice Time And Usage',
        'items': [
            {'abbr': 'GP', 'name': 'Games Played', 'description': 'Number of games included for the player or team.'},
            {'abbr': 'TOI', 'name': 'Time On Ice', 'description': 'Total ice time, usually shown in minutes.'},
            {'abbr': 'QoT', 'name': 'Quality of Teammates', 'description': 'Estimate of the strength of a player’s typical teammates.'},
            {'abbr': 'QoC', 'name': 'Quality of Competition', 'description': 'Estimate of the strength of the opponents a player usually faces.'},
            {'abbr': 'ZS', 'name': 'Zone Starts', 'description': 'Deployment context showing how favorable or defensive a player’s starting shifts are.'},
        ],
    },
    {
        'title': 'Shot And Chance Metrics',
        'items': [
            {'abbr': 'CF', 'name': 'Corsi For', 'description': 'All shot attempts for while the player or team is on the ice.'},
            {'abbr': 'CA', 'name': 'Corsi Against', 'description': 'All shot attempts against while the player or team is on the ice.'},
            {'abbr': 'FF', 'name': 'Fenwick For', 'description': 'Unblocked shot attempts for while the player or team is on the ice.'},
            {'abbr': 'FA', 'name': 'Fenwick Against', 'description': 'Unblocked shot attempts against while the player or team is on the ice.'},
            {'abbr': 'SF', 'name': 'Shots For', 'description': 'Shots on goal for while the player or team is on the ice.'},
            {'abbr': 'SA', 'name': 'Shots Against', 'description': 'Shots on goal against while the player or team is on the ice.'},
            {'abbr': 'xG', 'name': 'Expected Goals', 'description': 'Model-based estimate of how many goals a set of shots should produce.'},
            {'abbr': 'xGF', 'name': 'Expected Goals For', 'description': 'Expected goals generated by the player or team.'},
            {'abbr': 'xGA', 'name': 'Expected Goals Against', 'description': 'Expected goals allowed by the player or team.'},
            {'abbr': 'ixG', 'name': 'Individual Expected Goals', 'description': 'Expected goals from the player’s own shots.'},
        ],
    },
    {
        'title': 'Results And Percentages',
        'items': [
            {'abbr': 'GF', 'name': 'Goals For', 'description': 'Goals scored by the player’s team while he is on the ice.'},
            {'abbr': 'GA', 'name': 'Goals Against', 'description': 'Goals allowed by the player’s team while he is on the ice.'},
            {'abbr': 'GSAx', 'name': 'Goals Saved Above Expected', 'description': 'Goalie metric comparing actual goals allowed to expected goals against.'},
            {'abbr': 'GAx', 'name': 'Goals Above Expected', 'description': 'Difference between actual goals and expected goals in player shooting views.'},
            {'abbr': 'Sh%', 'name': 'Shooting Percentage', 'description': 'Goals divided by shots on goal.'},
            {'abbr': 'Sv%', 'name': 'Save Percentage', 'description': 'Saves divided by shots on goal against.'},
            {'abbr': 'xSv%', 'name': 'Expected Save Percentage', 'description': 'Save percentage implied by expected goals against.'},
            {'abbr': 'dSv%', 'name': 'Delta Save Percentage', 'description': 'Actual save percentage minus expected save percentage.'},
            {'abbr': 'PDO', 'name': 'PDO', 'description': 'Combined on-ice shooting and save percentage, often used as a puck-luck indicator.'},
        ],
    },
    {
        'title': 'Playmaking And Discipline',
        'items': [
            {'abbr': 'A1', 'name': 'Primary Assists', 'description': 'Assists awarded to the last passer before a goal scorer.'},
            {'abbr': 'A2', 'name': 'Secondary Assists', 'description': 'Second assists awarded on a goal.'},
            {'abbr': 'PIM', 'name': 'Penalty Minutes', 'description': 'Minutes assessed for penalties.'},
        ],
    },
    {
        'title': 'Modeling And Projection Terms',
        'items': [
            {'abbr': 'RAPM', 'name': 'Regularized Adjusted Plus-Minus', 'description': 'Model that estimates player impact while adjusting for teammates, opponents, and usage.'},
        ],
    },
]

_ABOUT_HEADLINES: List[str] = [
    'Inside the App',
    'What the App Does',
    'How to Use the App',
    'How to Interpret the Data',
    'Understanding the xG Models',
    'Understanding RAPM',
    'A Practical Way to Read the App',
    'Why This App Matters',
    'Please Help Share the App and Support the Work',
    'Future Perspectives',
    'Glossary',
]

_ABOUT_SECTION_TEXT: Dict[str, str] = {
    'inside-the-app': """Inside the App: A Modern NHL Analytics Platform for Fans, Analysts, and Hockey Nerds

There are a lot of hockey sites that show scores, standings, and player stats. There are fewer that try to connect those things into one place and answer the bigger questions: What happened in this game? Which players are truly driving results? How dangerous were those chances? Which lines are working? How does a player look beyond points? Where is a team creating its edge?

That is the problem this app is built to solve.

At its core, the app is an NHL analytics workspace. It combines schedule and live-game information, player and team dashboards, game reports, expected goals models, RAPM, lineup-based tools, odds history, and projection views into one connected experience. It is designed for people who want more than surface-level hockey stats, but it is also designed to remain usable for people who are still learning how to read modern hockey analytics.

This article walks through what the app does, how to use it, how to interpret the numbers, how the xG and RAPM models work, and where the platform can go next.""",
    'what-the-app-does': """What the App Does

The app is not a single leaderboard or a single model. It is a set of connected tools that let you move from a league-wide view down to a single game, single team, single line, or single player.

1. Schedule and game discovery

The home page is the front door. It gives you a schedule view that makes it easy to browse the calendar and jump into individual games. Instead of forcing users to start with raw data tables, the app starts with the hockey itself: what games are being played, what happened, and where you want to dig deeper.

This matters because hockey analysis usually starts with context. Before you look at a RAPM chart or an xG split, you need to know which game, team, or player you are trying to understand.

The schedule, standings and games include all games to ever have been played in the NHL - Dating back to the 1917/1918 season.
(Picture1)
2. Live Games

The Live Games page surfaces games in progress in a format that is fast to scan. It is built for active follow-up rather than postgame reading. If you want to see what is happening right now, move directly from the schedule into the live layer and then into the full game page.

For people who follow a slate of games at once, this page acts like a control panel.
(live_games_image)

3. Standings

The Standings page gives the seasonal league view. That may sound simple, but standings matter because they are the baseline everyone already understands. The app uses standings as a bridge between familiar hockey language and more advanced analysis.

You can think of standings as the answer to "what happened in the standings table," while the rest of the app increasingly answers "why?"
(Picture2)

4. Game pages

The game page is one of the most important parts of the platform. It is where raw game context becomes real analysis.

A game page includes:
- A report view
- Lineups
- Play-by-play
- Shifts

The report view is where a lot of users will spend most of their time. This is where you can move beyond final score and start asking questions about shot quality, territory, pressure, and momentum.
(Picture3)
(Picture4)
The play-by-play view lets you inspect the event stream directly. The shifts view lets you connect those events to who was actually on the ice. That is a major difference between a stats site and an analytics platform: the app is not just storing outcomes, it is trying to connect outcomes to players, matchups, usage, and context.
You can download raw Play-by-Play and Shift data directly from the game pages.

5. Skaters page

The Skaters page is a full player-analysis workspace. It includes:
- Card
- Charts
- Shooting
- Edge
- RAPM
- Projections
- Table View

This is where a user can move from a specific player question to a broader league comparison.

The Card tab gives a fast profile of the player. You can customize exactly what metrics you prefer the Card to show. The bar size and color indicate the percentile ranking.
(Picture5)
The Charts tab is for comparison and positioning. You can choose what metrics you want to compare.
(Picture6)
The Shooting tab helps explain finishing, shot patterns, and chance generation. You can filter the shots versus a specific goaltender, by clicking in the Goalie table. Likewise, you can filter the data by shot location using the Heat Map. The video button shows goals by the selected player.
(Picture7)
The Edge tab gives another layer of visual and context-driven interpretation. It loads the NHL EDGE data directly from the NHL Api.
(Picture8)
The RAPM tab isolates player impact more directly. You can read more about the RAPM models below. You can choose between different metrics and different outputs.
(Picture9)
The Projections tab provides forward-looking context. It shows the player projections used for the game projection model.
(Picture10)
The Table View lets you zoom out and compare one player against a larger peer group. You can configure the table view as you like, and all table views are downloadable.
(Picture11)

If you care about player evaluation, this is one of the app's deepest surfaces.

6. Goalies page

Goalies have their own dedicated space because skater logic and goalie logic should not be treated as the same problem.

The Goalies page includes:
- Card
- Charts
- Goaltending
- Table View
(Picture12)
Goaltending is unusually noisy and unusually dependent on environment, workload, shot quality, and game state. A dedicated goalie page helps separate those questions cleanly instead of forcing goalie analysis into skater frameworks that do not fit.

7. Teams page

The Teams page shifts the lens from player-level evaluation to team identity and team performance.

It includes:
- Card
- Charts
- Edge
- Projections
- Table View

This is where you can ask questions like:
- Is this team winning through shot volume or shot quality?
- Are they controlling play territorially?
- Are they overperforming their underlying numbers?
- How do they compare to the rest of the league?
- Where are their strengths and weaknesses coming from?

For many users, the Teams page will be the easiest gateway into analytics because team-level patterns are often easier to see before moving into player isolation.

8. Line Tool

The Line Tool is one of the most distinctive parts of the app.

It includes:
- A Heat Map view
- WOWY analysis
- Video integration
- Line, pair, and player selection
- KPI panels and zone-based filtering

You can choose up to 5 players, but you can also just pick just one player to see his On-Ice statistics. You can also include a goaltender, to see if a player performs better in front of one goalie over the other. The Video button will show the goals with selected players on the ice together - Both for and against.
(Picture13)
In the heat map you can compare the performance to the team or the league. If the color is blue, it means that the selected players are performing better than the team average or league average. So, in the defensive zone blue means fewer shots and in the offensive zone blue means more shots.
(Picture14)
This is the kind of tool that turns data from static information into something you can actively test.

You can use it to explore how combinations of players perform together, where they create or allow chances, and how results change depending on teammates. The WOWY view is especially useful when you want to ask whether a line's results are truly driven by the group or whether one player is carrying the outcome.
On the WOWY page you can see the results with and without the selected players. It's setup so you see the data from the perspective of a particular player. You can of course change the perspective.
(Picture15)

This kind of lineup analysis is one of the hardest things to do well in hockey, and it is one of the places where the app becomes more than a dashboard.

9. Odds page

The app also includes odds tracking and odds history. That gives users another lens on games: not only what the model says, but how the market has moved. The odds data is from DraftKings via the NHL Api and it only gets updated from 12PM ET.
(Picture16)
That does not mean the app is only for betting. It means the platform understands that betting markets are one of the most useful real-world signals in hockey. Odds movement can reflect lineup news, goalie confirmations, injury updates, and market sentiment. When combined with the rest of the app, odds history becomes context rather than noise.

10. Projections page

The Projections page includes:
- Games
- Series
- Bracket
- Stanley Cup Probability
(Picture17)
(Picture18)

This gives the app a forward-looking layer on top of the descriptive and diagnostic layers. Users can look at individual games, playoff series, bracket paths, and title probabilities in one place.

This article is deliberately not going into the internals of the projection model, but from a product point of view, the projection surfaces are important because they connect the rest of the platform to decision-making and future-looking analysis.

Getting access to the Projections page will require a subscription. This is what will cover the expenses, so even if you're not particularly interested in the game projections, you can help support the site by subscribing.

11. Update and maintenance surfaces

There are also update and admin capabilities in the app. Most users will never interact with them directly, but they matter because the platform is built around frequent refreshes and operational workflows, not static CSVs sitting on a forgotten page.

That is part of what makes the app feel alive.""",
    'how-to-use-the-app': """How to Use the App

One of the biggest challenges in analytics is not building the model. It is helping people use it correctly.

The best way to use this app is to think in layers.

Start broad, then narrow

A good workflow looks like this:
1. Start on the schedule, live games, standings, or team page.
2. Identify the game, team, or player you want to understand.
3. Use team and player pages to form an initial hypothesis.
4. Drop into the game page or line tool when you need to inspect the "why."

For example:
- If a team has been winning a lot, go to the Teams page and see whether that success is backed by chance quality or finishing.
- If a player's point totals look weak, go to the Skaters page and see whether the underlying play-driving data tells a different story.
- If a line is crushing territorially but not scoring, use the Line Tool and game reports to see whether the problem is finishing, shot quality, or usage.
- If a game result looks surprising, open the game page and inspect the play-by-play, shifts, and report tabs.

The app works best when you move between these surfaces instead of treating one chart as the whole answer.

Use filters intentionally

A lot of pages allow you to filter by:
- Season
- Season state
- Strength state
- Rates vs totals
- xG model
- Minimum thresholds like games played or time on ice

These are not cosmetic controls. They change the question you are asking.

If you choose totals, you are asking who produced the most overall value or volume. If you choose rates, you are asking who performed best on a per-minute or per-game basis. If you change strength state, you are separating even-strength results from special-teams results. If you set a minimum threshold, you are deciding how much you want to trade inclusiveness for reliability.

That matters because hockey is a small-sample sport. A player can look elite in 100 minutes and ordinary in 1,000. The app gives you the tools to manage that tension, but the user still has to decide what kind of comparison is fair.""",
    'how-to-interpret-the-data': """How to Interpret the Data

The app includes a lot of numbers, but most of them can be grouped into a few simple questions.

1. Results: what happened?

These are the most familiar measures:
- Goals
- Wins
- Points
- Save percentage
- Special teams outcomes
- Scorelines

They matter, but they are not enough on their own. Results can be noisy, especially in hockey.

2. Process: how did it happen?

This is where metrics like:
- Corsi
- Fenwick
- xG
- shot location
- zone context
- chance distributions
- on-ice event rates

become important.

Process metrics help answer whether performance is sustainable. A team winning despite weak shot quality may be living on finishing or goaltending. A player with modest point totals but strong chance-driving metrics may be better than the box score suggests.

3. Context: against whom, with whom, and in what role?

This is where hockey gets more complex.

A player's numbers are shaped by:
- Teammates
- Opponents
- Usage
- Zone starts
- Special teams role
- Score state
- Game state
- Team system

The app tries to keep that context visible instead of flattening everything into one number.

4. Isolation: what is the player or team actually contributing?

This is where RAPM and lineup tools become especially useful. They are designed to go beyond "what happened while this player was on the ice" and move closer to "what effect did this player have?"

No model can solve that perfectly. But the right tools can get you much closer than raw plus-minus, point totals, or on-ice goal share.""",
    'understanding-the-xg-models': """Understanding the xG Models

Expected goals, or xG, is one of the core ideas in modern hockey analysis. The goal of xG is simple: estimate the probability that a shot becomes a goal.

That sounds straightforward, but the real value of xG is not the probability assigned to one shot. The real value is what happens when you add those probabilities up across shifts, periods, games, players, teams, and seasons.

In this app, xG is used in multiple surfaces, and the app exposes different xG model choices because different event definitions can be useful for different analytical purposes.

What xG is trying to measure

At a high level, xG asks:

If we saw this exact shot or shot-like event many times, how often would it become a goal?

That lets you separate chance quality from actual finishing results.

A team may score four goals on 1.8 expected goals because they finished well, got a few bounces, or faced weak goaltending. Another team may score once on 3.1 expected goals because they generated better chances than the final score suggests.

That distinction is critical if you care about repeatability.

The three xG model options in the app

The app exposes three xG model views:

- xG_S
- xG_F
- xG_F2

These are not three completely unrelated philosophies. They are three related model variants.

xG_S

This is the shot-based model. It is trained on shot events and uses contextual features like:
- venue
- shot type
- score state
- rink context
- strength state
- box or ice location
- last event context

This is the cleanest traditional shot-model view in the app. If you want a direct shot-quality lens, this is the natural place to start. This is probably the best model for goaltender analysis - Depending on whether or not you believe can impact shot misses... And to what extent.

xG_F

This is a Fenwick-based model. Fenwick excludes blocked shots and focuses on unblocked attempts, which often gives a broader attacking-process view than shots on goal alone.

This model includes the same core contextual structure as the shot model, but also adds event-sequence context such as the last event. That can help the model capture how a chance developed, not just where it ended.
This is the preferred model for describing results.

xG_F2

This is also Fenwick-based, but with a slightly different feature set than xG_F. It uses venue, shot type, score state, rink context, strength state, and box location, while leaving out the last-event input.
This is the preferred model for predictive analysis. A rebound shot have a much larger chance of becoming a goal, but they are also quite random. This is why excluding rebound effects increases the predictiveness of the model.


How the models are trained

The xG scripts train gradient-boosted tree classifiers using rolling multi-season windows. In plain English, that means the model is trained on recent historical seasons rather than treating all hockey history as equally relevant forever.

That matters because the league changes:
- shot habits change
- team systems evolve
- tracking and event recording can drift
- scoring environments move over time

Using rolling windows is a practical way to keep the models current without overreacting to only one season.

The model inputs are categorical and contextual features that are converted into machine-readable form, then fed into XGBoost classifiers that estimate goal probability.

How to interpret xG in the app

A few practical rules help:

- xG is about chance quality, not certainty.
- Over one game, xG is descriptive, not definitive.
- Over larger samples, xG becomes more informative.
- xG is stronger when combined with usage and context, not read in isolation.
- Differences between xG and goals can reveal finishing, goaltending, luck, or short-term variance.

If a player consistently beats xG over many seasons, that may reflect real finishing talent. If a team beats xG for two weeks, that might just be a hot streak. The app is built to help you tell those apart.""",
    'understanding-rapm': """Understanding RAPM

RAPM stands for Regularized Adjusted Plus-Minus.

It sounds technical because it is technical. But the intuition is simple: hockey is a five-man game at even strength and a deeply contextual game everywhere else. If you want to estimate individual impact, you need a method that tries to separate a player from his teammates, opponents, and deployment.

That is what RAPM is trying to do.

Why raw plus-minus is not enough

Raw plus-minus tells you what happened while a player was on the ice. It does not tell you how much of that result belongs to the player.

If someone spends all night with elite linemates against weak competition, raw plus-minus will flatter him. If someone plays brutal defensive minutes with weak support, raw plus-minus can punish him unfairly.

RAPM exists because hockey is not a clean one-player sport.

How RAPM works in this app

The RAPM pipeline in the app:
- builds shift-segment data from play-by-play and shifts
- aggregates event outcomes by shift
- tracks who was on the ice
- models different outcomes with ridge regression
- separates strength states like 5v5, power play, and penalty kill
- outputs offensive, defensive, and differential views across metrics like Corsi, goals, and xG

In other words, the model tries to learn how outcomes move when players are present, while controlling for the fact that players share the ice with teammates and opponents.

The regularized part matters a lot. Hockey data is messy and collinear. The same players appear together repeatedly. Ridge regression helps stabilize those estimates and prevents the model from overreacting to noisy lineup combinations.

What RAPM is measuring

In the app, RAPM includes outputs tied to:
- Corsi
- goals
- xG
- offensive and defensive components
- plus-minus style differentials
- power-play and shorthanded states

That means RAPM is not just one number. It is a family of adjusted impact estimates.

A player may:
- look strong offensively but weak defensively
- grade well in xG impact but not goal results
- show stronger value at 5v5 than on special teams
- have more value in shot suppression than in shot creation

That is the point. Good evaluation should separate those things.

How to read RAPM correctly

A few rules matter here too:

- RAPM is an estimate, not a verdict.
- It is strongest over larger samples.
- It should be compared within role and context.
- Offensive and defensive components should be read together.
- xG-based RAPM is often more stable than goal-based RAPM.
- Special teams RAPM should be treated carefully because the samples are smaller.

If a player has strong xG RAPM and weaker goal RAPM, that often means the underlying process is better than the scoreboard results. If a player rates well in both, the case is stronger. If RAPM, xG, and lineup impacts all agree, confidence goes up.

The app gives you multiple ways to test that agreement.""",
    'a-practical-way-to-read-the-app': """A Practical Way to Read the App

If you are new to hockey analytics, this is a good mental model:

For teams
Use:
- Standings for results
- Teams page for identity
- Game page for game-level explanation
- Projections page for future-facing context

For skaters
Use:
- Card for summary
- Charts for comparison
- Shooting for chance and finishing context
- RAPM for isolation
- Table View for league benchmarking
- Line Tool when you want to test teammate effects

For goalies
Use:
- Card and Goaltending views first
- Team and game context second
- Never interpret goalie results without chance quality context

For single games
Use:
- Game report first
- Play-by-play second
- Shifts when you want to connect events to personnel
- Odds and live context when you want to understand how information moved before and during the game

For lineup questions
Use:
- Line Tool
- Heat map
- WOWY
- Video
- Game report confirmation

That last step matters. Numbers are strongest when they point you toward something you can then inspect more directly.""",
    'why-this-app-matters': """Why This App Matters

There are two kinds of hockey analytics products.

The first kind gives you numbers.

The second kind helps you think.

This app is trying to be the second kind.

It is built around the idea that hockey data becomes useful when users can move across levels:
- from schedule to game
- from team to player
- from result to process
- from process to context
- from context to interpretation

That is what makes a platform valuable. Not just one stat, one leaderboard, or one model, but the ability to connect the pieces.""",
    'please-help-share-the-app-and-support-the-work': """Please Help Share the App and Support the Work

If you use the app and find it valuable, the simplest way to help is to share it.

Share it with:
- hockey fans
- fantasy players
- writers
- podcasters
- analysts
- team-level community accounts
- anyone who wants a smarter way to look at the game

Independent work grows because people pass it on.

If you want better public hockey analytics tools, more transparent models, more useful player and team dashboards, and more features that go beyond generic stat pages, support matters. Time, infrastructure, data pipelines, model maintenance, and product development all cost something. Every person who shares the app, talks about it, links to it, or supports the project directly helps keep the work going.

If you want this kind of platform to improve, please help put it in front of more people.""",
    'future-perspectives': """Future Perspectives

The app already covers a lot of ground, but there is still plenty of room to grow.

A few directions that make sense for the future:

1. More historical depth
Deeper season archives, cross-era comparisons, and career trend tools would make the player and team pages even more useful.

2. Better lineup and matchup analysis
The Line Tool is already powerful, but there is room for richer matchup views, coach deployment patterns, and opposition-quality overlays.

3. More video-linked analytics
The bridge between numbers and film is one of the most interesting parts of modern sports analysis. More direct event-to-video connections would make the app even more valuable.

4. Custom dashboards
Different users care about different questions. Letting people save their own views, filters, tables, or metric bundles would make the app more personal and more efficient.

5. Alerts and tracking
Watchlists for players, teams, line combinations, or model-driven changes could turn the app from a destination into a daily tool.

6. Expanded game reporting
Game pages could grow into full postgame analytical reports with automated summaries, turning each game into a richer story.

7. Public explainers and education
A bigger library of built-in metric explanations, tutorials, and interpretation guides would help newer users get comfortable faster.

8. Deeper team-style fingerprints
It would be useful to have even clearer team identity views: rush teams, cycle teams, forecheck-heavy teams, slot-denial teams, east-west power-play teams, and so on.

9. More goalie-specific modeling and visuals
Goaltending remains one of the hardest analytical problems in hockey. A deeper goalie toolkit would be a natural extension of the current platform.

10. Community-facing features
Saved screenshots, exportable reports, sharable charts, and public comparison pages would make the app easier to spread organically.

11. Adding contract context
Including contract information and analysis would help answer if a player is living up to his contract.

Closing

The goal of this app is not to replace watching hockey. It is to deepen it.

It is for the fan who wants to know whether a result was real.
It is for the analyst who wants more than points and plus-minus.
It is for the writer who wants sharper evidence.
It is for the hockey obsessive who wants one place to explore the game properly.

That is what makes the project worth building.

If you have used the app, shared it, talked about it, or supported the work in any way: thank you.

And if you want to help it grow, the best next step is simple: use it, share it, and tell other people why it matters.""",
}


def _about_slug_from_title(title: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '-', str(title or '').strip().lower())
    return slug.strip('-')


def _about_nav_items(active_slug: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for title in _ABOUT_HEADLINES:
        slug = _about_slug_from_title(title)
        out.append({
            'title': title,
            'slug': slug,
            'url': url_for('main.about_page_slug', section_slug=slug),
            'active': slug == active_slug,
        })
    return out


def _about_text_segments(text: str) -> List[Dict[str, Any]]:
    token_re = re.compile(r'\((Picture\d+|live_games_image)\)')
    segments: List[Dict[str, Any]] = []
    pos = 0
    for m in token_re.finditer(text or ''):
        if m.start() > pos:
            segments.append({'type': 'text', 'value': text[pos:m.start()]})
        token = m.group(1)
        if token == 'live_games_image':
            fname = 'live_games_image.png'
        else:
            fname = f'{token}.png'
        fpath = _static_path(os.path.join('about', fname))
        segments.append({
            'type': 'image',
            'token': token,
            'filename': fname,
            'url': url_for('static', filename=f'about/{fname}'),
            'exists': os.path.exists(fpath),
        })
        pos = m.end()
    if pos < len(text or ''):
        segments.append({'type': 'text', 'value': text[pos:]})
    return segments


def _about_strip_leading_heading(text: str, heading: str) -> str:
    raw = str(text or '')
    if not raw.strip():
        return raw
    lines = raw.splitlines()
    if not lines:
        return raw

    first = lines[0].strip().rstrip(':').lower()
    expected = str(heading or '').strip().rstrip(':').lower()
    if first != expected:
        return raw

    idx = 1
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    return '\n'.join(lines[idx:])


@main_bp.route('/about')
def about_page():
    return redirect(url_for('main.about_page_slug', section_slug=_about_slug_from_title(_ABOUT_HEADLINES[0])))


@main_bp.route('/about/<section_slug>')
def about_page_slug(section_slug: str):
    """About page with one headline section per route."""
    section_slug = str(section_slug or '').strip().lower()
    if not section_slug:
        return redirect(url_for('main.about_page'))

    valid_slugs = {_about_slug_from_title(x) for x in _ABOUT_HEADLINES}
    if section_slug not in valid_slugs:
        return redirect(url_for('main.about_page'))

    is_glossary = (section_slug == _about_slug_from_title('Glossary'))
    section_title = next((x for x in _ABOUT_HEADLINES if _about_slug_from_title(x) == section_slug), 'About')
    section_text = _about_strip_leading_heading(_ABOUT_SECTION_TEXT.get(section_slug, ''), section_title)
    segments = _about_text_segments(section_text) if section_text else []

    return render_template(
        'about.html',
        teams=TEAM_ROWS,
        active_tab='About',
        show_filters=True,
        show_season_state=False,
        show_include_historic=False,
        glossary_sections=_ABOUT_GLOSSARY_SECTIONS,
        about_headlines=_about_nav_items(section_slug),
        about_section_slug=section_slug,
        about_section_title=section_title,
        about_segments=segments,
        about_is_glossary=is_glossary,
    )


@main_bp.route('/odds/<int:game_id>')
def odds_page(game_id: int):
    """Odds page showing ML history for a game (from Sheet1)."""
    # Keep the primary nav highlighted on Game Projections.
    return render_template('odds.html', teams=TEAM_ROWS, game_id=game_id, active_tab='Game Projections', show_season_state=False)


# Projections enrichment cache: {playerId -> {name, team, pos}}
_ALL_ROSTERS_CACHE: Optional[Tuple[float, Dict[int, Dict[str, str]]]] = None

# NHL skater bios cache by seasonId: {seasonId -> (timestamp, {playerId -> info})}
_SKATER_BIOS_CACHE: Dict[int, Tuple[float, Dict[int, Dict[str, str]]]] = {}


@main_bp.route('/api/odds/history/<int:game_id>')
def api_odds_history(game_id: int):
    """Return ML history time series for both teams for a given game id.

    Data source: Supabase odds_snapshots.
    """
    # Team colors from Teams.csv
    color_by_team: Dict[str, str] = {}
    try:
        for row in TEAM_ROWS:
            t = (row.get('Team') or '').strip().upper()
            c = (row.get('Color') or '').strip()
            if t and c:
                color_by_team[t] = c
    except Exception:
        color_by_team = {}

    away_abbrev, home_abbrev = _load_game_team_abbrevs(int(game_id))
    snapshot_rows = _load_odds_snapshot_rows(game_id=int(game_id))
    _latest_snapshot, points_by_team = _build_odds_snapshot_payloads(
        snapshot_rows,
        away_abbrev,
        home_abbrev,
    )
    if points_by_team:
        teams_out: List[Dict[str, Any]] = []
        for team_s, points in points_by_team.items():
            teams_out.append({'abbrev': team_s, 'color': color_by_team.get(team_s) or None, 'points': points})
        j = jsonify({'gameId': int(game_id), 'teams': teams_out})
        try:
            j.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
        return j

    # No odds data available for this game.
    j = jsonify({'gameId': int(game_id), 'teams': []})
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/lineups/all')
def api_lineups_all():
    """Return lineup data used by the projections lineup selector.

    Source is Supabase dailyfaceoff_lineups.
    """
    try:
        data = _load_lineups_all()
        j = jsonify(data)
        try:
            j.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
        return j
    except Exception as e:
        # Log full error server-side, but return only a safe, actionable code to clients.
        try:
            print('[api_lineups_all] load failed:', repr(e))
        except Exception:
            pass
        msg = str(e or '')
        code = 'lineups_load_failed'
        j = jsonify({
            'error': code,
            'hint': 'Check Supabase access for dailyfaceoff_lineups',
        })
        try:
            j.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
        return j, 500


@main_bp.route('/api/player-projections/sheets')
def api_player_projections_sheets():
    """Fetch player projections from Supabase (falls back to CSV).
    Returns: { playerId: { PlayerID, Position, Age, Rookie, EVO, EVD, PP, SH, GSAx, ... }, ... }
    """
    data = _load_player_projections_from_sheets()
    # Convert int keys to str for JSON
    out = {str(k): v for k, v in data.items()}
    j = jsonify(out)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/player-current-projections')
def api_player_current_projections():
    """Fetch current-season player projections for the lineup editor.

    Returns rows keyed by player id, preferring player_current_projections and falling back
    to the legacy player projections source when the current table is unavailable.
    """
    data = _load_current_player_projections_cached()
    out = {str(k): v for k, v in data.items()}
    j = jsonify(out)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


_PLAYOFF_SERIES_HOME_PATTERN: Tuple[bool, ...] = (True, True, False, False, True, False, True)
_PLAYOFF_RESTED_RESTED_SITUATION = -0.153396566
_PLAYOFF_SERIES_ORDER: Tuple[str, ...] = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O')
_PLAYOFF_SERIES_CHILDREN: Dict[str, Tuple[str, str]] = {
    'I': ('A', 'B'),
    'J': ('C', 'D'),
    'K': ('E', 'F'),
    'L': ('G', 'H'),
    'M': ('I', 'J'),
    'N': ('K', 'L'),
    'O': ('M', 'N'),
}
_PLAYOFF_BRACKET_ROUNDS: Dict[str, Tuple[str, ...]] = {
    'eastRound1': ('A', 'B', 'C', 'D'),
    'westRound1': ('E', 'F', 'G', 'H'),
    'eastRound2': ('I', 'J'),
    'westRound2': ('K', 'L'),
    'conferenceFinals': ('M', 'N'),
    'final': ('O',),
}


def _current_playoff_bracket_year(now: Optional[datetime] = None) -> int:
    try:
        season_i = int(current_season_id(now))
        return int(str(season_i)[4:])
    except Exception:
        d = now or datetime.utcnow()
        return d.year if d.month < 9 else d.year + 1


def _load_playoff_bracket_cached(year: int) -> Dict[str, Any]:
    try:
        ttl_s = max(30, int(os.getenv('PLAYOFF_BRACKET_CACHE_TTL_SECONDS', '300') or '300'))
    except Exception:
        ttl_s = 300
    now = time.time()
    cached = _PLAYOFF_BRACKET_CACHE.get(int(year))
    if cached and (now - cached[0]) < ttl_s:
        return cached[1]

    url = f'https://api-web.nhle.com/v1/playoff-bracket/{int(year)}'
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            return {}
        data = resp.json() if resp.content else {}
    except Exception:
        return {}

    if isinstance(data, dict):
        _PLAYOFF_BRACKET_CACHE[int(year)] = (now, data)
        return data
    return {}


def _matchup_projection_summary(away_abbrev: str,
                                home_abbrev: str,
                                lineups_all: Dict[str, Any],
                                proj_map: Dict[int, Dict[str, Any]],
                                situation_value: float = _PLAYOFF_RESTED_RESTED_SITUATION) -> Dict[str, float]:
    proj_away = _team_proj_from_lineup(str(away_abbrev), lineups_all, proj_map)
    proj_home = _team_proj_from_lineup(str(home_abbrev), lineups_all, proj_map)
    dproj = proj_away - proj_home
    win_away = 1.0 / (1.0 + math.exp(-(dproj) - situation_value))
    return {
        'projAway': float(proj_away),
        'projHome': float(proj_home),
        'dProj': float(dproj),
        'winProbAway': float(win_away),
        'winProbHome': float(1.0 - win_away),
        'situationValue': float(situation_value),
    }


def _compute_series_outcome_distribution(top_abbrev: str,
                                         bottom_abbrev: str,
                                         top_wins: int,
                                         bottom_wins: int,
                                         lineups_all: Dict[str, Any],
                                         proj_map: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    total_played = max(0, int(top_wins) + int(bottom_wins))
    future_games: List[Dict[str, Any]] = []
    for game_index in range(total_played, len(_PLAYOFF_SERIES_HOME_PATTERN)):
        top_is_home = bool(_PLAYOFF_SERIES_HOME_PATTERN[game_index])
        if top_is_home:
            home_abbrev = str(top_abbrev or '').upper()
            away_abbrev = str(bottom_abbrev or '').upper()
            summary = _matchup_projection_summary(away_abbrev, home_abbrev, lineups_all, proj_map)
            top_win_prob = float(summary['winProbHome'])
        else:
            home_abbrev = str(bottom_abbrev or '').upper()
            away_abbrev = str(top_abbrev or '').upper()
            summary = _matchup_projection_summary(away_abbrev, home_abbrev, lineups_all, proj_map)
            top_win_prob = float(summary['winProbAway'])
        future_games.append({
            'gameNumber': int(game_index + 1),
            'homeTeam': home_abbrev,
            'awayTeam': away_abbrev,
            'topSeedWinProb': float(top_win_prob),
            'bottomSeedWinProb': float(1.0 - top_win_prob),
        })

    states: Dict[Tuple[int, int], float] = {(int(top_wins), int(bottom_wins)): 1.0}
    for game in future_games:
        next_states: Dict[Tuple[int, int], float] = {}
        top_win_prob = float(game['topSeedWinProb'])
        for (cur_top, cur_bottom), prob in states.items():
            if cur_top >= 4 or cur_bottom >= 4:
                next_states[(cur_top, cur_bottom)] = next_states.get((cur_top, cur_bottom), 0.0) + float(prob)
                continue
            next_states[(cur_top + 1, cur_bottom)] = next_states.get((cur_top + 1, cur_bottom), 0.0) + float(prob) * top_win_prob
            next_states[(cur_top, cur_bottom + 1)] = next_states.get((cur_top, cur_bottom + 1), 0.0) + float(prob) * (1.0 - top_win_prob)
        states = next_states

    top_outcomes = {f'4-{losses}': 0.0 for losses in range(4)}
    bottom_outcomes = {f'4-{losses}': 0.0 for losses in range(4)}
    for (final_top, final_bottom), prob in states.items():
        if final_top == 4 and 0 <= final_bottom <= 3:
            top_outcomes[f'4-{final_bottom}'] += float(prob)
        elif final_bottom == 4 and 0 <= final_top <= 3:
            bottom_outcomes[f'4-{final_top}'] += float(prob)

    return {
        'futureGames': future_games,
        'topSeedOutcomeProbs': top_outcomes,
        'bottomSeedOutcomeProbs': bottom_outcomes,
        'topSeedSeriesWinProb': float(sum(top_outcomes.values())),
        'bottomSeedSeriesWinProb': float(sum(bottom_outcomes.values())),
    }


def _playoff_team_payload(team_info: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(team_info, dict):
        return {}
    return {
        'id': team_info.get('id'),
        'abbrev': team_info.get('abbrev'),
        'name': team_info.get('name'),
        'commonName': team_info.get('commonName'),
        'logo': team_info.get('logo'),
        'darkLogo': team_info.get('darkLogo'),
        'placeNameWithPreposition': team_info.get('placeNameWithPreposition'),
    }


def _playoff_team_payload_for_abbrev(abbrev: str, team_info_by_abbrev: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    team_info = team_info_by_abbrev.get(str(abbrev or '').upper()) or {'abbrev': str(abbrev or '').upper()}
    return _playoff_team_payload(team_info)


def _is_resolved_playoff_team(team_info: Dict[str, Any]) -> bool:
    if not isinstance(team_info, dict):
        return False
    abbrev = str(team_info.get('abbrev') or '').strip().upper()
    if not abbrev or abbrev in {'TBD', 'TBA'}:
        return False
    # NHL bracket placeholders usually lack a numeric team id.
    try:
        team_id = int(team_info.get('id')) if team_info.get('id') is not None else 0
    except Exception:
        team_id = 0
    return team_id > 0


def _playoff_probability_rows(prob_map: Dict[str, float],
                              team_info_by_abbrev: Dict[str, Dict[str, Any]],
                              limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for team_abbrev, prob in (prob_map or {}).items():
        p = float(prob or 0.0)
        if p <= 0:
            continue
        rows.append({
            'team': _playoff_team_payload_for_abbrev(team_abbrev, team_info_by_abbrev),
            'prob': p,
        })
    rows.sort(key=lambda item: (-float(item['prob']), str(((item.get('team') or {}).get('abbrev') or ''))))
    if isinstance(limit, int) and limit > 0:
        return rows[:limit]
    return rows


def _build_actual_playoff_series_node(item: Dict[str, Any],
                                      team_info_by_abbrev: Dict[str, Dict[str, Any]],
                                      lineups_all: Dict[str, Any],
                                      proj_map: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    top_team = _playoff_team_payload(item.get('topSeedTeam') or {})
    bottom_team = _playoff_team_payload(item.get('bottomSeedTeam') or {})
    top_abbrev = str(top_team.get('abbrev') or '').upper()
    bottom_abbrev = str(bottom_team.get('abbrev') or '').upper()
    top_wins = int(item.get('topSeedWins') or 0)
    bottom_wins = int(item.get('bottomSeedWins') or 0)

    series_dist = _compute_series_outcome_distribution(
        top_abbrev,
        bottom_abbrev,
        top_wins,
        bottom_wins,
        lineups_all,
        proj_map,
    )
    top_slot_probs = {top_abbrev: 1.0}
    bottom_slot_probs = {bottom_abbrev: 1.0}
    winner_prob_map = {
        top_abbrev: float(series_dist['topSeedSeriesWinProb']),
        bottom_abbrev: float(series_dist['bottomSeedSeriesWinProb']),
    }
    matchup_rows = [{
        'matchupProb': 1.0,
        'topTeam': top_team,
        'bottomTeam': bottom_team,
        'topTeamSeriesWinProb': float(series_dist['topSeedSeriesWinProb']),
        'bottomTeamSeriesWinProb': float(series_dist['bottomSeedSeriesWinProb']),
    }]

    return {
        'seriesUrl': item.get('seriesUrl'),
        'seriesTitle': item.get('seriesTitle'),
        'seriesAbbrev': item.get('seriesAbbrev'),
        'seriesLetter': item.get('seriesLetter'),
        'playoffRound': item.get('playoffRound'),
        'topSeedRank': item.get('topSeedRank'),
        'topSeedRankAbbrev': item.get('topSeedRankAbbrev'),
        'topSeedWins': top_wins,
        'bottomSeedRank': item.get('bottomSeedRank'),
        'bottomSeedRankAbbrev': item.get('bottomSeedRankAbbrev'),
        'bottomSeedWins': bottom_wins,
        'topSeedTeam': top_team,
        'bottomSeedTeam': bottom_team,
        'futureGames': series_dist['futureGames'],
        'topSeedOutcomeProbs': series_dist['topSeedOutcomeProbs'],
        'bottomSeedOutcomeProbs': series_dist['bottomSeedOutcomeProbs'],
        'topSeedSeriesWinProb': float(series_dist['topSeedSeriesWinProb']),
        'bottomSeedSeriesWinProb': float(series_dist['bottomSeedSeriesWinProb']),
        'topSlotTeamProbs': _playoff_probability_rows(top_slot_probs, team_info_by_abbrev),
        'bottomSlotTeamProbs': _playoff_probability_rows(bottom_slot_probs, team_info_by_abbrev),
        'winnerProbs': _playoff_probability_rows(winner_prob_map, team_info_by_abbrev),
        'matchups': matchup_rows,
        'winnerProbMap': winner_prob_map,
        'topSlotProbMap': top_slot_probs,
        'bottomSlotProbMap': bottom_slot_probs,
        'isProjected': False,
    }


def _build_projected_playoff_series_node(item: Dict[str, Any],
                                         top_slot_probs: Dict[str, float],
                                         bottom_slot_probs: Dict[str, float],
                                         team_info_by_abbrev: Dict[str, Dict[str, Any]],
                                         lineups_all: Dict[str, Any],
                                         proj_map: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    top_slot_probs = dict(top_slot_probs or {})
    bottom_slot_probs = dict(bottom_slot_probs or {})
    winner_prob_map: Dict[str, float] = {}
    top_slot_series_win_prob = 0.0
    bottom_slot_series_win_prob = 0.0
    matchups: List[Dict[str, Any]] = []

    # If only one slot is known so far, keep the series projected and carry that side forward at 100%.
    if top_slot_probs and not bottom_slot_probs:
        winner_prob_map = dict(top_slot_probs)
        top_slot_series_win_prob = float(sum(top_slot_probs.values()))
        bottom_slot_series_win_prob = 0.0
    elif bottom_slot_probs and not top_slot_probs:
        winner_prob_map = dict(bottom_slot_probs)
        top_slot_series_win_prob = 0.0
        bottom_slot_series_win_prob = float(sum(bottom_slot_probs.values()))

    for top_abbrev, top_appearance_prob in top_slot_probs.items():
        for bottom_abbrev, bottom_appearance_prob in bottom_slot_probs.items():
            matchup_prob = float(top_appearance_prob or 0.0) * float(bottom_appearance_prob or 0.0)
            if matchup_prob <= 0:
                continue
            series_dist = _compute_series_outcome_distribution(
                str(top_abbrev or '').upper(),
                str(bottom_abbrev or '').upper(),
                0,
                0,
                lineups_all,
                proj_map,
            )
            top_prob = float(series_dist['topSeedSeriesWinProb'])
            bottom_prob = float(series_dist['bottomSeedSeriesWinProb'])
            winner_prob_map[str(top_abbrev).upper()] = winner_prob_map.get(str(top_abbrev).upper(), 0.0) + matchup_prob * top_prob
            winner_prob_map[str(bottom_abbrev).upper()] = winner_prob_map.get(str(bottom_abbrev).upper(), 0.0) + matchup_prob * bottom_prob
            top_slot_series_win_prob += matchup_prob * top_prob
            bottom_slot_series_win_prob += matchup_prob * bottom_prob
            matchups.append({
                'matchupProb': matchup_prob,
                'topTeam': _playoff_team_payload_for_abbrev(str(top_abbrev).upper(), team_info_by_abbrev),
                'bottomTeam': _playoff_team_payload_for_abbrev(str(bottom_abbrev).upper(), team_info_by_abbrev),
                'topTeamSeriesWinProb': top_prob,
                'bottomTeamSeriesWinProb': bottom_prob,
            })

    matchups.sort(
        key=lambda row: (
            -float(row.get('matchupProb') or 0.0),
            -float(row.get('topTeamSeriesWinProb') or 0.0),
            str((((row.get('topTeam') or {}).get('abbrev')) or '')),
            str((((row.get('bottomTeam') or {}).get('abbrev')) or '')),
        )
    )
    return {
        'seriesUrl': item.get('seriesUrl'),
        'seriesTitle': item.get('seriesTitle'),
        'seriesAbbrev': item.get('seriesAbbrev'),
        'seriesLetter': item.get('seriesLetter'),
        'playoffRound': item.get('playoffRound'),
        'topSeedRank': item.get('topSeedRank'),
        'topSeedRankAbbrev': item.get('topSeedRankAbbrev'),
        'topSeedWins': int(item.get('topSeedWins') or 0),
        'bottomSeedRank': item.get('bottomSeedRank'),
        'bottomSeedRankAbbrev': item.get('bottomSeedRankAbbrev'),
        'bottomSeedWins': int(item.get('bottomSeedWins') or 0),
        'topSeedTeam': None,
        'bottomSeedTeam': None,
        'futureGames': [],
        'topSeedOutcomeProbs': {},
        'bottomSeedOutcomeProbs': {},
        'topSeedSeriesWinProb': float(top_slot_series_win_prob),
        'bottomSeedSeriesWinProb': float(bottom_slot_series_win_prob),
        'topSlotTeamProbs': _playoff_probability_rows(top_slot_probs, team_info_by_abbrev),
        'bottomSlotTeamProbs': _playoff_probability_rows(bottom_slot_probs, team_info_by_abbrev),
        'winnerProbs': _playoff_probability_rows(winner_prob_map, team_info_by_abbrev),
        'matchups': matchups[:12],
        'winnerProbMap': winner_prob_map,
        'topSlotProbMap': dict(top_slot_probs or {}),
        'bottomSlotProbMap': dict(bottom_slot_probs or {}),
        'isProjected': True,
    }


def _build_playoff_projection_payload(bracket: Dict[str, Any],
                                      lineups_all: Dict[str, Any],
                                      proj_map: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    team_info_by_abbrev: Dict[str, Dict[str, Any]] = {}
    raw_series_by_letter: Dict[str, Dict[str, Any]] = {}
    for item in (bracket.get('series') or []):
        letter = str(item.get('seriesLetter') or '').upper().strip()
        if not letter:
            continue
        raw_series_by_letter[letter] = item
        for key in ('topSeedTeam', 'bottomSeedTeam'):
            team = item.get(key) or {}
            team_abbrev = str(team.get('abbrev') or '').upper().strip()
            if team_abbrev:
                team_info_by_abbrev[team_abbrev] = _playoff_team_payload(team)

    nodes_by_letter: Dict[str, Dict[str, Any]] = {}
    current_series: List[Dict[str, Any]] = []
    for letter in _PLAYOFF_SERIES_ORDER:
        item = raw_series_by_letter.get(letter)
        if not item:
            continue
        top_team = item.get('topSeedTeam') or {}
        bottom_team = item.get('bottomSeedTeam') or {}
        has_actual_matchup = _is_resolved_playoff_team(top_team) and _is_resolved_playoff_team(bottom_team)
        if has_actual_matchup:
            node = _build_actual_playoff_series_node(item, team_info_by_abbrev, lineups_all, proj_map)
            current_series.append(node)
        else:
            deps = _PLAYOFF_SERIES_CHILDREN.get(letter)
            if not deps:
                continue
            top_source = nodes_by_letter.get(deps[0]) or {}
            bottom_source = nodes_by_letter.get(deps[1]) or {}
            node = _build_projected_playoff_series_node(
                item,
                dict(top_source.get('winnerProbMap') or {}),
                dict(bottom_source.get('winnerProbMap') or {}),
                team_info_by_abbrev,
                lineups_all,
                proj_map,
            )
        nodes_by_letter[letter] = node

    bracket_series = [nodes_by_letter[letter] for letter in _PLAYOFF_SERIES_ORDER if letter in nodes_by_letter]

    second_round_probs: Dict[str, float] = {}
    conference_final_probs: Dict[str, float] = {}
    final_probs: Dict[str, float] = {}
    cup_probs: Dict[str, float] = dict((nodes_by_letter.get('O') or {}).get('winnerProbMap') or {})

    for letter in _PLAYOFF_BRACKET_ROUNDS['eastRound2'] + _PLAYOFF_BRACKET_ROUNDS['westRound2']:
        node = nodes_by_letter.get(letter) or {}
        for side_key in ('topSlotProbMap', 'bottomSlotProbMap'):
            for team_abbrev, prob in (node.get(side_key) or {}).items():
                second_round_probs[team_abbrev] = second_round_probs.get(team_abbrev, 0.0) + float(prob or 0.0)
    for letter in ('M', 'N'):
        node = nodes_by_letter.get(letter) or {}
        for side_key in ('topSlotProbMap', 'bottomSlotProbMap'):
            for team_abbrev, prob in (node.get(side_key) or {}).items():
                conference_final_probs[team_abbrev] = conference_final_probs.get(team_abbrev, 0.0) + float(prob or 0.0)
    final_node = nodes_by_letter.get('O') or {}
    for side_key in ('topSlotProbMap', 'bottomSlotProbMap'):
        for team_abbrev, prob in (final_node.get(side_key) or {}).items():
            final_probs[team_abbrev] = final_probs.get(team_abbrev, 0.0) + float(prob or 0.0)

    all_team_abbrevs = sorted(set(team_info_by_abbrev.keys()) | set(second_round_probs.keys()) | set(conference_final_probs.keys()) | set(final_probs.keys()) | set(cup_probs.keys()))
    team_summary_rows: List[Dict[str, Any]] = []
    for team_abbrev in all_team_abbrevs:
        team_summary_rows.append({
            'team': _playoff_team_payload_for_abbrev(team_abbrev, team_info_by_abbrev),
            'secondRoundProb': float(second_round_probs.get(team_abbrev, 0.0)),
            'conferenceFinalProb': float(conference_final_probs.get(team_abbrev, 0.0)),
            'stanleyCupFinalProb': float(final_probs.get(team_abbrev, 0.0)),
            'stanleyCupProb': float(cup_probs.get(team_abbrev, 0.0)),
        })
    team_summary_rows.sort(
        key=lambda row: (
            -float(row.get('stanleyCupProb') or 0.0),
            -float(row.get('stanleyCupFinalProb') or 0.0),
            -float(row.get('conferenceFinalProb') or 0.0),
            str((((row.get('team') or {}).get('abbrev')) or '')),
        )
    )

    return {
        'series': current_series,
        'bracketSeries': bracket_series,
        'bracketRounds': {key: [nodes_by_letter[letter] for letter in letters if letter in nodes_by_letter] for key, letters in _PLAYOFF_BRACKET_ROUNDS.items()},
        'stanleyCupProbabilities': team_summary_rows,
        'assumptions': {
            'homeIcePattern': '2-2-1-1-1',
            'futureRoundsTopSlotHasHomeIce': True,
        },
    }


@main_bp.route('/api/projections/series')
def api_projections_series():
    """Return playoff series win probabilities using current lineup-based team projections."""
    year_raw = request.args.get('year')
    try:
        bracket_year = int(str(year_raw).strip()) if year_raw not in (None, '') else int(_current_playoff_bracket_year())
    except Exception:
        bracket_year = int(_current_playoff_bracket_year())

    bracket = _load_playoff_bracket_cached(bracket_year)
    if not bracket:
        return jsonify({'year': bracket_year, 'series': [], 'error': 'fetch_failed'}), 502

    lineups_all = _load_lineups_all()
    proj_map = _load_current_player_projections_cached()
    playoff_payload = _build_playoff_projection_payload(bracket, lineups_all, proj_map)

    j = jsonify({
        'year': bracket_year,
        'bracketLogo': bracket.get('bracketLogo'),
        'bracketLogoFr': bracket.get('bracketLogoFr'),
        'series': playoff_payload.get('series') or [],
        'bracketSeries': playoff_payload.get('bracketSeries') or [],
        'bracketRounds': playoff_payload.get('bracketRounds') or {},
        'stanleyCupProbabilities': playoff_payload.get('stanleyCupProbabilities') or [],
        'assumptions': playoff_payload.get('assumptions') or {},
    })
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/projections/games')
def api_projections_games():
    """Return list of games for 'today', 'yesterday', or 'tomorrow' based on Eastern Time.
    Query params:
      - which: 'today' (default) | 'yesterday' | 'tomorrow'
    """
    which = str(request.args.get('which', 'today')).lower().strip()
    # Determine ET date
    try:
        if ZoneInfo is None:
            raise RuntimeError('zoneinfo_unavailable')
        now_et = datetime.now(ZoneInfo('America/New_York'))
    except Exception:
        # Fallback to UTC if ET tz not available
        now_et = datetime.utcnow()
    if which == 'yesterday':
        date_et = (now_et - timedelta(days=1)).date()
    elif which == 'tomorrow':
        date_et = (now_et + timedelta(days=1)).date()
    else:
        date_et = now_et.date()
    date_str = date_et.isoformat()
    # Fetch schedule for ET date
    url = f'https://api-web.nhle.com/v1/schedule/{date_str}'
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return jsonify({'games': [], 'date': date_str, 'error': 'upstream_error', 'status': r.status_code}), 502
        js = r.json() or {}
    except Exception:
        return jsonify({'games': [], 'date': date_str, 'error': 'fetch_failed'}), 502

    # Build output
    def to_et(iso_utc: Optional[str]) -> Optional[str]:
        if not iso_utc:
            return None
        try:
            # Parse ISO with Z
            s = iso_utc.replace('Z', '+00:00')
            dt = datetime.fromisoformat(s)
            if ZoneInfo is not None:
                et = dt.astimezone(ZoneInfo('America/New_York'))
            else:
                et = dt  # best effort
            return et.isoformat()
        except Exception:
            return iso_utc

    logo_by_abbrev: Dict[str, str] = {}
    try:
        for tr in TEAM_ROWS:
            ab = (tr.get('Team') or '').upper()
            logo_by_abbrev[ab] = tr.get('Logo') or ''
    except Exception:
        pass

    out: List[Dict[str, Any]] = []
    # The API nests games inside gameWeek -> [ { date, games: [...] } ]
    for wk in (js.get('gameWeek') or []):
        day_date = (wk.get('date') or '')[:10]
        if day_date != date_str:
            continue
        for g in (wk.get('games') or []):
            home = (g.get('homeTeam') or {})
            away = (g.get('awayTeam') or {})
            ha = (home.get('abbrev') or '').upper()
            aa = (away.get('abbrev') or '').upper()
            out.append({
                'id': g.get('id'),
                'season': g.get('season'),
                'gameType': g.get('gameType'),
                'startTimeUTC': g.get('startTimeUTC'),
                'startTimeET': to_et(g.get('startTimeUTC')),
                'gameState': g.get('gameState') or g.get('gameStatus'),
                'venue': g.get('venue'),
                'homeTeam': { 'abbrev': ha, 'score': home.get('score'), 'logo': logo_by_abbrev.get(ha, '') },
                'awayTeam': { 'abbrev': aa, 'score': away.get('score'), 'logo': logo_by_abbrev.get(aa, '') },
                'periodDescriptor': g.get('periodDescriptor'),
            })
    # Fallback: some variants include a flat 'games' array
    if not out and isinstance(js, dict):
        for g in (js.get('games') or []):
            st = g.get('startTimeUTC') or g.get('gameDate')
            if not isinstance(st, str):
                continue
            if st.replace('Z', '').strip()[:10] != date_str:
                continue
            home = (g.get('homeTeam') or {})
            away = (g.get('awayTeam') or {})
            ha = (home.get('abbrev') or '').upper()
            aa = (away.get('abbrev') or '').upper()
            out.append({
                'id': g.get('id') or g.get('gamePk') or g.get('gameId'),
                'season': g.get('season'),
                'gameType': g.get('gameType') or g.get('gameTypeId'),
                'startTimeUTC': st,
                'startTimeET': to_et(st),
                'gameState': g.get('gameState') or g.get('gameStatus'),
                'venue': g.get('venue'),
                'homeTeam': { 'abbrev': ha, 'score': home.get('score'), 'logo': logo_by_abbrev.get(ha, '') },
                'awayTeam': { 'abbrev': aa, 'score': away.get('score'), 'logo': logo_by_abbrev.get(aa, '') },
                'periodDescriptor': g.get('periodDescriptor'),
            })
    # Compute B2B status using the previous ET date
    prev_date_et = (date_et - timedelta(days=1)).isoformat()
    prev_url = f'https://api-web.nhle.com/v1/schedule/{prev_date_et}'
    prev_set: set[str] = set()
    try:
        r2 = requests.get(prev_url, timeout=20)
        if r2.status_code == 200:
            js2 = r2.json() or {}
            for wk in (js2.get('gameWeek') or []):
                if (wk.get('date') or '')[:10] != prev_date_et:
                    continue
                for g2 in (wk.get('games') or []):
                    home2 = (g2.get('homeTeam') or {})
                    away2 = (g2.get('awayTeam') or {})
                    if home2.get('abbrev'):
                        prev_set.add(str(home2.get('abbrev')).upper())
                    if away2.get('abbrev'):
                        prev_set.add(str(away2.get('abbrev')).upper())
            if not prev_set and isinstance(js2, dict):
                for g2 in (js2.get('games') or []):
                    st2 = g2.get('startTimeUTC') or g2.get('gameDate') or ''
                    if str(st2).replace('Z','').strip()[:10] != prev_date_et:
                        continue
                    home2 = (g2.get('homeTeam') or {})
                    away2 = (g2.get('awayTeam') or {})
                    if home2.get('abbrev'):
                        prev_set.add(str(home2.get('abbrev')).upper())
                    if away2.get('abbrev'):
                        prev_set.add(str(away2.get('abbrev')).upper())
    except Exception:
        prev_set = set()

    # Load lineups and current player projections once
    lineups_all = _load_lineups_all()
    proj_map = _load_current_player_projections_cached()
    # Situation mapping values
    SITUATION = {
        'Away-B2B-B2B': -0.126602018,
        'Away-B2B-Rested': -0.400515738,
        'Away-Rested-B2B': 0.174538991,
        'Away-Rested-Rested': -0.153396566,
    }

    def situation_for(away_abbrev: str, home_abbrev: str) -> tuple[str, float, bool, bool]:
        a_b2b = (away_abbrev.upper() in prev_set)
        h_b2b = (home_abbrev.upper() in prev_set)
        if a_b2b and h_b2b:
            key = 'Away-B2B-B2B'
        elif a_b2b and not h_b2b:
            key = 'Away-B2B-Rested'
        elif (not a_b2b) and h_b2b:
            key = 'Away-Rested-B2B'
        else:
            key = 'Away-Rested-Rested'
        return key, SITUATION.get(key, 0.0), a_b2b, h_b2b

    # Compute projections per game
    for g in out:
        aa = (g.get('awayTeam') or {}).get('abbrev') or ''
        ha = (g.get('homeTeam') or {}).get('abbrev') or ''
        try:
            proj_away = _team_proj_from_lineup(str(aa), lineups_all, proj_map)
            proj_home = _team_proj_from_lineup(str(ha), lineups_all, proj_map)
            dproj = proj_away - proj_home
            key, sval, a_b2b, h_b2b = situation_for(str(aa), str(ha))
            import math
            win_away = 1.0 / (1.0 + math.exp(-(dproj) - sval))
            win_home = 1.0 - win_away
            g['b2bAway'] = bool(a_b2b)
            g['b2bHome'] = bool(h_b2b)
            g['projections'] = {
                'projAway': round(float(proj_away), 6),
                'projHome': round(float(proj_home), 6),
                'dProj': round(float(dproj), 6),
                'situationKey': key,
                'situationValue': round(float(sval), 9),
                'winProbAway': round(float(win_away), 6),
                'winProbHome': round(float(win_home), 6),
            }
        except Exception:
            # If anything fails, still return the game
            continue

    game_team_map: Dict[int, Tuple[str, str]] = {}
    for g in out:
        try:
            gid_i = int(str(g.get('id')).strip())
        except Exception:
            continue
        game_team_map[gid_i] = (
            str((g.get('awayTeam') or {}).get('abbrev') or '').strip().upper(),
            str((g.get('homeTeam') or {}).get('abbrev') or '').strip().upper(),
        )
    odds_snapshot_map = _load_latest_odds_snapshot_map(game_team_map)

    # Attach odds for not-started games only, and attach prestart for started games
    try:
        odds_map = _fetch_partner_odds_map(date_str)
    except Exception:
        odds_map = {}
    try:
        # Determine not-started strictly by comparing schedule startTimeUTC to current UTC
        from datetime import timezone as _tz
        now_utc = datetime.now(_tz.utc)
        for g in out:
            not_started = False
            started = False
            try:
                st_raw = g.get('startTimeUTC')
                if isinstance(st_raw, str):
                    se_utc = datetime.fromisoformat(st_raw.replace('Z', '+00:00'))
                    # If parsed datetime is naive, force UTC
                    if se_utc.tzinfo is None:
                        se_utc = se_utc.replace(tzinfo=_tz.utc)
                    not_started = now_utc < se_utc
                    started = now_utc >= se_utc
            except Exception:
                not_started = False
                started = False
            g['started'] = bool(started)
            gid = None
            try:
                val_id = g.get('id')
                if val_id is not None:
                    gid = int(val_id)
            except Exception:
                gid = None
            snapshot = odds_snapshot_map.get(gid) if gid is not None else None
            if not_started and gid is not None:
                if snapshot and (snapshot.get('oddsAway') is not None or snapshot.get('oddsHome') is not None):
                    g['odds'] = {'away': snapshot.get('oddsAway'), 'home': snapshot.get('oddsHome')}
                elif gid in odds_map:
                    g['odds'] = odds_map.get(gid)
            # When started, use the latest Supabase odds snapshot.
            if started and gid is not None:
                if snapshot and (snapshot.get('oddsAway') is not None or snapshot.get('oddsHome') is not None):
                    g['prestart'] = {
                        'oddsAway': snapshot.get('oddsAway'),
                        'oddsHome': snapshot.get('oddsHome'),
                        'winAwayPct': snapshot.get('winAwayPct'),
                        'winHomePct': snapshot.get('winHomePct'),
                        'betAwayPct': snapshot.get('betAwayPct'),
                        'betHomePct': snapshot.get('betHomePct'),
                    }
    except Exception:
        pass

    return jsonify({ 'date': date_str, 'timezone': 'ET', 'games': out })


@main_bp.route('/api/roster/<team_code>/current')
def api_roster_current(team_code: str):
    """Proxy NHL roster endpoint to bypass browser CORS.
    Example upstream: https://api-web.nhle.com/v1/roster/TBL/current
    """
    team = (team_code or '').upper().strip()
    if not team:
        return jsonify({'forwards': [], 'defensemen': [], 'goalies': []})
    url = f'https://api-web.nhle.com/v1/roster/{team}/current'
    try:
        r = requests.get(url, timeout=20, allow_redirects=True)
    except Exception:
        return jsonify({'forwards': [], 'defensemen': [], 'goalies': [], 'error': 'fetch_failed'}), 502
    if r.status_code != 200:
        return jsonify({'forwards': [], 'defensemen': [], 'goalies': [], 'error': 'upstream_error', 'status': r.status_code}), 502
    try:
        data = r.json()
    except Exception:
        return jsonify({'forwards': [], 'defensemen': [], 'goalies': [], 'error': 'invalid_upstream'}), 502
    # Normalize expected keys
    out = {
        'forwards': data.get('forwards') or [],
        'defensemen': data.get('defensemen') or [],
        'goalies': data.get('goalies') or [],
    }
    j = jsonify(out)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/skaters/players')
def api_skaters_players():
    """Return selectable skaters for a given team + season, or full league.

    We want *all* skaters who played for the team in that season (including traded / inactive).

    Query params:
      team=BOS
      season=20252026 (optional; defaults to current season)
      seasonState=regular|playoffs|all (optional; default regular)
      scope=team|league (optional; default team)
    """
    scope = str(request.args.get('scope') or request.args.get('playerScope') or 'team').strip().lower()
    is_league = scope in {'league', 'all', 'full'} or str(request.args.get('league') or '').strip() in {'1', 'true', 'yes'}

    team = str(request.args.get('team') or '').upper().strip()
    season = str(request.args.get('season') or '').strip()
    season_state = str(request.args.get('seasonState') or request.args.get('season_state') or 'regular').strip().lower()
    if (not is_league) and (not team):
        return jsonify({'players': []})

    # Prefer NHL stats skater bios (seasonId + currentTeamAbbrev) for any season.
    # Fallback for older seasons: roster/{team}/{season} (historical roster by season).
    # NOTE: skater bios endpoint returns 500 if cayenneExp is omitted.
    season_i = _safe_int(season)
    try:
        current_i = int(current_season_id())
    except Exception:
        current_i = 0
    season_ids = _parse_request_season_ids(season, default=current_i)
    season_i = _primary_season_id(season_ids, default=current_i) or current_i

    players: List[Dict[str, Any]] = []

    # Primary source: NHL stats skater summary filtered by teamAbbrev.
    # This includes players who played for the team that season (including multi-team "teamAbbrevs").
    # Docs/shape: https://api.nhle.com/stats/rest/en/skater/summary
    try:
        players_ttl_s = max(60, int(os.getenv('SKATERS_PLAYERS_CACHE_TTL_SECONDS', '21600') or '21600'))
    except Exception:
        players_ttl_s = 21600

    try:
        players_cache_max = max(1, int(os.getenv('SKATERS_PLAYERS_CACHE_MAX_ITEMS', '12') or '12'))
    except Exception:
        players_cache_max = 12

    if season_state not in {'regular', 'playoffs', 'all'}:
        season_state = 'regular'

    cache_key = (tuple(season_ids), '__LEAGUE__' if is_league else team, season_state)
    now = time.time()
    try:
        _cache_prune_ttl_and_size(_SKATERS_PLAYERS_CACHE, ttl_s=players_ttl_s, max_items=players_cache_max)
        cached = _SKATERS_PLAYERS_CACHE.get(cache_key)
        if cached and (now - float(cached[0])) < float(players_ttl_s):
            players = cached[1] or []
            j = jsonify({'players': players})
            try:
                j.headers['Cache-Control'] = 'no-store'
            except Exception:
                pass
            return j
    except Exception:
        pass

    if not players:
        try:
            players_by_pid: Dict[int, Dict[str, Any]] = {}
            url = 'https://api.nhle.com/stats/rest/en/skater/summary'
            for season_id in season_ids:
                if is_league:
                    if season_state == 'regular':
                        cay = f'seasonId={int(season_id)} and gameTypeId=2'
                    elif season_state == 'playoffs':
                        cay = f'seasonId={int(season_id)} and gameTypeId=3'
                    else:
                        cay = f'seasonId={int(season_id)} and (gameTypeId=2 or gameTypeId=3)'
                else:
                    if season_state == 'regular':
                        cay = f'seasonId={int(season_id)} and gameTypeId=2 and teamAbbrev="{team}"'
                    elif season_state == 'playoffs':
                        cay = f'seasonId={int(season_id)} and gameTypeId=3 and teamAbbrev="{team}"'
                    else:
                        cay = f'seasonId={int(season_id)} and (gameTypeId=2 or gameTypeId=3) and teamAbbrev="{team}"'

                r = requests.get(
                    url,
                    params={'limit': -1, 'start': 0, 'cayenneExp': cay},
                    headers={'User-Agent': 'Mozilla/5.0'},
                    timeout=25,
                    allow_redirects=True,
                )
                if r.status_code != 200:
                    continue
                data = r.json() if r.content else {}
                rows = data.get('data') if isinstance(data, dict) else None
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    pid = _safe_int(row.get('playerId'))
                    if not pid or pid <= 0:
                        continue
                    name = str(row.get('skaterFullName') or '').strip() or str(pid)
                    pos = str(row.get('positionCode') or '').strip().upper()
                    if pos.startswith('G'):
                        continue
                    team_raw = row.get('teamAbbrev') or row.get('teamAbbrevs') or row.get('currentTeamAbbrev') or ''
                    team_abbrev = str(team_raw or '').strip().upper()
                    if '/' in team_abbrev:
                        team_abbrev = team_abbrev.split('/')[0].strip().upper()
                    rec: Dict[str, Any] = {'playerId': int(pid), 'name': name, 'pos': pos}
                    if team_abbrev:
                        rec['team'] = team_abbrev
                    players_by_pid[int(pid)] = rec
            players = list(players_by_pid.values())
        except Exception:
            players = []

        try:
            # De-dupe by playerId
            seen: set[int] = set()
            uniq: List[Dict[str, Any]] = []
            for p in players:
                try:
                    pid_i = int(p.get('playerId') or 0)
                    if pid_i <= 0 or pid_i in seen:
                        continue
                    seen.add(pid_i)
                    uniq.append(p)
                except Exception:
                    continue
            players = uniq
        except Exception:
            pass

        try:
            _cache_set_multi_bounded(_SKATERS_PLAYERS_CACHE, cache_key, players, ttl_s=players_ttl_s, max_items=players_cache_max)
        except Exception:
            pass

    # Fallbacks (best-effort) if stats summary fails.
    if (not players) and (not is_league):
        bios_map = _load_skater_bios_season_cached(int(season_i or 0))
        try:
            for pid, info in (bios_map or {}).items():
                try:
                    if not pid:
                        continue
                    t = str((info or {}).get('team') or '').strip().upper()
                    if t != team:
                        continue
                    name = str((info or {}).get('name') or '').strip() or str(pid)
                    pos_code = str((info or {}).get('positionCode') or (info or {}).get('position') or '').strip().upper()
                    players.append({'playerId': int(pid), 'name': name, 'pos': pos_code})
                except Exception:
                    continue
        except Exception:
            players = []

    if (not players) and (not is_league) and season_i and current_i and season_i != current_i:
        url = f'https://api-web.nhle.com/v1/roster/{team.lower()}/{season_i}'
        try:
            r = requests.get(url, timeout=20, allow_redirects=True)
            if r.status_code == 200:
                data = r.json() if r.content else {}
                if isinstance(data, dict):
                    forwards = data.get('forwards') or []
                    defensemen = data.get('defensemen') or []
                    for p in list(forwards) + list(defensemen):
                        if not isinstance(p, dict):
                            continue
                        pid = _safe_int(p.get('id') or p.get('playerId'))
                        if not pid or pid <= 0:
                            continue
                        fn = (p.get('firstName') or {}).get('default') if isinstance(p.get('firstName'), dict) else (p.get('firstName') or '')
                        ln = (p.get('lastName') or {}).get('default') if isinstance(p.get('lastName'), dict) else (p.get('lastName') or '')
                        pos = str(p.get('positionCode') or p.get('position') or '').strip().upper()
                        name = (str(fn).strip() + ' ' + str(ln).strip()).strip() or str(pid)
                        players.append({'playerId': int(pid), 'name': name, 'pos': pos})
        except Exception:
            pass

    j = jsonify({'players': players})
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/goalies/players')
def api_goalies_players():
    """Return selectable goalies for a given team + season, or full league.

    Query params:
      team=BOS
      season=20252026 (optional; defaults to current season)
      seasonState=regular|playoffs|all (optional; default regular)
      scope=team|league (optional; default team)
    """
    scope = str(request.args.get('scope') or request.args.get('playerScope') or 'team').strip().lower()
    is_league = scope in {'league', 'all', 'full'} or str(request.args.get('league') or '').strip() in {'1', 'true', 'yes'}

    team = str(request.args.get('team') or '').upper().strip()
    season = str(request.args.get('season') or '').strip()
    season_state = str(request.args.get('seasonState') or request.args.get('season_state') or 'regular').strip().lower()
    if (not is_league) and (not team):
        return jsonify({'players': []})

    season_i = _safe_int(season)
    try:
        current_i = int(current_season_id())
    except Exception:
        current_i = 0
    season_ids = _parse_request_season_ids(season, default=current_i)
    season_i = _primary_season_id(season_ids, default=current_i) or current_i

    if season_state not in {'regular', 'playoffs', 'all'}:
        season_state = 'regular'

    try:
        players_ttl_s = max(60, int(os.getenv('GOALIES_PLAYERS_CACHE_TTL_SECONDS', '21600') or '21600'))
    except Exception:
        players_ttl_s = 21600

    try:
        players_cache_max = max(1, int(os.getenv('GOALIES_PLAYERS_CACHE_MAX_ITEMS', '12') or '12'))
    except Exception:
        players_cache_max = 12

    cache_key = (tuple(season_ids), '__LEAGUE__' if is_league else team, season_state)
    # Cache hit: return immediately (avoids extra allocations/work)
    if True:
        try:
            _cache_prune_ttl_and_size(_GOALIES_PLAYERS_CACHE, ttl_s=players_ttl_s, max_items=players_cache_max)
            cached_players = _cache_get(_GOALIES_PLAYERS_CACHE, cache_key, int(players_ttl_s))
            if cached_players is not None:
                j = jsonify({'players': cached_players})
                try:
                    j.headers['Cache-Control'] = 'no-store'
                except Exception:
                    pass
                return j
        except Exception:
            pass

    players: List[Dict[str, Any]] = []
    now = time.time()
    if not players:
        try:
            players_by_pid: Dict[int, Dict[str, Any]] = {}
            url = 'https://api.nhle.com/stats/rest/en/goalie/summary'
            for season_id in season_ids:
                if is_league:
                    if season_state == 'regular':
                        cay = f'seasonId={int(season_id)} and gameTypeId=2'
                    elif season_state == 'playoffs':
                        cay = f'seasonId={int(season_id)} and gameTypeId=3'
                    else:
                        cay = f'seasonId={int(season_id)} and (gameTypeId=2 or gameTypeId=3)'
                else:
                    if season_state == 'regular':
                        cay = f'seasonId={int(season_id)} and gameTypeId=2 and teamAbbrev="{team}"'
                    elif season_state == 'playoffs':
                        cay = f'seasonId={int(season_id)} and gameTypeId=3 and teamAbbrev="{team}"'
                    else:
                        cay = f'seasonId={int(season_id)} and (gameTypeId=2 or gameTypeId=3) and teamAbbrev="{team}"'

                r = requests.get(
                    url,
                    params={'limit': -1, 'start': 0, 'cayenneExp': cay},
                    headers={'User-Agent': 'Mozilla/5.0'},
                    timeout=25,
                    allow_redirects=True,
                )
                if r.status_code != 200:
                    continue
                data = r.json() if r.content else {}
                rows = data.get('data') if isinstance(data, dict) else None
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    pid = _safe_int(row.get('playerId') or row.get('goalieId') or row.get('id'))
                    if not pid or pid <= 0:
                        continue
                    name = str(row.get('goalieFullName') or row.get('playerFullName') or row.get('skaterFullName') or '').strip() or str(pid)
                    team_raw = row.get('teamAbbrev') or row.get('teamAbbrevs') or row.get('currentTeamAbbrev') or ''
                    team_abbrev = str(team_raw or '').strip().upper()
                    if '/' in team_abbrev:
                        team_abbrev = team_abbrev.split('/')[0].strip().upper()
                    rec: Dict[str, Any] = {'playerId': int(pid), 'name': name, 'pos': 'G'}
                    if team_abbrev:
                        rec['team'] = team_abbrev
                    players_by_pid[int(pid)] = rec
            players = list(players_by_pid.values())
        except Exception:
            players = []

        # Fallback for older seasons (team roster endpoint) for team-scoped queries.
        if (not players) and (not is_league) and season_i and current_i and season_i != current_i:
            url = f'https://api-web.nhle.com/v1/roster/{team.lower()}/{season_i}'
            try:
                r = requests.get(url, timeout=20, allow_redirects=True)
                if r.status_code == 200:
                    data = r.json() if r.content else {}
                    if isinstance(data, dict):
                        goalies = data.get('goalies') or []
                        for p in list(goalies):
                            if not isinstance(p, dict):
                                continue
                            pid = _safe_int(p.get('id') or p.get('playerId'))
                            if not pid or pid <= 0:
                                continue
                            fn = (p.get('firstName') or {}).get('default') if isinstance(p.get('firstName'), dict) else (p.get('firstName') or '')
                            ln = (p.get('lastName') or {}).get('default') if isinstance(p.get('lastName'), dict) else (p.get('lastName') or '')
                            name = (str(fn).strip() + ' ' + str(ln).strip()).strip() or str(pid)
                            players.append({'playerId': int(pid), 'name': name, 'pos': 'G', 'team': team})
            except Exception:
                pass

        try:
            seen: set[int] = set()
            uniq: List[Dict[str, Any]] = []
            for p in players:
                try:
                    pid_i = int(p.get('playerId') or 0)
                    if pid_i <= 0 or pid_i in seen:
                        continue
                    seen.add(pid_i)
                    uniq.append(p)
                except Exception:
                    continue
            players = uniq
        except Exception:
            pass

        try:
            _cache_set_multi_bounded(_GOALIES_PLAYERS_CACHE, cache_key, players, ttl_s=players_ttl_s, max_items=players_cache_max)
        except Exception:
            pass

    j = jsonify({'players': players})
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/player/<int:player_id>/landing')
def api_player_landing(player_id: int):
    """Proxy NHL player landing endpoint to bypass browser CORS."""
    pid = int(player_id)
    if pid <= 0:
        return jsonify({'error': 'invalid_player_id'}), 400

    try:
        ttl_s = max(10, int(os.getenv('PLAYER_LANDING_CACHE_TTL_SECONDS', '3600') or '3600'))
    except Exception:
        ttl_s = 3600

    try:
        max_items = max(1, int(os.getenv('PLAYER_LANDING_CACHE_MAX_ITEMS', '512') or '512'))
    except Exception:
        max_items = 512

    _cache_prune_ttl_and_size(_PLAYER_LANDING_CACHE, ttl_s=ttl_s, max_items=max_items)

    cached = _cache_get(_PLAYER_LANDING_CACHE, pid, ttl_s)
    if cached is not None:
        j = jsonify(cached)
        try:
            j.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
        return j

    url = f'https://api-web.nhle.com/v1/player/{pid}/landing'
    try:
        r = requests.get(url, timeout=20, allow_redirects=True)
    except Exception:
        return jsonify({'error': 'fetch_failed'}), 502
    if r.status_code != 200:
        return jsonify({'error': 'upstream_error', 'status': r.status_code}), 502
    try:
        data = r.json()
    except Exception:
        return jsonify({'error': 'invalid_upstream'}), 502
    if not isinstance(data, dict):
        return jsonify({'error': 'invalid_upstream'}), 502

    _cache_set_multi_bounded(_PLAYER_LANDING_CACHE, pid, data, ttl_s=ttl_s, max_items=max_items)
    j = jsonify(data)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/player-projections/<int:player_id>')
def api_player_projections(player_id: int):
    """Return a single projections row for a playerId (Sheets3 preferred; CSV fallback)."""
    pid = int(player_id)
    if pid <= 0:
        return jsonify({'error': 'invalid_player_id'}), 400
    proj_map = _load_player_projections_cached()
    row = proj_map.get(pid)
    if not row:
        return jsonify({'error': 'not_found'}), 404
    j = jsonify({'playerId': pid, 'row': row})
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/rapm/player/<int:player_id>')
def api_rapm_player(player_id: int):
    """Return RAPM rows from app/static/rapm/rapm.csv for a player.

    Optional query params:
      season=20252026
    """
    pid = int(player_id)
    if pid <= 0:
        return jsonify({'rows': [], 'error': 'invalid_player_id'}), 400
    season = str(request.args.get('season') or '').strip()
    season_ids = _parse_request_season_ids(season)
    season_set = set(_normalize_season_id_list(season_ids))
    single_season = next(iter(season_set)) if len(season_set) == 1 else None

    rows: List[Dict[str, Any]]
    source = 'supabase'
    # Supabase-backed loaders handle all seasons.
    # Falls back to CSV internally if Supabase is unavailable.
    rows = _load_rapm_player_rows_static(pid, single_season)
    out: List[Dict[str, Any]] = []
    for r in rows:
        try:
            if str(r.get('PlayerID') or '').strip() != str(pid):
                continue
            if season_set and not _row_season_in_selected(r, season_ids):
                continue
            # Keep only a subset needed by the Skaters RAPM tab
            out.append({
                'PlayerID': pid,
                'Season': r.get('Season'),
                'StrengthState': r.get('StrengthState'),
                'Rates_Totals': r.get('Rates_Totals') or r.get('Rates/Totals') or r.get('RatesTotals'),

                'CF': r.get('CF'),
                'CA': r.get('CA'),
                'GF': r.get('GF'),
                'GA': r.get('GA'),
                'xGF': r.get('xGF'),
                'xGA': r.get('xGA'),
                'C_plusminus': r.get('C_plusminus'),
                'G_plusminus': r.get('G_plusminus'),
                'xG_plusminus': r.get('xG_plusminus'),

                'CF_zscore': r.get('CF_zscore'),
                'CA_zscore': r.get('CA_zscore'),
                'GF_zscore': r.get('GF_zscore'),
                'GA_zscore': r.get('GA_zscore'),
                'xGF_zscore': r.get('xGF_zscore'),
                'xGA_zscore': r.get('xGA_zscore'),
                'C_plusminus_zscore': r.get('C_plusminus_zscore'),
                'G_plusminus_zscore': r.get('G_plusminus_zscore'),
                'xG_plusminus_zscore': r.get('xG_plusminus_zscore'),

                'PP_CF': r.get('PP_CF'),
                'PP_GF': r.get('PP_GF'),
                'PP_xGF': r.get('PP_xGF'),
                'PP_CF_zscore': r.get('PP_CF_zscore'),
                'PP_GF_zscore': r.get('PP_GF_zscore'),
                'PP_xGF_zscore': r.get('PP_xGF_zscore'),

                'SH_CA': r.get('SH_CA'),
                'SH_GA': r.get('SH_GA'),
                'SH_xGA': r.get('SH_xGA'),
                'SH_CA_zscore': r.get('SH_CA_zscore'),
                'SH_GA_zscore': r.get('SH_GA_zscore'),
                'SH_xGA_zscore': r.get('SH_xGA_zscore'),
            })
        except Exception:
            continue

    # Stable ordering
    order = {'5v5': 0, 'PP': 1, 'SH': 2}
    out.sort(key=lambda x: (int(x.get('Season') or 0), order.get(str(x.get('StrengthState') or ''), 99)))
    j = jsonify({'playerId': pid, 'rows': out, 'source': source})
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/context/player/<int:player_id>')
def api_context_player(player_id: int):
    """Return context rows from Supabase / app/static/rapm/context.csv for a player.

    Optional query params:
      season=20252026
    """
    pid = int(player_id)
    if pid <= 0:
        return jsonify({'rows': [], 'error': 'invalid_player_id'}), 400
    season = str(request.args.get('season') or '').strip()
    season_ids = _parse_request_season_ids(season)
    season_set = set(_normalize_season_id_list(season_ids))
    single_season = next(iter(season_set)) if len(season_set) == 1 else None

    rows: List[Dict[str, Any]]
    source = 'supabase'
    # Supabase-backed loaders handle all seasons.
    rows = _load_context_player_rows_static(pid, single_season)

    out: List[Dict[str, Any]] = []
    for r in rows:
        try:
            if str(r.get('PlayerID') or '').strip() != str(pid):
                continue
            if season_set and not _row_season_in_selected(r, season_ids):
                continue
            out.append({
                'PlayerID': pid,
                'Season': r.get('Season'),
                'StrengthState': r.get('StrengthState'),
                'Minutes': r.get('Minutes'),
                'QoT_blend_xG67_G33': r.get('QoT_blend_xG67_G33'),
                'QoC_blend_xG67_G33': r.get('QoC_blend_xG67_G33'),
                'ZS_Difficulty': r.get('ZS_Difficulty'),
            })
        except Exception:
            continue

    order = {'5v5': 0, 'PP': 1, 'SH': 2}
    out.sort(key=lambda x: (int(x.get('Season') or 0), order.get(str(x.get('StrengthState') or ''), 99)))
    j = jsonify({'playerId': pid, 'rows': out, 'source': source})
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/skaters/card')
def api_skaters_card():
    """Player card metrics + league percentiles from SeasonStats (Sheets6).

    Query params:
      season=20252026
      playerId=<int>
      seasonState=regular|playoffs|all
      strengthState=5v5|PP|SH|Other|all
      xgModel=xG_S|xG_F|xG_F2
      rates=Totals|Per60|PerGame
      metricIds=<comma-separated Category|Metric ids>
      scope=season|career
      minGP=<int>
      minTOI=<float minutes>
    """
    season = str(request.args.get('season') or '').strip()
    player_id_q = str(request.args.get('playerId') or request.args.get('player_id') or '').strip()
    season_state = str(request.args.get('seasonState') or 'regular').strip().lower()
    strength_state = str(request.args.get('strengthState') or '5v5').strip()
    xg_model = str(request.args.get('xgModel') or 'xG_F').strip()
    rates = str(request.args.get('rates') or request.args.get('ratesTotals') or 'Totals').strip() or 'Totals'
    metric_ids_raw = str(request.args.get('metricIds') or request.args.get('metrics') or '').strip()

    scope = str(request.args.get('scope') or 'season').strip().lower()
    min_gp = _safe_int(request.args.get('minGP') or request.args.get('minGp') or request.args.get('min_gp') or 0) or 0
    min_toi_raw = request.args.get('minTOI') or request.args.get('minToi') or request.args.get('min_toi') or 0
    try:
        min_toi = float(_parse_locale_float(min_toi_raw) or 0.0)
    except Exception:
        min_toi = 0.0
    if min_gp < 0:
        min_gp = 0
    if min_toi < 0:
        min_toi = 0.0

    pid = _safe_int(player_id_q)
    if not pid or pid <= 0:
        return jsonify({'error': 'missing_playerId'}), 400

    season_ids = _parse_request_season_ids(season, default=20252026)
    season_int = _primary_season_id(season_ids, default=20252026) or 20252026

    if season_state not in {'regular', 'playoffs', 'all'}:
        season_state = 'regular'
    if strength_state not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        strength_state = '5v5'
    if xg_model not in {'xG_S', 'xG_F', 'xG_F2'}:
        xg_model = 'xG_F'
    if rates not in {'Totals', 'Per60', 'PerGame'}:
        rates = 'Totals'
    if scope not in {'season', 'career'}:
        scope = 'season'

    metric_ids: List[str] = []
    if metric_ids_raw:
        metric_ids = [s.strip() for s in metric_ids_raw.split(',') if s and s.strip()]

    # Aggregate by player under the requested filters (cached).
    agg, pos_group_by_pid = _build_seasonstats_agg(
        scope=scope,
        season_int=season_int,
        season_ids=season_ids,
        season_state=season_state,
        strength_state=strength_state,
    )

    # Apply minimum requirements (affects both the returned player and percentile pools).
    if min_gp > 0 or min_toi > 0:
        eligible = {pid_k for pid_k, d in agg.items() if float(d.get('GP') or 0) >= float(min_gp) and float(d.get('TOI') or 0.0) >= float(min_toi)}
        agg = {pid_k: d for pid_k, d in agg.items() if pid_k in eligible}
        pos_group_by_pid = {pid_k: g for pid_k, g in pos_group_by_pid.items() if pid_k in eligible}

    def _pct(n: Optional[float], d: Optional[float]) -> Optional[float]:
        try:
            if n is None or d is None:
                return None
            if d <= 0:
                return None
            return 100.0 * (float(n) / float(d))
        except Exception:
            return None

    # Choose attempts + xG based on the selected xG model.
    def _attempts(v: Dict[str, Any]) -> float:
        vv = v.get('iShots') if xg_model == 'xG_S' else v.get('iFenwick')
        return float(vv or 0.0)

    def _ixg(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('ixG_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('ixG_F2') or 0.0)
        return float(v.get('ixG_S') or 0.0)

    def _xgf(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('xGF_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('xGF_F2') or 0.0)
        return float(v.get('xGF_S') or 0.0)

    def _xga(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('xGA_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('xGA_F2') or 0.0)
        return float(v.get('xGA_S') or 0.0)

    def _rate_from(gp: float, toi: float, vv: Optional[float]) -> Optional[float]:
        if rates == 'Totals':
            return vv
        denom = None
        if rates == 'PerGame':
            denom = gp if gp > 0 else None
        elif rates == 'Per60':
            denom = (toi / 60.0) if toi > 0 else None
        if vv is None or denom is None or denom <= 0:
            return None
        try:
            return float(vv) / float(denom)
        except Exception:
            return None

    # Build per-player computed metrics used by cards.
    # Keys are metric IDs: "Category|Metric".
    defs = _load_card_metrics_defs()
    def_map: Dict[str, Dict[str, Any]] = {str(m.get('id')): m for m in (defs.get('metrics') or []) if isinstance(m, dict) and m.get('id')}

    # Backwards-compatible (old UI) mode
    if not metric_ids:
        # Old keys list
        metric_ids = [
            'Ice Time|GP',
            'Ice Time|TOI',
            'Production|iGoals',
            'Production|Assists1',
            'Production|Assists2',
            'Production|Points',
            'Shooting|ixG',
            'Shooting|Sh% or FSh%',
            'Shooting|xSh% or xFS%',
            'Shooting|dSh% or dFSh%',
        ]

    def _norm_rates_totals(v: Any) -> str:
        s = str(v or '').strip().lower()
        if s.startswith('tot'):
            return 'Totals'
        if s.startswith('rate'):
            return 'Rates'
        return str(v or '').strip() or 'Rates'

    want_strength = strength_state if strength_state in {'5v5', 'PP', 'SH'} else '5v5'
    want_rapm_rates = 'Totals' if rates == 'Totals' else 'Rates'

    # For some external metrics (RAPM/context z-scores) we can derive percentiles directly.
    special_pct: Dict[str, Optional[float]] = {}

    def _z_to_pct(z: Optional[float]) -> Optional[float]:
        if z is None:
            return None
        try:
            zz = float(z)
            if not math.isfinite(zz):
                return None
            # Normal CDF via erf
            return 50.0 * (1.0 + math.erf(zz / math.sqrt(2.0)))
        except Exception:
            return None

    def _lower_is_better(metric_id: str) -> bool:
        m = metric_id
        if '|' in metric_id:
            _, m = metric_id.split('|', 1)
        m = str(m or '').strip()
        return m in {
            'CA', 'FA', 'SA', 'GA', 'xGA',
            'PIM_taken', 'PIM_Against',
            'Giveaways',
            'RAPM CA', 'RAPM GA', 'RAPM xGA',
        }

    # Load RAPM/context rows only if requested.
    rapm_row: Optional[Dict[str, Any]] = None
    ctx_row: Optional[Dict[str, Any]] = None
    needs_rapm = any(('|RAPM ' in mid) for mid in metric_ids)
    needs_ctx = any((mid in {'Context|QoT', 'Context|QoC', 'Context|ZS'}) for mid in metric_ids)
    needs_projection = any(mid.startswith('Projection|') for mid in metric_ids)
    projection_map: Dict[int, Dict[str, Any]] = {}

    if needs_rapm:
        rapm_rows = _load_rapm_player_rows_static(int(pid), season_int)

        # Filter to the requested player/season and pick the requested strength+rates.
        candidates: List[Dict[str, Any]] = []
        for r in rapm_rows:
            try:
                if str(r.get('PlayerID') or '').strip() != str(pid):
                    continue
                if season_int is not None:
                    try:
                        if int(str(r.get('Season') or '').strip()) != season_int:
                            continue
                    except Exception:
                        continue
                candidates.append(r)
            except Exception:
                continue
        for r in candidates:
            if str(r.get('StrengthState') or '').strip() == want_strength and _norm_rates_totals(r.get('Rates_Totals') or r.get('Rates/Totals') or r.get('RatesTotals')) == want_rapm_rates:
                rapm_row = r
                break
        if rapm_row is None:
            for r in candidates:
                if _norm_rates_totals(r.get('Rates_Totals') or r.get('Rates/Totals') or r.get('RatesTotals')) == want_rapm_rates:
                    rapm_row = r
                    break
        if rapm_row is None and candidates:
            rapm_row = candidates[0]

    if needs_ctx:
        ctx_rows = _load_context_player_rows_static(int(pid), season_int)

        candidates2: List[Dict[str, Any]] = []
        for r in ctx_rows:
            try:
                if str(r.get('PlayerID') or '').strip() != str(pid):
                    continue
                if season_int is not None:
                    try:
                        if int(str(r.get('Season') or '').strip()) != season_int:
                            continue
                    except Exception:
                        continue
                candidates2.append(r)
            except Exception:
                continue
        for r in candidates2:
            if str(r.get('StrengthState') or '').strip() == want_strength:
                ctx_row = r
                break
        if ctx_row is None:
            for r in candidates2:
                if str(r.get('StrengthState') or '').strip() == '5v5':
                    ctx_row = r
                    break
        if ctx_row is None and candidates2:
            ctx_row = candidates2[0]

    if needs_projection:
        try:
            projection_map = _load_current_player_projections_cached() or {}
        except Exception:
            projection_map = {}

    def _projection_value_for_player(player_id: int) -> Optional[float]:
        row = None
        if isinstance(projection_map, dict):
            pid_int = int(player_id)
            pid_str = str(pid_int)
            row = projection_map.get(pid_int)
            if not isinstance(row, dict):
                row = projection_map.get(pid_str)
            if not isinstance(row, dict):
                for k, v in projection_map.items():
                    try:
                        if int(str(k).strip()) == pid_int and isinstance(v, dict):
                            row = v
                            break
                    except Exception:
                        continue
        if not isinstance(row, dict):
            return None
        for key in ('projected_value', 'projection', 'projectedValue', 'Projection', 'ProjectedValue'):
            val = _parse_locale_float(row.get(key))
            if val is not None:
                return float(val)
        return None

    def _projection_pos_group_for_player(player_id: int) -> str:
        row = None
        if isinstance(projection_map, dict):
            pid_int = int(player_id)
            pid_str = str(pid_int)
            row = projection_map.get(pid_int)
            if not isinstance(row, dict):
                row = projection_map.get(pid_str)
        if not isinstance(row, dict):
            return ''
        return _projection_position_group(row.get('position'))

    def _projection_pool_for_group(group: str) -> List[float]:
        if not isinstance(projection_map, dict) or not projection_map:
            return []
        g = str(group or '').strip().upper()
        out_vals: List[float] = []
        for raw in projection_map.values():
            if not isinstance(raw, dict):
                continue
            try:
                pg = _projection_position_group(raw.get('position'))
            except Exception:
                pg = ''
            if g in {'F', 'D'} and pg != g:
                continue
            v = None
            for key in ('projected_value', 'projection', 'projectedValue', 'Projection', 'ProjectedValue'):
                v = _parse_locale_float(raw.get(key))
                if v is not None:
                    break
            if v is None:
                continue
            try:
                fv = float(v)
                if math.isfinite(fv):
                    out_vals.append(fv)
            except Exception:
                continue
        out_vals.sort()
        return out_vals

    def _compute_metric(metric_id: str, v: Dict[str, Any], player_id: int) -> Optional[float]:
        # SeasonStats base vars
        gp = float(v.get('GP') or 0.0)
        toi = float(v.get('TOI') or 0.0)
        igoals = float(v.get('iGoals') or 0.0)
        a1 = float(v.get('Assists1') or 0.0)
        a2 = float(v.get('Assists2') or 0.0)
        pts = igoals + a1 + a2
        att = _attempts(v)
        ixg = _ixg(v)

        cf = float(v.get('CF') or 0.0)
        ca = float(v.get('CA') or 0.0)
        ff = float(v.get('FF') or 0.0)
        fa = float(v.get('FA') or 0.0)
        sf = float(v.get('SF') or 0.0)
        sa = float(v.get('SA') or 0.0)
        gf = float(v.get('GF') or 0.0)
        ga = float(v.get('GA') or 0.0)
        xgf = _xgf(v)
        xga = _xga(v)

        pim_taken = float(v.get('PIM_taken') or 0.0)
        pim_drawn = float(v.get('PIM_drawn') or 0.0)
        pim_for = float(v.get('PIM_for') or 0.0)
        pim_against = float(v.get('PIM_against') or 0.0)
        hits = float(v.get('Hits') or 0.0)
        takeaways = float(v.get('Takeaways') or 0.0)
        giveaways = float(v.get('Giveaways') or 0.0)

        category = None
        metric = None
        if '|' in metric_id:
            category, metric = metric_id.split('|', 1)
        else:
            metric = metric_id

        if category == 'Projection' and metric in {'Percentile', 'Projection Percentile', 'Projection'}:
            return _projection_value_for_player(int(player_id))

        # NHL Edge metrics (value + percentile come from the NHL Edge API; don't compute percentiles locally).
        if category == 'Edge':
            if int(player_id) != int(pid):
                return None
            if season_int is not None and int(season_int) < 20212022:
                special_pct[metric_id] = None
                return None
            mdef = def_map.get(metric_id) or {}
            link = str(mdef.get('link') or '').strip()
            if not link:
                special_pct[metric_id] = None
                return None

            game_type = _edge_game_type(season_state)
            url = _edge_format_url(link, int(pid), int(season_int or 0), int(game_type))
            if not url:
                special_pct[metric_id] = None
                return None
            payload_edge = _edge_get_cached_json(url)
            if not payload_edge:
                special_pct[metric_id] = None
                return None

            strength_code = None
            if str(mdef.get('strengthCode') or '').strip().lower() == 'strengthcode':
                strength_code = _edge_strength_code(strength_state)

            # Distance Skated: alternate total/per60; PerGame uses distanceTotal/GP but keeps distanceTotal percentile.
            if metric == 'distanceTotal or distancePer60':
                total_val, total_pct = _edge_extract_value_and_pct(payload_edge, 'distanceTotal', strength_code)
                per60_val, per60_pct = _edge_extract_value_and_pct(payload_edge, 'distancePer60', strength_code)
                if rates == 'Per60':
                    if per60_pct is not None:
                        special_pct[metric_id] = per60_pct
                    return per60_val
                if rates == 'PerGame':
                    if total_pct is not None:
                        special_pct[metric_id] = total_pct
                    if total_val is None or gp <= 0:
                        return None
                    return float(total_val) / float(gp)
                # Totals
                if total_pct is not None:
                    special_pct[metric_id] = total_pct
                return total_val

            # Default Edge extraction
            val_e, pct_e = _edge_extract_value_and_pct(payload_edge, str(metric or ''), strength_code)
            if pct_e is not None:
                special_pct[metric_id] = pct_e
            else:
                special_pct[metric_id] = None

            # Percent-of-time fields (zone time) come back as 0..1 fractions.
            try:
                if metric and str(metric).lower().endswith('pctg') and val_e is not None:
                    fv = float(val_e)
                    if 0.0 <= fv <= 1.5:
                        val_e = 100.0 * fv
            except Exception:
                pass
            return val_e

        # External metrics (RAPM/context) for requested player only.
        if player_id == int(pid):
            if metric and str(metric).startswith('RAPM '):
                if not rapm_row:
                    return None
                base = str(metric).replace('RAPM', '', 1).strip()
                # Map to columns + zscore columns
                col = None
                zcol = None
                if base in {'CF', 'CA', 'GF', 'GA', 'xGF', 'xGA'}:
                    col = base
                    zcol = f'{base}_zscore'
                elif base == 'C+/-':
                    col = 'C_plusminus'
                    zcol = 'C_plusminus_zscore'
                elif base == 'G+/-':
                    col = 'G_plusminus'
                    zcol = 'G_plusminus_zscore'
                elif base == 'xG+/-':
                    col = 'xG_plusminus'
                    zcol = 'xG_plusminus_zscore'
                if not col:
                    return None
                val = _parse_locale_float(rapm_row.get(col))
                z = _parse_locale_float(rapm_row.get(zcol)) if zcol else None
                pct = _z_to_pct(z)
                if pct is not None and _lower_is_better(metric_id):
                    pct = 100.0 - pct
                special_pct[metric_id] = pct
                return float(val) if val is not None else None

            if category == 'Context' and metric in {'QoT', 'QoC', 'ZS'}:
                if not ctx_row:
                    return None
                col2 = None
                if metric == 'QoT':
                    col2 = 'QoT_blend_xG67_G33'
                elif metric == 'QoC':
                    col2 = 'QoC_blend_xG67_G33'
                elif metric == 'ZS':
                    col2 = 'ZS_Difficulty'
                val2 = _parse_locale_float(ctx_row.get(col2)) if col2 else None
                pct2 = _z_to_pct(float(val2)) if val2 is not None else None
                special_pct[metric_id] = pct2
                return float(val2) if val2 is not None else None

        # Map common seasonstats metrics
        if metric == 'GP':
            return gp
        if metric == 'TOI':
            return toi

        if metric == 'iGoals':
            return _rate_from(gp, toi, igoals)
        if metric == 'Assists1':
            return _rate_from(gp, toi, a1)
        if metric == 'Assists2':
            return _rate_from(gp, toi, a2)
        if metric == 'Points':
            return _rate_from(gp, toi, pts)

        if metric in {'iShots', 'iFenwick', 'iShots or iFenwick'}:
            vv = float(v.get('iShots') or 0.0) if xg_model == 'xG_S' else float(v.get('iFenwick') or 0.0)
            return _rate_from(gp, toi, vv)

        if metric in {'ixG', 'Individual xG'}:
            return _rate_from(gp, toi, ixg)

        # Individual shooting metrics belong to the Shooting category.
        # (Context has its own on-ice Sh%/PDO derived from on-ice GF/GA/SF/SA.)
        if category == 'Shooting' and metric in {'Sh% or FSh%', 'Sh%'}:
            return _pct(igoals, att)
        if category == 'Shooting' and metric in {'xSh% or xFS%', 'xSh% or xFSh%', 'xSh%'}:
            return _pct(ixg, att)
        if category == 'Shooting' and metric in {'dSh% or dFSh%'}:
            sh = _pct(igoals, att)
            xsh = _pct(ixg, att)
            return (sh - xsh) if (sh is not None and xsh is not None) else None

        if metric == 'GAx' and category == 'Shooting':
            # Individual goals above expected
            return _rate_from(gp, toi, (igoals - ixg))

        # On-ice totals
        if metric == 'CF':
            return _rate_from(gp, toi, cf)
        if metric == 'CA':
            return _rate_from(gp, toi, ca)
        if metric == 'FF':
            return _rate_from(gp, toi, ff)
        if metric == 'FA':
            return _rate_from(gp, toi, fa)
        if metric == 'SF':
            return _rate_from(gp, toi, sf)
        if metric == 'SA':
            return _rate_from(gp, toi, sa)
        if metric == 'GF':
            return _rate_from(gp, toi, gf)
        if metric == 'GA':
            return _rate_from(gp, toi, ga)
        if metric == 'xGF':
            return _rate_from(gp, toi, xgf)
        if metric == 'xGA':
            return _rate_from(gp, toi, xga)

        # On-ice percentages / differentials
        if metric == 'CF%':
            return _pct(cf, (cf + ca))
        if metric == 'FF%':
            return _pct(ff, (ff + fa))
        if metric == 'SF%':
            return _pct(sf, (sf + sa))
        if metric == 'GF%':
            return _pct(gf, (gf + ga))
        if metric == 'xGF%':
            return _pct(xgf, (xgf + xga))
        if metric == 'C+/-':
            return _rate_from(gp, toi, (cf - ca))
        if metric == 'F+/-':
            return _rate_from(gp, toi, (ff - fa))
        if metric == 'S+/-':
            return _rate_from(gp, toi, (sf - sa))
        if metric == 'G+/-':
            return _rate_from(gp, toi, (gf - ga))
        if metric == 'xG+/-':
            return _rate_from(gp, toi, (xgf - xga))

        if category == 'Context' and metric == 'Sh%':
            return _pct(gf, sf)
        if category == 'Context' and metric == 'Sv%':
            # Convention: if SA=0, treat on-ice Sv% as 100%.
            if sa <= 0:
                return 100.0 if ga <= 0 else 0.0
            return 100.0 * (1.0 - (ga / sa))
        if category == 'Context' and metric == 'PDO':
            sh_oi = _pct(gf, sf)
            sv_oi = 100.0 if sa <= 0 and ga <= 0 else (0.0 if sa <= 0 else 100.0 * (1.0 - (ga / sa)))
            return (sh_oi + sv_oi) if (sh_oi is not None and sv_oi is not None) else None
        if category == 'Context' and metric == 'GAx':
            return _rate_from(gp, toi, (gf - xgf))
        if category == 'Context' and metric == 'GSAx':
            return _rate_from(gp, toi, (xga - ga))

        if category == 'Penalties' and metric == 'PIM_taken':
            return _rate_from(gp, toi, pim_taken)
        if category == 'Penalties' and metric == 'PIM_drawn':
            return _rate_from(gp, toi, pim_drawn)
        if category == 'Penalties' and metric == 'PIM+/-':
            return _rate_from(gp, toi, (pim_drawn - pim_taken))
        if category == 'Penalties' and metric == 'PIM_For':
            return _rate_from(gp, toi, pim_for)
        if category == 'Penalties' and metric == 'PIM_Against':
            return _rate_from(gp, toi, pim_against)
        if category == 'Penalties' and metric == 'oiPIM+/-':
            return _rate_from(gp, toi, (pim_for - pim_against))

        if category == 'Other' and metric == 'Hits':
            return _rate_from(gp, toi, hits)
        if category == 'Other' and metric == 'Takeaways':
            return _rate_from(gp, toi, takeaways)
        if category == 'Other' and metric == 'Giveaways':
            return _rate_from(gp, toi, giveaways)

        # If the definition names a seasonstats column directly, use it.
        if metric and metric in v:
            try:
                return _rate_from(gp, toi, float(v.get(metric) or 0.0))
            except Exception:
                return None

        # Unknown
        _ = def_map.get(metric_id)
        return None

    derived: Dict[int, Dict[str, Optional[float]]] = {}
    for pid_i, v in agg.items():
        per_player: Dict[str, Optional[float]] = {}
        for mid in metric_ids:
            per_player[mid] = _compute_metric(mid, v, pid_i)
        derived[pid_i] = per_player

    # Percentiles (higher is better).
    def _percentile_sorted(values_sorted: List[float], v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        if not values_sorted:
            return None
        try:
            idx = bisect.bisect_right(values_sorted, float(v))
            return 100.0 * (idx / float(len(values_sorted)))
        except Exception:
            return None

    # Prepare distributions per metric, grouped by position (F vs D).
    # (Skater Card is skaters-only; goalies are excluded above.)
    dist_all: Dict[str, List[float]] = {mid: [] for mid in metric_ids}
    dist_by_pos: Dict[Tuple[str, str], List[float]] = {}
    edge_metric_ids = {mid for mid in metric_ids if str(mid).startswith('Edge|')}
    for pid_i, m in derived.items():
        g = pos_group_by_pid.get(int(pid_i)) or 'F'
        if g not in {'F', 'D'}:
            g = 'F'
        for mid in metric_ids:
            if mid in edge_metric_ids:
                continue
            vv = m.get(mid)
            if vv is None:
                continue
            try:
                fv = float(vv)
                if not math.isfinite(fv):
                    continue
                dist_all[mid].append(fv)
                dist_by_pos.setdefault((g, mid), []).append(fv)
            except Exception:
                continue

    # Sort pools once; avoids repeated allocations inside percentile calls.
    for mid, arr in dist_all.items():
        try:
            arr.sort()
        except Exception:
            continue
    for k, arr in dist_by_pos.items():
        try:
            arr.sort()
        except Exception:
            continue

    mine = derived.get(int(pid))
    seasonstats_missing = False
    if mine is None:
        # For some seasons (notably the in-progress 20252026), SeasonStats may be unavailable
        # (e.g., missing Sheets6 config) or incomplete. Don't hard-fail the entire card;
        # instead, fall back to an empty SeasonStats row so Edge/RAPM/Context metrics can render.
        if int(season_int or 0) == 20252026:
            seasonstats_missing = True
            empty_v: Dict[str, Any] = {
                'GP': 0,
                'TOI': 0.0,
                'iGoals': 0.0,
                'Assists1': 0.0,
                'Assists2': 0.0,
                'iShots': 0.0,
                'iFenwick': 0.0,
                'ixG_S': 0.0,
                'ixG_F': 0.0,
                'ixG_F2': 0.0,
                'CA': 0.0,
                'CF': 0.0,
                'FA': 0.0,
                'FF': 0.0,
                'SA': 0.0,
                'SF': 0.0,
                'GA': 0.0,
                'GF': 0.0,
                'xGA_S': 0.0,
                'xGF_S': 0.0,
                'xGA_F': 0.0,
                'xGF_F': 0.0,
                'xGA_F2': 0.0,
                'xGF_F2': 0.0,
                'PIM_taken': 0.0,
                'PIM_drawn': 0.0,
                'PIM_for': 0.0,
                'PIM_against': 0.0,
                'Hits': 0.0,
                'Takeaways': 0.0,
                'Giveaways': 0.0,
            }
            mine = {mid: _compute_metric(mid, empty_v, int(pid)) for mid in metric_ids}
            derived[int(pid)] = mine
        else:
            return jsonify({'error': 'not_found', 'playerId': int(pid), 'source': 'supabase'}), 404

    out_metrics: Dict[str, Any] = {}
    my_group = pos_group_by_pid.get(int(pid)) or 'F'
    if my_group not in {'F', 'D'}:
        my_group = 'F'
    for mid in metric_ids:
        val = mine.get(mid)
        pct = special_pct.get(mid)
        if mid.startswith('Projection|'):
            proj_group = _projection_pos_group_for_player(int(pid))
            if proj_group not in {'F', 'D'}:
                proj_group = my_group if my_group in {'F', 'D'} else 'F'
            pool = _projection_pool_for_group(proj_group)
            pct = _percentile_sorted(pool, val)
        if pct is None:
            pool = dist_by_pos.get((my_group, mid))
            if not pool:
                pool = dist_all.get(mid) or []
            pct = _percentile_sorted(pool, val)
            if pct is not None and _lower_is_better(mid):
                pct = 100.0 - pct
        out_metrics[mid] = {
            'value': val,
            'pct': pct,
        }

    # Provide dynamic labels used by the UI for some "or" metrics.
    label_attempts = 'iShots' if xg_model == 'xG_S' else 'iFenwick'
    label_sh = 'Sh%' if xg_model == 'xG_S' else 'FSh%'
    label_xsh = 'xSh%' if xg_model == 'xG_S' else 'xFSh%'
    label_dsh = 'dSh%' if xg_model == 'xG_S' else 'dFSh%'

    payload = {
        'playerId': int(pid),
        'season': season_int,
        'positionGroup': my_group,
        'scope': scope,
        'seasonState': season_state,
        'strengthState': strength_state,
        'xgModel': xg_model,
        'rates': rates,
        'minGP': int(min_gp),
        'minTOI': float(min_toi),
        'source': 'supabase',
        'seasonStatsMissing': bool(seasonstats_missing),
        'labels': {
            'Attempts': label_attempts,
            'Sh': label_sh,
            'xSh': label_xsh,
            'dSh': label_dsh,
        },
        'metrics': out_metrics,
    }
    j = jsonify(payload)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/goalies/card')
def api_goalies_card():
    """Goalie card metrics + league percentiles from SeasonStats.

    Query params:
      season=20252026
      playerId=<int>
      seasonState=regular|playoffs|all
      strengthState=5v5|PP|SH|Other|all
      xgModel=xG_S|xG_F|xG_F2
      rates=Totals|Per60|PerGame
      metricIds=<comma-separated Category|Metric ids>
      scope=season|career
      minGP=<int>
      minTOI=<float minutes>
    """
    season = str(request.args.get('season') or '').strip()
    player_id_q = str(request.args.get('playerId') or request.args.get('player_id') or '').strip()
    season_state = str(request.args.get('seasonState') or 'regular').strip().lower()
    strength_state = str(request.args.get('strengthState') or '5v5').strip()
    xg_model = str(request.args.get('xgModel') or 'xG_F').strip()
    rates = str(request.args.get('rates') or request.args.get('ratesTotals') or 'Totals').strip() or 'Totals'
    metric_ids_raw = str(request.args.get('metricIds') or request.args.get('metrics') or '').strip()
    scope = str(request.args.get('scope') or 'season').strip().lower()

    min_gp = _safe_int(request.args.get('minGP') or request.args.get('minGp') or request.args.get('min_gp') or 0) or 0
    min_toi_raw = request.args.get('minTOI') or request.args.get('minToi') or request.args.get('min_toi') or 0
    try:
        min_toi = float(_parse_locale_float(min_toi_raw) or 0.0)
    except Exception:
        min_toi = 0.0
    if min_gp < 0:
        min_gp = 0
    if min_toi < 0:
        min_toi = 0.0

    pid = _safe_int(player_id_q)
    if not pid or pid <= 0:
        return jsonify({'error': 'missing_playerId'}), 400

    season_ids = _parse_request_season_ids(season, default=20252026)
    season_int = _primary_season_id(season_ids, default=20252026) or 20252026

    if season_state not in {'regular', 'playoffs', 'all'}:
        season_state = 'regular'
    if strength_state not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        strength_state = '5v5'
    if xg_model not in {'xG_S', 'xG_F', 'xG_F2'}:
        xg_model = 'xG_F'
    if rates not in {'Totals', 'Per60', 'PerGame'}:
        rates = 'Totals'
    if scope not in {'season', 'career'}:
        scope = 'season'

    metric_ids: List[str] = []
    if metric_ids_raw:
        metric_ids = [s.strip() for s in metric_ids_raw.split(',') if s and s.strip()]
    if not metric_ids:
        defs0 = _load_card_metrics_defs('goalies')
        metric_ids = [str(m.get('id')) for m in (defs0.get('metrics') or []) if isinstance(m, dict) and m.get('id')]

    agg, _pos_group_by_pid = _build_goalies_seasonstats_agg(
        scope=scope,
        season_int=season_int,
        season_ids=season_ids,
        season_state=season_state,
        strength_state=strength_state,
    )

    if min_gp > 0 or min_toi > 0:
        eligible = {pid_k for pid_k, d in agg.items() if float(d.get('GP') or 0) >= float(min_gp) and float(d.get('TOI') or 0.0) >= float(min_toi)}
        agg = {pid_k: d for pid_k, d in agg.items() if pid_k in eligible}

    mine_raw = agg.get(int(pid))
    if mine_raw is None:
        return jsonify({'error': 'not_found', 'playerId': int(pid)}), 404

    def _xga(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('xGA_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('xGA_F2') or 0.0)
        return float(v.get('xGA_S') or 0.0)

    def _rate_from(gp: float, toi: float, vv: Optional[float]) -> Optional[float]:
        if rates == 'Totals':
            return vv
        denom = None
        if rates == 'PerGame':
            denom = gp if gp > 0 else None
        elif rates == 'Per60':
            denom = (toi / 60.0) if toi > 0 else None
        if vv is None or denom is None or denom <= 0:
            return None
        try:
            return float(vv) / float(denom)
        except Exception:
            return None

    def _sv_frac(ga: float, att: float) -> float:
        if att <= 0:
            return 1.0 if ga <= 0 else 0.0
        return 1.0 - (ga / att)

    # League average Sv% (weighted by SA) for GSAA.
    total_sa = 0.0
    total_ga = 0.0
    for _pid_i, vv in agg.items():
        try:
            total_sa += float(vv.get('SA') or 0.0)
            total_ga += float(vv.get('GA') or 0.0)
        except Exception:
            continue
    avg_sv = _sv_frac(float(total_ga or 0.0), float(total_sa or 0.0))

    career_gsaa_by_pid: Dict[int, float] = {}
    career_gsax_by_pid: Dict[int, float] = {}
    if scope == 'career' and any(str(mid) in {'Results|GSAA', 'Results|GSAx'} for mid in metric_ids):
        try:
            by_pid_season, league_sa_ga = _build_goalies_career_season_matrix(
                season_state=season_state,
                strength_state=strength_state,
            )

            for pid_i, vv in agg.items():
                seasons = by_pid_season.get(int(pid_i)) or {}
                gsaa_sum = 0.0
                gsax_sum = 0.0
                for s_id, srow in seasons.items():
                    try:
                        sa_s = float(srow.get('SA') or 0.0)
                        ga_s = float(srow.get('GA') or 0.0)
                    except Exception:
                        continue

                    tot_sa, tot_ga = league_sa_ga.get(int(s_id), (0.0, 0.0))
                    avg_sv_s = _sv_frac(float(tot_ga or 0.0), float(tot_sa or 0.0))
                    sv_s = _sv_frac(ga_s, sa_s)
                    gsaa_sum += (sv_s - avg_sv_s) * float(sa_s or 0.0)

                    if int(s_id) >= 20102011:
                        try:
                            xga_s = _xga(srow)
                            gsax_sum += float(xga_s or 0.0) - float(ga_s or 0.0)
                        except Exception:
                            continue

                career_gsaa_by_pid[int(pid_i)] = float(gsaa_sum)
                career_gsax_by_pid[int(pid_i)] = float(gsax_sum)
        except Exception:
            career_gsaa_by_pid = {}
            career_gsax_by_pid = {}

    needs_projection = any(mid.startswith('Projection|') for mid in metric_ids)
    projection_map_g: Dict[int, Dict[str, Any]] = {}
    if needs_projection:
        try:
            projection_map_g = _load_current_player_projections_cached() or {}
        except Exception:
            projection_map_g = {}

    def _goalie_projection_value(player_id: int) -> Optional[float]:
        if not isinstance(projection_map_g, dict):
            return None
        pid_int = int(player_id)
        row = projection_map_g.get(pid_int)
        if not isinstance(row, dict):
            row = projection_map_g.get(str(pid_int))
        if not isinstance(row, dict):
            for k, vv in projection_map_g.items():
                try:
                    if int(str(k).strip()) == pid_int and isinstance(vv, dict):
                        row = vv
                        break
                except Exception:
                    continue
        if not isinstance(row, dict):
            return None
        for key in ('projected_value', 'projection', 'projectedValue', 'Projection', 'ProjectedValue'):
            val = _parse_locale_float(row.get(key))
            if val is not None:
                return float(val)
        return None

    def _goalie_projection_pool() -> List[float]:
        if not isinstance(projection_map_g, dict) or not projection_map_g:
            return []
        out_vals: List[float] = []
        for raw in projection_map_g.values():
            if not isinstance(raw, dict):
                continue
            pg = _projection_position_group(raw.get('position'))
            if pg != 'G':
                continue
            v = None
            for key in ('projected_value', 'projection', 'projectedValue', 'Projection', 'ProjectedValue'):
                v = _parse_locale_float(raw.get(key))
                if v is not None:
                    break
            if v is None:
                continue
            try:
                fv = float(v)
                if math.isfinite(fv):
                    out_vals.append(fv)
            except Exception:
                continue
        out_vals.sort()
        return out_vals

    # Pre-compute goalie projection percentile so it bypasses the derived pool.
    projection_special_pct: Dict[str, Optional[float]] = {}
    if needs_projection:
        try:
            g_pool = _goalie_projection_pool()
            g_val = _goalie_projection_value(int(pid))
            if g_val is not None and g_pool:
                import bisect
                idx = bisect.bisect_right(g_pool, float(g_val))
                projection_special_pct['Projection|Percentile'] = 100.0 * (idx / float(len(g_pool)))
        except Exception:
            pass

    def _compute_metric(metric_id: str, pid_i: int, v: Dict[str, Any]) -> Optional[float]:
        gp = float(v.get('GP') or 0.0)
        toi = float(v.get('TOI') or 0.0)
        fa = float(v.get('FA') or 0.0)
        sa = float(v.get('SA') or 0.0)
        ga = float(v.get('GA') or 0.0)
        xga = _xga(v)

        category = None
        metric = None
        if '|' in metric_id:
            category, metric = metric_id.split('|', 1)
        else:
            metric = metric_id

        if category == 'Projection' and metric in {'Percentile', 'Projection Percentile', 'Projection'}:
            return _goalie_projection_value(int(pid_i))

        if category == 'Workload' and metric == 'FA':
            return _rate_from(gp, toi, fa)
        if category == 'Workload' and metric == 'SA':
            return _rate_from(gp, toi, sa)
        if category == 'Workload' and metric == 'xGA':
            return _rate_from(gp, toi, xga)
        if category == 'Workload' and metric == 'GA':
            return _rate_from(gp, toi, ga)

        if category == 'Save Percentage' and metric == 'Sv% or FSv%':
            denom = sa if xg_model == 'xG_S' else fa
            return 100.0 * _sv_frac(ga, denom)
        if category == 'Save Percentage' and metric == 'xSv% or xFSv%':
            denom = sa if xg_model == 'xG_S' else fa
            return 100.0 * _sv_frac(xga, denom)
        if category == 'Save Percentage' and metric == 'dSv% or dFSv%':
            denom = sa if xg_model == 'xG_S' else fa
            sv = 100.0 * _sv_frac(ga, denom)
            xsv = 100.0 * _sv_frac(xga, denom)
            return (sv - xsv)

        if category == 'Results' and metric == 'GSAx':
            if scope == 'career':
                return _rate_from(gp, toi, float(career_gsax_by_pid.get(int(pid_i), 0.0)))
            if int(season_int or 0) < 20102011:
                return _rate_from(gp, toi, 0.0)
            return _rate_from(gp, toi, (xga - ga))
        if category == 'Results' and metric == 'GSAA':
            if scope == 'career':
                return _rate_from(gp, toi, float(career_gsaa_by_pid.get(int(pid_i), 0.0)))
            sv = _sv_frac(ga, sa)
            gsaa = (sv - avg_sv) * sa
            return _rate_from(gp, toi, gsaa)

        return None

    derived: Dict[int, Dict[str, Optional[float]]] = {}
    for pid_i, v in agg.items():
        per_player: Dict[str, Optional[float]] = {}
        for mid in metric_ids:
            per_player[mid] = _compute_metric(mid, int(pid_i), v)
        derived[pid_i] = per_player

    def _percentile(values: List[float], v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        if not values:
            return None
        import bisect
        arr = sorted(values)
        idx = bisect.bisect_right(arr, float(v))
        return 100.0 * (idx / float(len(arr)))

    def _lower_is_better(mid: str) -> bool:
        m = mid
        if '|' in mid:
            _, m = mid.split('|', 1)
        m = str(m or '').strip()
        return m in {'GA', 'xGA'}

    dist_all: Dict[str, List[float]] = {mid: [] for mid in metric_ids}
    for _pid_i, mm in derived.items():
        for mid in metric_ids:
            vv = mm.get(mid)
            if vv is None:
                continue
            try:
                fv = float(vv)
                if not math.isfinite(fv):
                    continue
                dist_all[mid].append(fv)
            except Exception:
                continue

    mine = derived.get(int(pid)) or {}
    out_metrics: Dict[str, Any] = {}
    for mid in metric_ids:
        val = mine.get(mid)
        # Use pre-computed percentile for projection (filtered to goalie pool).
        if mid in projection_special_pct:
            pct = projection_special_pct[mid]
        else:
            pool = dist_all.get(mid) or []
            pct = _percentile(pool, val)
            if pct is not None and _lower_is_better(mid):
                pct = 100.0 - pct
        out_metrics[mid] = {'value': val, 'pct': pct}

    label_attempts = 'SA' if xg_model == 'xG_S' else 'FA'
    label_sv = 'Sv%' if xg_model == 'xG_S' else 'FSv%'
    label_xsv = 'xSv%' if xg_model == 'xG_S' else 'xFSv%'
    label_dsv = 'dSv%' if xg_model == 'xG_S' else 'dFSv%'

    payload = {
        'playerId': int(pid),
        'season': int(season_int),
        'scope': scope,
        'seasonState': season_state,
        'strengthState': strength_state,
        'xgModel': xg_model,
        'rates': rates,
        'minGP': int(min_gp),
        'minTOI': float(min_toi),
        'labels': {
            'Attempts': label_attempts,
            'Sv': label_sv,
            'xSv': label_xsv,
            'dSv': label_dsv,
        },
        'metrics': out_metrics,
    }
    j = jsonify(payload)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


# -----------------------------
# Teams APIs
# -----------------------------

_TEAM_STATS_REST_CACHE: Dict[str, Tuple[float, Any]] = {}


def _team_stats_rest_get(url: str) -> Optional[Dict[str, Any]]:
    """Fetch NHL stats REST JSON with a small in-memory TTL cache."""
    try:
        ttl_s = max(30, int(os.getenv('TEAM_STATS_REST_CACHE_TTL_SECONDS', '3600') or '3600'))
    except Exception:
        ttl_s = 3600

    try:
        max_items = max(1, int(os.getenv('TEAM_STATS_REST_CACHE_MAX_ITEMS', '128') or '128'))
    except Exception:
        max_items = 128

    now = time.time()
    _cache_prune_ttl_and_size(_TEAM_STATS_REST_CACHE, ttl_s=ttl_s, max_items=max_items)
    cached = _TEAM_STATS_REST_CACHE.get(url)
    if cached and (now - cached[0]) < ttl_s:
        try:
            return cached[1]
        except Exception:
            return None

    try:
        r = requests.get(url, timeout=25)
        if r.status_code != 200:
            return None
        j = r.json()
        if not isinstance(j, dict):
            return None
        _cache_set_multi_bounded(_TEAM_STATS_REST_CACHE, url, j, ttl_s=ttl_s, max_items=max_items)
        return j
    except Exception:
        return None


def _edge_rank_to_pct(rank: Any, total_teams: int = 32) -> Optional[float]:
    try:
        rr = int(float(rank))
        if total_teams <= 1:
            return None
        if rr < 1 or rr > total_teams:
            return None
        return 100.0 * ((float(total_teams) - float(rr)) / float(total_teams - 1))
    except Exception:
        return None


def _edge_team_extract_value_rank_avg(
    payload: Dict[str, Any],
    metric_key: str,
    strength_code: Optional[str],
    position_code: Optional[str],
) -> Tuple[Optional[float], Optional[int], Optional[float]]:
    """Extract {value, rank, leagueAvg} from NHL Edge team payloads."""
    try:
        # Find a list-of-rows node we can filter by strengthCode/positionCode.
        rows: Optional[List[Dict[str, Any]]] = None
        for v in payload.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                keys0 = {str(k).lower() for k in v[0].keys()}
                has_sc = 'strengthcode' in keys0
                # Some team endpoints (e.g. shot speed) use `position` instead of `positionCode`.
                has_pc = ('positioncode' in keys0) or ('position' in keys0)
                if (strength_code and has_sc) or (position_code and has_pc):
                    rows = v  # type: ignore[assignment]
                    break
                if has_sc or has_pc:
                    rows = v  # type: ignore[assignment]

        row: Optional[Dict[str, Any]] = None
        if rows:
            candidates = rows
            sc = str(strength_code).lower() if strength_code else None
            pc = str(position_code).lower() if position_code else None

            def _sc(rr0: Dict[str, Any]) -> str:
                return str(rr0.get('strengthCode') or '').lower()

            def _pc(rr0: Dict[str, Any]) -> str:
                return str(rr0.get('positionCode') or rr0.get('position') or '').lower()

            # Best match: strength + position together.
            if sc and pc:
                for rr in candidates:
                    if _sc(rr) == sc and _pc(rr) == pc:
                        row = rr
                        break
            # Fallbacks: exact strength, exact position, then mixed "all".
            if row is None and sc:
                for rr in candidates:
                    if _sc(rr) == sc:
                        row = rr
                        break
            if row is None and pc:
                for rr in candidates:
                    if _pc(rr) == pc:
                        row = rr
                        break
            if row is None and sc and pc:
                for rr in candidates:
                    if _sc(rr) == sc and _pc(rr) == 'all':
                        row = rr
                        break
            if row is None and sc and pc:
                for rr in candidates:
                    if _sc(rr) == 'all' and _pc(rr) == pc:
                        row = rr
                        break
            if row is None and sc:
                for rr in candidates:
                    if _sc(rr) == 'all':
                        row = rr
                        break
            if row is None and pc:
                for rr in candidates:
                    if _pc(rr) == 'all':
                        row = rr
                        break
            if row is None and candidates:
                row = candidates[0]
        else:
            row = None

        def _to_float(x: Any) -> Optional[float]:
            try:
                if x is None:
                    return None
                f = float(x)
                if not math.isfinite(f):
                    return None
                return f
            except Exception:
                return None

        def _pick_val(node: Any) -> Any:
            if isinstance(node, dict):
                return _ci_get(node, 'imperial') or _ci_get(node, 'value') or _ci_get(node, 'metric')
            return node

        def _pick_avg(node: Any) -> Any:
            if isinstance(node, dict):
                avg = _ci_get(node, 'leagueAvg') or _ci_get(node, 'leagueAverage')
                if isinstance(avg, dict):
                    return _ci_get(avg, 'imperial') or _ci_get(avg, 'value') or _ci_get(avg, 'metric')
                return avg
            return None

        def _pick_rank_from_row(r0: Dict[str, Any]) -> Any:
            mk = str(metric_key)
            # zone time uses offensiveZoneRank for offensiveZonePctg
            if mk.endswith('Pctg'):
                base = mk[:-4]
                return _ci_get(r0, f'{base}Rank') or _ci_get(r0, f'{mk}Rank')
            return _ci_get(r0, f'{mk}Rank')

        # Primary extraction: row-based if present.
        if isinstance(row, dict):
            node = _ci_get(row, metric_key)
            rank_raw = None
            avg_raw = None
            if isinstance(node, dict):
                rank_raw = _ci_get(node, 'rank')
                avg_raw = _pick_avg(node)
            if rank_raw is None:
                rank_raw = _pick_rank_from_row(row)

            val = _pick_val(node)
            out_val = _to_float(val)
            out_avg = _to_float(avg_raw)
            out_rank = None
            try:
                if rank_raw is not None:
                    out_rank = int(float(rank_raw))
            except Exception:
                out_rank = None

            if out_val is not None and str(metric_key).endswith('Pctg') and 0.0 <= out_val <= 1.5:
                out_val = 100.0 * out_val
            if out_avg is not None and str(metric_key).endswith('Pctg') and 0.0 <= out_avg <= 1.5:
                out_avg = 100.0 * out_avg
            return (out_val, out_rank, out_avg)

        # Fallback: nested dict-of-metrics (rare for team endpoints, but safe)
        for v in payload.values():
            if isinstance(v, dict):
                node = _ci_get(v, metric_key)
                if not isinstance(node, dict):
                    continue
                out_val = _to_float(_pick_val(node))
                out_avg = _to_float(_pick_avg(node))
                out_rank = None
                try:
                    out_rank = int(float(_ci_get(node, 'rank')))
                except Exception:
                    out_rank = None
                if out_val is not None and str(metric_key).endswith('Pctg') and 0.0 <= out_val <= 1.5:
                    out_val = 100.0 * out_val
                if out_avg is not None and str(metric_key).endswith('Pctg') and 0.0 <= out_avg <= 1.5:
                    out_avg = 100.0 * out_avg
                return (out_val, out_rank, out_avg)

        return (None, None, None)
    except Exception:
        return (None, None, None)


def _team_id_by_abbrev() -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in (TEAM_ROWS or []):
        try:
            ab = str(r.get('Team') or '').strip().upper()
            tid = _safe_int(r.get('TeamID'))
            if ab and tid and tid > 0:
                out[ab] = int(tid)
        except Exception:
            continue
    return out


def _build_team_base_stats(*, scope: str, season_int: Optional[int] = None, season_ids: Optional[Sequence[int]] = None, season_state: str, strength_state: str = '5v5', debug_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """Return per-team base totals keyed by team abbrev.

        For scope='season', prefers our derived team SeasonStats from Supabase / CSV.

        For scope='total', falls back to NHL stats REST aggregates.
    """
    scope_norm = (scope or 'season').strip().lower()
    if scope_norm not in {'season', 'total'}:
        scope_norm = 'season'
    ss_norm = (season_state or 'regular').strip().lower()
    if ss_norm not in {'regular', 'playoffs', 'all'}:
        ss_norm = 'regular'
    st_norm = (strength_state or '5v5').strip()
    if st_norm not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        st_norm = '5v5'

    season_ids_norm = _normalize_season_id_list(season_ids, default=season_int)
    primary_season_int = _primary_season_id(season_ids_norm, default=season_int) or 0

    team_id_map = _team_id_by_abbrev()
    abbrev_by_id = {tid: ab for ab, tid in team_id_map.items()}

    def _flt_num(v: Any) -> float:
        x = _parse_locale_float(v)
        if x is None:
            return 0.0
        try:
            f = float(x)
            return f if math.isfinite(f) else 0.0
        except Exception:
            return 0.0

    # Season scope: prefer derived team SeasonStats from Supabase (all seasons).
    # Falls back to CSV if Supabase is unavailable.
    if scope_norm == 'season':
        if isinstance(debug_meta, dict):
            debug_meta.update({
                'scope': scope_norm,
                'season': int(primary_season_int or 0),
                'seasons': list(season_ids_norm),
                'seasonState': ss_norm,
                'strengthState': st_norm,
            })

        rows_iter: Iterable[Dict[str, Any]]
        rows_iter = _iter_teamseasonstats_static_rows(seasons=season_ids_norm or [int(primary_season_int)])
        if isinstance(debug_meta, dict):
            debug_meta['teamSeasonStatsSource'] = 'supabase'

        def _ss_row(v: Any) -> str:
            raw = str(v or '').strip().lower()
            if raw in {'2', 'reg', 'regular', 'regularseason', 'regular_season'}:
                return 'regular'
            if raw in {'3', 'po', 'playoffs', 'playoff'}:
                return 'playoffs'
            return raw or 'regular'

        def _st_row(v: Any) -> str:
            s = str(v or '').strip()
            return s or 'Other'

        acc: Dict[str, Dict[str, Any]] = {}
        gp_max_by_team_ss: Dict[Tuple[str, str], int] = {}

        for r in rows_iter:
            try:
                team = str(r.get('Team') or _ci_get(r, 'Team') or '').strip().upper()
                if not team:
                    continue

                ss_row = _ss_row(r.get('SeasonState') or _ci_get(r, 'SeasonState') or _ci_get(r, 'seasonState'))
                st_row = _st_row(r.get('StrengthState') or _ci_get(r, 'StrengthState') or _ci_get(r, 'strengthState'))
                if ss_norm != 'all' and ss_row != ss_norm:
                    continue
                if st_norm != 'all' and st_row != st_norm:
                    continue

                d = acc.setdefault(team, {
                    'team': team,
                    'teamId': int(team_id_map.get(team) or 0),
                    'scope': 'season',
                    'season': int(primary_season_int),
                    'seasonState': ss_norm,
                    'GP': 0,
                    'TOI': 0.0,
                    'CF': 0.0,
                    'CA': 0.0,
                    'FF': 0.0,
                    'FA': 0.0,
                    'SF': 0.0,
                    'SA': 0.0,
                    'GF': 0.0,
                    'GA': 0.0,
                    'xGF': 0.0,
                    'xGA': 0.0,
                })

                gp_row = int(_safe_int(r.get('GP') or _ci_get(r, 'GP')) or 0)
                k = (team, ss_row)
                prev = gp_max_by_team_ss.get(k)
                if prev is None or gp_row > prev:
                    gp_max_by_team_ss[k] = gp_row

                d['TOI'] = float(d.get('TOI') or 0.0) + _flt_num(r.get('TOI') or _ci_get(r, 'TOI'))
                d['CF'] = float(d.get('CF') or 0.0) + _flt_num(r.get('CF') or _ci_get(r, 'CF'))
                d['CA'] = float(d.get('CA') or 0.0) + _flt_num(r.get('CA') or _ci_get(r, 'CA'))
                d['FF'] = float(d.get('FF') or 0.0) + _flt_num(r.get('FF') or _ci_get(r, 'FF'))
                d['FA'] = float(d.get('FA') or 0.0) + _flt_num(r.get('FA') or _ci_get(r, 'FA'))
                d['SF'] = float(d.get('SF') or 0.0) + _flt_num(r.get('SF') or _ci_get(r, 'SF'))
                d['SA'] = float(d.get('SA') or 0.0) + _flt_num(r.get('SA') or _ci_get(r, 'SA'))
                d['GF'] = float(d.get('GF') or 0.0) + _flt_num(r.get('GF') or _ci_get(r, 'GF'))
                d['GA'] = float(d.get('GA') or 0.0) + _flt_num(r.get('GA') or _ci_get(r, 'GA'))

                # Default xG model for Teams page: xG_F columns (matches skaters default).
                d['xGF'] = float(d.get('xGF') or 0.0) + _flt_num(r.get('xGF_F') or _ci_get(r, 'xGF_F'))
                d['xGA'] = float(d.get('xGA') or 0.0) + _flt_num(r.get('xGA_F') or _ci_get(r, 'xGA_F'))
            except Exception:
                continue

        gp_sum_by_team: Dict[str, int] = {}
        for (team, _ss), gp in gp_max_by_team_ss.items():
            gp_sum_by_team[team] = int(gp_sum_by_team.get(team, 0) + int(gp or 0))
        for team, d in acc.items():
            d['GP'] = int(gp_sum_by_team.get(team, 0))

        # If we successfully built derived SeasonStats, we're done.
        if acc:
            if isinstance(debug_meta, dict):
                debug_meta['teamSeasonStatsTeamsCount'] = int(len(acc))
            return acc
        # Else fall through to NHL stats REST season aggregates below.
        if isinstance(debug_meta, dict):
            debug_meta['teamSeasonStatsTeamsCount'] = 0
            debug_meta['teamSeasonStatsSource'] = 'nhl_rest_fallback'

    def _summary_url(season_id: int, game_type_id: int) -> str:
        return (
            'https://api.nhle.com/stats/rest/en/team/summary'
            '?isAggregate=false&isGame=false&reportType=basic&reportName=teamsummary'
            f'&cayenneExp=seasonId={int(season_id)}%20and%20gameTypeId={int(game_type_id)}'
        )

    def _shoot_url(season_id: int, game_type_id: int) -> str:
        return (
            'https://api.nhle.com/stats/rest/en/team/summaryshooting'
            '?isAggregate=false&isGame=false&reportType=basic&reportName=teamsummaryshooting'
            f'&cayenneExp=seasonId={int(season_id)}%20and%20gameTypeId={int(game_type_id)}'
        )

    def _summary_url_total(team_id: int) -> str:
        return (
            'https://api.nhle.com/stats/rest/en/team/summary'
            '?isAggregate=true&isGame=false&reportType=basic&reportName=teamsummary'
            f'&cayenneExp=teamId={int(team_id)}'
        )

    def _shoot_url_total(team_id: int) -> str:
        return (
            'https://api.nhle.com/stats/rest/en/team/summaryshooting'
            '?isAggregate=true&isGame=false&reportType=basic&reportName=teamsummaryshooting'
            f'&cayenneExp=teamId={int(team_id)}'
        )

    def _rows_by_teamid(j: Optional[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}
        try:
            rows = j.get('data') if isinstance(j, dict) else None
            if not isinstance(rows, list):
                return out
            for r in rows:
                if not isinstance(r, dict):
                    continue
                tid = _safe_int(r.get('teamId'))
                if tid and tid > 0:
                    out[int(tid)] = r
        except Exception:
            return out
        return out

    # Season scope: fetch league-wide in a handful of requests.
    if scope_norm == 'season':
        gtypes: List[int]
        if ss_norm == 'regular':
            gtypes = [2]
        elif ss_norm == 'playoffs':
            gtypes = [3]
        else:
            gtypes = [2, 3]

        summary_by_tid: Dict[int, Dict[str, Any]] = {}
        shoot_by_tid: Dict[int, Dict[str, Any]] = {}
        for season_id in (season_ids_norm or [int(primary_season_int)]):
            for gt in gtypes:
                js = _team_stats_rest_get(_summary_url(int(season_id), gt)) or {}
                jj = _team_stats_rest_get(_shoot_url(int(season_id), gt)) or {}
                for tid, r in _rows_by_teamid(js).items():
                    prev = summary_by_tid.get(tid)
                    if not prev:
                        summary_by_tid[tid] = dict(r)
                    else:
                        # Additive totals
                        prev['gamesPlayed'] = _flt_num(prev.get('gamesPlayed')) + _flt_num(r.get('gamesPlayed'))
                        prev['goalsFor'] = _flt_num(prev.get('goalsFor')) + _flt_num(r.get('goalsFor'))
                        prev['goalsAgainst'] = _flt_num(prev.get('goalsAgainst')) + _flt_num(r.get('goalsAgainst'))
                        prev['_shotsForTotal'] = _flt_num(prev.get('_shotsForTotal')) + (_flt_num(r.get('shotsForPerGame')) * _flt_num(r.get('gamesPlayed')))
                        prev['_shotsAgainstTotal'] = _flt_num(prev.get('_shotsAgainstTotal')) + (_flt_num(r.get('shotsAgainstPerGame')) * _flt_num(r.get('gamesPlayed')))
                for tid, r in _rows_by_teamid(jj).items():
                    prev = shoot_by_tid.get(tid)
                    if not prev:
                        shoot_by_tid[tid] = dict(r)
                    else:
                        prev['gamesPlayed'] = _flt_num(prev.get('gamesPlayed')) + _flt_num(r.get('gamesPlayed'))
                        prev['satFor'] = _flt_num(prev.get('satFor')) + _flt_num(r.get('satFor'))
                        prev['satAgainst'] = _flt_num(prev.get('satAgainst')) + _flt_num(r.get('satAgainst'))
                        prev['usatFor'] = _flt_num(prev.get('usatFor')) + _flt_num(r.get('usatFor'))
                        prev['usatAgainst'] = _flt_num(prev.get('usatAgainst')) + _flt_num(r.get('usatAgainst'))

        out: Dict[str, Dict[str, Any]] = {}
        for tid, rsum in summary_by_tid.items():
            ab = abbrev_by_id.get(int(tid))
            if not ab:
                continue
            gp = float(rsum.get('gamesPlayed') or 0.0)
            sf_total = _flt_num(rsum.get('_shotsForTotal')) if '_shotsForTotal' in rsum else (_flt_num(rsum.get('shotsForPerGame')) * gp)
            sa_total = _flt_num(rsum.get('_shotsAgainstTotal')) if '_shotsAgainstTotal' in rsum else (_flt_num(rsum.get('shotsAgainstPerGame')) * gp)
            out[ab] = {
                'team': ab,
                'teamId': int(tid),
                'scope': 'season',
                'season': int(primary_season_int),
                'seasonState': ss_norm,
                'GP': gp,
                'GF': _flt_num(rsum.get('goalsFor')),
                'GA': _flt_num(rsum.get('goalsAgainst')),
                'SF': sf_total,
                'SA': sa_total,
                'xGF': None,
                'xGA': None,
            }

        for tid, rsh in shoot_by_tid.items():
            ab = abbrev_by_id.get(int(tid))
            if not ab or ab not in out:
                continue
            out[ab].update({
                'CF': _flt_num(rsh.get('satFor')),
                'CA': _flt_num(rsh.get('satAgainst')),
                'FF': _flt_num(rsh.get('usatFor')),
                'FA': _flt_num(rsh.get('usatAgainst')),
            })

        return out

    # Total scope: per-team (teamId-filtered) aggregate endpoints.
    out2: Dict[str, Dict[str, Any]] = {}
    for ab, tid in team_id_map.items():
        js = _team_stats_rest_get(_summary_url_total(tid))
        jj = _team_stats_rest_get(_shoot_url_total(tid))
        try:
            row_s = (js or {}).get('data')
            row_j = (jj or {}).get('data')
            rs = row_s[0] if isinstance(row_s, list) and row_s and isinstance(row_s[0], dict) else {}
            rsh = row_j[0] if isinstance(row_j, list) and row_j and isinstance(row_j[0], dict) else {}

            gp = _flt_num(rs.get('gamesPlayed'))
            sf_total = _flt_num(rs.get('shotsForPerGame')) * gp
            sa_total = _flt_num(rs.get('shotsAgainstPerGame')) * gp
            out2[ab] = {
                'team': ab,
                'teamId': int(tid),
                'scope': 'total',
                'season': None,
                'seasonState': 'all',
                'GP': gp,
                'GF': _flt_num(rs.get('goalsFor')),
                'GA': _flt_num(rs.get('goalsAgainst')),
                'SF': sf_total,
                'SA': sa_total,
                'CF': _flt_num(rsh.get('satFor')),
                'CA': _flt_num(rsh.get('satAgainst')),
                'FF': _flt_num(rsh.get('usatFor')),
                'FA': _flt_num(rsh.get('usatAgainst')),
                'xGF': None,
                'xGA': None,
            }
        except Exception:
            continue
    return out2


@main_bp.route('/api/teams/card')
def api_teams_card():
    """Team card metrics + league percentiles (32 teams) from NHL stats REST + NHL Edge (rank->pct).

    Query params:
      season=<int>
      team=<abbrev>
      seasonState=regular|playoffs|all
      strengthState=5v5|PP|SH|Other|all
      rates=Totals|PerGame
      metricIds=<comma-separated Category|Metric ids>
      scope=season|total
    """
    season = str(request.args.get('season') or '').strip()
    team_ab = str(request.args.get('team') or request.args.get('teamAbbrev') or request.args.get('team_abbrev') or '').strip().upper()
    season_state = str(request.args.get('seasonState') or 'regular').strip().lower()
    strength_state = str(request.args.get('strengthState') or '5v5').strip()
    position_code_req = str(request.args.get('positionCode') or request.args.get('posCode') or request.args.get('position') or 'all').strip().lower() or 'all'
    rates = str(request.args.get('rates') or 'Totals').strip() or 'Totals'
    scope = str(request.args.get('scope') or 'season').strip().lower()
    metric_ids_raw = str(request.args.get('metricIds') or request.args.get('metrics') or '').strip()
    want_debug = str(request.args.get('debug', '')).strip() in ('1', 'true', 'yes', 'y')

    season_ids = _parse_request_season_ids(season, default=20252026)
    season_int = _primary_season_id(season_ids, default=20252026) or 20252026

    if season_state not in {'regular', 'playoffs', 'all'}:
        season_state = 'regular'
    if strength_state not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        strength_state = '5v5'
    if rates not in {'Totals', 'PerGame'}:
        rates = 'Totals'
    if scope not in {'season', 'total'}:
        scope = 'season'

    if not team_ab:
        return jsonify({'error': 'missing_team'}), 400

    metric_ids: List[str] = []
    if metric_ids_raw:
        metric_ids = [s.strip() for s in metric_ids_raw.split(',') if s and s.strip()]
    if not metric_ids:
        defs0 = _load_card_metrics_defs('teams')
        metric_ids = [str(m.get('id')) for m in (defs0.get('metrics') or []) if isinstance(m, dict) and m.get('id')]

    debug_meta: Optional[Dict[str, Any]] = {} if want_debug else None
    base = _build_team_base_stats(
        scope=scope,
        season_int=season_int,
        season_ids=season_ids,
        season_state=season_state,
        strength_state=strength_state,
        debug_meta=debug_meta,
    )
    mine_base = base.get(team_ab)
    if not mine_base:
        payload: Dict[str, Any] = {'error': 'not_found', 'team': team_ab}
        if want_debug and isinstance(debug_meta, dict):
            payload['debug'] = debug_meta
            payload['debug']['baseTeamsCount'] = int(len(base or {}))
            payload['debug']['baseTeamsSample'] = sorted(list((base or {}).keys()))[:10]
        return jsonify(payload), 404

    defs = _load_card_metrics_defs('teams')
    def_map: Dict[str, Dict[str, Any]] = {str(m.get('id')): m for m in (defs.get('metrics') or []) if isinstance(m, dict) and m.get('id')}

    def _pct(n: Optional[float], d: Optional[float]) -> Optional[float]:
        try:
            if n is None or d is None:
                return None
            if d <= 0:
                return None
            return 100.0 * (float(n) / float(d))
        except Exception:
            return None

    def _rate_from(gp: float, vv: Optional[float]) -> Optional[float]:
        if rates == 'Totals':
            return vv
        if gp <= 0:
            return None
        try:
            return float(vv) / float(gp) if vv is not None else None
        except Exception:
            return None

    # For Edge metrics we compute percentiles directly from rank (not from league distributions).
    special_pct: Dict[str, Optional[float]] = {}

    # Pre-compute projection rankings using the same current-lineup aggregation
    # logic as the Teams Projections league table.
    needs_proj_ranking = any(mid.startswith('Projection|') for mid in metric_ids)
    team_projection_rank: Dict[str, int] = {}
    if needs_proj_ranking:
        try:
            lineups_all = _load_lineups_all() or {}
            proj_map_t = _load_current_player_projections_cached() or {}
            skater_bios = _load_skater_bios_season_cached(int(season_int)) or {}
            goalie_bios = _load_goalie_bios_season_cached(int(season_int)) or {}
            roster_map = {**skater_bios, **goalie_bios}
            team_proj_totals: Dict[str, float] = {}

            lineup_pids_by_team: Dict[str, set[int]] = {}
            for team_abbrev_i, node in lineups_all.items():
                try:
                    team_key = str(team_abbrev_i or '').strip().upper()
                    if not team_key:
                        continue
                    ids: set[int] = set()
                    for bucket in ('forwards', 'defense', 'goalies'):
                        for player in (node.get(bucket) or []):
                            pid_i = _safe_int((player or {}).get('playerId'))
                            if pid_i and pid_i > 0:
                                ids.add(int(pid_i))
                    lineup_pids_by_team[team_key] = ids
                except Exception:
                    continue

            for pid_raw, raw in (proj_map_t or {}).items():
                try:
                    pid_i = _safe_int((raw or {}).get('player_id') or (raw or {}).get('playerId') or pid_raw)
                    if not pid_i or pid_i <= 0:
                        continue
                    info = roster_map.get(int(pid_i)) or {}
                    team_abbrev_i = str(info.get('team') or '').strip().upper()
                    if not team_abbrev_i:
                        continue
                    if int(pid_i) not in lineup_pids_by_team.get(team_abbrev_i, set()):
                        continue
                    projection_i = _parse_locale_float((raw or {}).get('projected_value'))
                    if projection_i is None:
                        continue
                    team_proj_totals[team_abbrev_i] = float(team_proj_totals.get(team_abbrev_i, 0.0)) + float(projection_i)
                except Exception:
                    continue

            sorted_teams = sorted(team_proj_totals.items(), key=lambda x: x[1], reverse=True)
            for rank_i, (tab, _) in enumerate(sorted_teams, start=1):
                team_projection_rank[tab] = rank_i
        except Exception:
            team_projection_rank = {}

    if needs_proj_ranking and team_ab in team_projection_rank:
        # No percentile for ranking; frontend colors by rank value directly.
        special_pct['Projection|Ranking'] = None

    edge_seasons = [int(s) for s in season_ids if int(s) >= 20212022] if scope == 'season' else ([int(season_int)] if int(season_int) >= 20212022 else [])

    def _team_edge_value_mode(metric_name: str) -> str:
        metric_norm = str(metric_name or '').strip()
        if metric_norm in {'topShotSpeed', 'maxSkatingSpeed'}:
            return 'max'
        if metric_norm in {
            'shotAttempts70to80', 'shotAttempts80to90', 'shotAttempts90to100', 'shotAttemptsOver100',
            'bursts18to20', 'bursts20to22', 'burstsOver22',
            'distanceTotal', 'distanceTotal or distancePer60',
        }:
            return 'sum'
        return 'avg'

    def _aggregate_team_edge_metric(
        *,
        team_id: int,
        metric_name: str,
        link: str,
        strength_code: Optional[str],
        pos_code: Optional[str],
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        if not edge_seasons:
            return None, None, None, None

        game_type = _edge_game_type(season_state)
        value_mode = _team_edge_value_mode(metric_name)
        vals: List[float] = []
        pcts: List[float] = []
        avgs: List[float] = []
        ranks: List[float] = []

        for season_id_i in edge_seasons:
            url = _edge_format_url(link, int(team_id), int(season_id_i), int(game_type))
            if not url:
                continue
            payload_edge = _edge_get_cached_json(url)
            if not payload_edge:
                continue
            val_e, rank_e, avg_e = _edge_team_extract_value_rank_avg(payload_edge, str(metric_name or ''), strength_code, pos_code)
            pct_e = _edge_rank_to_pct(rank_e, 32) if rank_e is not None else None

            try:
                if val_e is not None and str(metric_name or '').lower().endswith('pctg') and 0.0 <= float(val_e) <= 1.5:
                    val_e = 100.0 * float(val_e)
            except Exception:
                pass
            try:
                if avg_e is not None and str(metric_name or '').lower().endswith('pctg') and 0.0 <= float(avg_e) <= 1.5:
                    avg_e = 100.0 * float(avg_e)
            except Exception:
                pass

            if val_e is not None:
                vals.append(float(val_e))
            if pct_e is not None:
                pcts.append(float(pct_e))
            if avg_e is not None:
                avgs.append(float(avg_e))
            if rank_e is not None:
                ranks.append(float(rank_e))

        if not vals and not pcts and not avgs and not ranks:
            return None, None, None, None

        if vals:
            if value_mode == 'sum':
                val_out = float(sum(vals))
            elif value_mode == 'max':
                val_out = float(max(vals))
            else:
                val_out = float(sum(vals) / len(vals))
        else:
            val_out = None

        pct_out = float(sum(pcts) / len(pcts)) if pcts else None
        avg_out = float(sum(avgs) / len(avgs)) if avgs else None
        rank_out = float(sum(ranks) / len(ranks)) if ranks else None
        return val_out, pct_out, avg_out, rank_out

    def _compute_metric(metric_id: str, v: Dict[str, Any], team_id: int) -> Optional[float]:
        gp = float(v.get('GP') or 0.0)
        cf = float(v.get('CF') or 0.0)
        ca = float(v.get('CA') or 0.0)
        ff = float(v.get('FF') or 0.0)
        fa = float(v.get('FA') or 0.0)
        sf = float(v.get('SF') or 0.0)
        sa = float(v.get('SA') or 0.0)
        gf = float(v.get('GF') or 0.0)
        ga = float(v.get('GA') or 0.0)
        xgf = v.get('xGF')
        xga = v.get('xGA')

        category = None
        metric = None
        if '|' in metric_id:
            category, metric = metric_id.split('|', 1)
        else:
            metric = metric_id

        if category == 'Edge':
            if scope == 'season' and not edge_seasons:
                special_pct[metric_id] = None
                return None
            mdef = def_map.get(metric_id) or {}
            link = str(mdef.get('link') or '').strip()
            if not link:
                special_pct[metric_id] = None
                return None
            game_type = _edge_game_type(season_state)
            url = _edge_format_url(link, int(team_id), int(season_int or 0), int(game_type))
            if not url:
                special_pct[metric_id] = None
                return None
            payload_edge = _edge_get_cached_json(url)
            if not payload_edge:
                special_pct[metric_id] = None
                return None

            strength_code = None
            if str(mdef.get('strengthCode') or '').strip().lower() == 'strengthcode':
                strength_code = _edge_strength_code(strength_state)
            pos_code = None
            if str(mdef.get('positionCode') or '').strip().lower() in {'positioncode', 'position'}:
                pos_code = position_code_req or 'all'

            val_e, pct_e, avg_e, rank_e = _aggregate_team_edge_metric(
                team_id=int(team_id),
                metric_name=str(metric or ''),
                link=link,
                strength_code=strength_code,
                pos_code=pos_code,
            )
            special_pct[metric_id] = pct_e if pct_e is not None else None
            # Keep leagueAvg in case the UI wants it later.
            try:
                if avg_e is not None:
                    v.setdefault('_edgeAvg', {})[metric_id] = avg_e
                if rank_e is not None:
                    v.setdefault('_edgeRank', {})[metric_id] = rank_e
            except Exception:
                pass
            return val_e

        # Base totals
        if metric == 'CA':
            return _rate_from(gp, ca)
        if metric == 'FA':
            return _rate_from(gp, fa)
        if metric == 'SA':
            return _rate_from(gp, sa)
        if metric == 'GA':
            return _rate_from(gp, ga)
        if metric == 'xGA':
            return _rate_from(gp, float(xga) if xga is not None else None)

        if metric == 'CF':
            return _rate_from(gp, cf)
        if metric == 'FF':
            return _rate_from(gp, ff)
        if metric == 'SF':
            return _rate_from(gp, sf)
        if metric == 'GF':
            return _rate_from(gp, gf)
        if metric == 'xGF':
            return _rate_from(gp, float(xgf) if xgf is not None else None)

        # Percentages
        if metric == 'CF%':
            return _pct(cf, (cf + ca))
        if metric == 'FF%':
            return _pct(ff, (ff + fa))
        if metric == 'SF%':
            return _pct(sf, (sf + sa))
        if metric == 'GF%':
            return _pct(gf, (gf + ga))
        if metric == 'xGF%':
            if xgf is None or xga is None:
                return None
            return _pct(float(xgf), float(xgf) + float(xga))

        # Differentials
        if metric == 'C+/-':
            return _rate_from(gp, (cf - ca))
        if metric == 'F+/-':
            return _rate_from(gp, (ff - fa))
        if metric == 'S+/-':
            return _rate_from(gp, (sf - sa))
        if metric == 'G+/-':
            return _rate_from(gp, (gf - ga))
        if metric == 'xG+/-':
            if xgf is None or xga is None:
                return None
            return _rate_from(gp, (float(xgf) - float(xga)))

        # Context
        if category == 'Context' and metric == 'Sh%':
            return _pct(gf, sf)
        if category == 'Context' and metric == 'Sv%':
            if sa <= 0:
                return 100.0 if ga <= 0 else 0.0
            return 100.0 * (1.0 - (ga / sa))
        if category == 'Context' and metric == 'PDO':
            sh_oi = _pct(gf, sf)
            sv_oi = 100.0 if sa <= 0 and ga <= 0 else (0.0 if sa <= 0 else 100.0 * (1.0 - (ga / sa)))
            return (sh_oi + sv_oi) if (sh_oi is not None and sv_oi is not None) else None
        if category == 'Context' and metric == 'GAx':
            if xgf is None:
                return None
            return _rate_from(gp, (gf - float(xgf)))
        if category == 'Context' and metric == 'GSAx':
            if xga is None:
                return None
            return _rate_from(gp, (float(xga) - ga))

        if category == 'Context' and metric == 'GDAx':
            if xgf is None or xga is None:
                return None
            # Goal differential above expected = (GF - xGF) + (xGA - GA)
            return _rate_from(gp, (gf - float(xgf)) + (float(xga) - ga))

        if category == 'Projection' and metric in {'Ranking', 'Rank', 'Projection Ranking'}:
            # Return the team's projection ranking (1=best, 32=worst).
            rank = team_projection_rank.get(team_ab)
            if rank is not None:
                return float(rank)
            return None

        return None

    edge_metric_ids = {mid for mid in metric_ids if str(mid).startswith('Edge|')}

    # If the request is Edge-only, do NOT compute Edge for all teams.
    # Edge endpoints already provide league rank, so we can render rank-based bars
    # without expensive 32-team fanout.
    if edge_metric_ids and len(edge_metric_ids) == len(metric_ids):
        team_id_i = int(mine_base.get('teamId') or 0)
        mine_edge: Dict[str, Optional[float]] = {}
        for mid in metric_ids:
            mine_edge[mid] = _compute_metric(mid, mine_base, team_id_i)

        edge_rank_map = (mine_base.get('_edgeRank') or {}) if isinstance(mine_base, dict) else {}
        edge_avg_map = (mine_base.get('_edgeAvg') or {}) if isinstance(mine_base, dict) else {}
        out_metrics_edge: Dict[str, Any] = {}
        for mid in metric_ids:
            out_metrics_edge[mid] = {
                'value': mine_edge.get(mid),
                'pct': special_pct.get(mid),
                'rank': edge_rank_map.get(mid),
                'avg': edge_avg_map.get(mid),
            }

        payload = {
            'team': team_ab,
            'teamId': team_id_i,
            'season': int(season_int) if scope == 'season' else None,
            'scope': scope,
            'seasonState': season_state,
            'strengthState': strength_state,
            'rates': rates,
            'metrics': out_metrics_edge,
        }
        if want_debug and isinstance(debug_meta, dict):
            payload['debug'] = debug_meta
            payload['debug']['baseTeamsCount'] = int(len(base or {}))
        j = jsonify(payload)
        try:
            j.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
        return j

    # Compute derived metrics for all teams (for percentile pools).
    derived: Dict[str, Dict[str, Optional[float]]] = {}
    for ab, v in base.items():
        tid = int(v.get('teamId') or 0)
        per_team: Dict[str, Optional[float]] = {}
        for mid in metric_ids:
            per_team[mid] = _compute_metric(mid, v, tid)
        derived[ab] = per_team

    def _percentile(values: List[float], v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        if not values:
            return None
        import bisect
        arr = sorted(values)
        idx = bisect.bisect_right(arr, float(v))
        return 100.0 * (idx / float(len(arr)))

    # Build distributions excluding Edge (Edge percentiles come from rank).
    dist_all: Dict[str, List[float]] = {mid: [] for mid in metric_ids}
    for _ab, mm in derived.items():
        for mid in metric_ids:
            if mid in edge_metric_ids:
                continue
            vv = mm.get(mid)
            if vv is None:
                continue
            try:
                fv = float(vv)
                if not math.isfinite(fv):
                    continue
                dist_all[mid].append(fv)
            except Exception:
                continue

    mine = derived.get(team_ab) or {}

    def _lower_is_better_team(metric_id: str) -> bool:
        m = metric_id
        if '|' in metric_id:
            _, m = metric_id.split('|', 1)
        m = str(m or '').strip()
        return m in {
            'CA', 'FA', 'SA', 'GA', 'xGA',
        }

    edge_rank_map = (mine_base.get('_edgeRank') or {}) if isinstance(mine_base, dict) else {}
    edge_avg_map = (mine_base.get('_edgeAvg') or {}) if isinstance(mine_base, dict) else {}

    out_metrics: Dict[str, Any] = {}
    for mid in metric_ids:
        val = mine.get(mid)
        if mid in special_pct:
            pct = special_pct[mid]
        elif mid not in edge_metric_ids:
            pool = dist_all.get(mid) or []
            pct = _percentile(pool, val)
            if pct is not None and _lower_is_better_team(mid):
                pct = 100.0 - pct
        else:
            pct = None
        mm: Dict[str, Any] = {'value': val, 'pct': pct}
        if mid in edge_metric_ids:
            mm['rank'] = edge_rank_map.get(mid)
            mm['avg'] = edge_avg_map.get(mid)
        out_metrics[mid] = mm

    payload = {
        'team': team_ab,
        'teamId': int(mine_base.get('teamId') or 0),
        'season': int(season_int) if scope == 'season' else None,
        'seasons': season_ids if scope == 'season' else [],
        'scope': scope,
        'seasonState': season_state,
        'strengthState': strength_state,
        'rates': rates,
        'metrics': out_metrics,
    }
    if want_debug and isinstance(debug_meta, dict):
        payload['debug'] = debug_meta
        payload['debug']['baseTeamsCount'] = int(len(base or {}))
    j = jsonify(payload)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/teams/table', methods=['GET', 'POST'])
def api_teams_table():
    """Bulk team table metrics for the selected season/scope.

    Query params:
      season=<int>
      seasonState=regular|playoffs|all
      strengthState=5v5|PP|SH|Other|all
      rates=Totals|PerGame
      scope=season|total
      metricIds=<comma separated Category|Metric>
      includeHistoric=0|1

    Notes:
      - Does not compute NHL Edge metrics in bulk by default (those require upstream calls).
    """
    body: Optional[Dict[str, Any]] = None
    try:
        if request.method == 'POST':
            maybe = request.get_json(silent=True)
            if isinstance(maybe, dict):
                body = maybe
    except Exception:
        body = None

    def _get(key: str, default: Any = None) -> Any:
        try:
            if isinstance(body, dict) and key in body and body.get(key) is not None:
                return body.get(key)
        except Exception:
            pass
        return request.args.get(key, default)

    season = str(_get('season') or '').strip()
    season_state = str(_get('seasonState', 'regular') or 'regular').strip().lower()
    strength_state = str(_get('strengthState', '5v5') or '5v5').strip()
    rates = str(_get('rates') or 'Totals').strip() or 'Totals'
    scope = str(_get('scope', 'season') or 'season').strip().lower()
    include_historic = str(_get('includeHistoric') or _get('include_historic') or '').strip()
    metric_ids_val = _get('metricIds') or _get('metrics')

    season_ids = _parse_request_season_ids(season, default=20252026)
    season_int = _primary_season_id(season_ids, default=20252026) or 20252026

    if season_state not in {'regular', 'playoffs', 'all'}:
        season_state = 'regular'
    if strength_state not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        strength_state = '5v5'
    if rates not in {'Totals', 'PerGame'}:
        rates = 'Totals'
    if scope not in {'season', 'total'}:
        scope = 'season'

    metric_ids: List[str] = []
    if isinstance(metric_ids_val, list):
        metric_ids = [str(s).strip() for s in metric_ids_val if s is not None and str(s).strip()]
    else:
        metric_ids_raw = str(metric_ids_val or '').strip()
        if metric_ids_raw:
            metric_ids = [s.strip() for s in metric_ids_raw.split(',') if s and s.strip()]
    if not metric_ids:
        defs0 = _load_card_metrics_defs('teams')
        metric_ids = [str(m.get('id')) for m in (defs0.get('metrics') or []) if isinstance(m, dict) and m.get('id')]

    base = _build_team_base_stats(
        scope=scope,
        season_int=season_int,
        season_ids=season_ids,
        season_state=season_state,
        strength_state=strength_state,
    )

    # Filter historic teams (based on Teams.csv Active flag) unless requested.
    show_hist = include_historic in {'1', 'true', 'True', 'yes', 'YES'}
    if not show_hist:
        active = {str(r.get('Team') or '').strip().upper() for r in (TEAM_ROWS or []) if str(r.get('Active') or '').strip() == '1'}
        base = {ab: v for ab, v in base.items() if ab in active}

    def _pct(n: Optional[float], d: Optional[float]) -> Optional[float]:
        try:
            if n is None or d is None:
                return None
            if d <= 0:
                return None
            return 100.0 * (float(n) / float(d))
        except Exception:
            return None

    def _rate_from(gp: float, vv: Optional[float]) -> Optional[float]:
        if rates == 'Totals':
            return vv
        if gp <= 0:
            return None
        try:
            return float(vv) / float(gp) if vv is not None else None
        except Exception:
            return None

    def _compute_non_edge(metric_id: str, v: Dict[str, Any]) -> Optional[float]:
        gp = float(v.get('GP') or 0.0)
        cf = float(v.get('CF') or 0.0)
        ca = float(v.get('CA') or 0.0)
        ff = float(v.get('FF') or 0.0)
        fa = float(v.get('FA') or 0.0)
        sf = float(v.get('SF') or 0.0)
        sa = float(v.get('SA') or 0.0)
        gf = float(v.get('GF') or 0.0)
        ga = float(v.get('GA') or 0.0)
        xgf = v.get('xGF')
        xga = v.get('xGA')

        category = None
        metric = None
        if '|' in metric_id:
            category, metric = metric_id.split('|', 1)
        else:
            metric = metric_id

        if category == 'Edge':
            return None

        if metric == 'CA':
            return _rate_from(gp, ca)
        if metric == 'FA':
            return _rate_from(gp, fa)
        if metric == 'SA':
            return _rate_from(gp, sa)
        if metric == 'GA':
            return _rate_from(gp, ga)
        if metric == 'xGA':
            return _rate_from(gp, float(xga) if xga is not None else None)

        if metric == 'CF':
            return _rate_from(gp, cf)
        if metric == 'FF':
            return _rate_from(gp, ff)
        if metric == 'SF':
            return _rate_from(gp, sf)
        if metric == 'GF':
            return _rate_from(gp, gf)
        if metric == 'xGF':
            return _rate_from(gp, float(xgf) if xgf is not None else None)

        if metric == 'CF%':
            return _pct(cf, (cf + ca))
        if metric == 'FF%':
            return _pct(ff, (ff + fa))
        if metric == 'SF%':
            return _pct(sf, (sf + sa))
        if metric == 'GF%':
            return _pct(gf, (gf + ga))
        if metric == 'xGF%':
            if xgf is None or xga is None:
                return None
            return _pct(float(xgf), float(xgf) + float(xga))

        if metric == 'C+/-':
            return _rate_from(gp, (cf - ca))
        if metric == 'F+/-':
            return _rate_from(gp, (ff - fa))
        if metric == 'S+/-':
            return _rate_from(gp, (sf - sa))
        if metric == 'G+/-':
            return _rate_from(gp, (gf - ga))
        if metric == 'xG+/-':
            if xgf is None or xga is None:
                return None
            return _rate_from(gp, (float(xgf) - float(xga)))

        if category == 'Context' and metric == 'Sh%':
            return _pct(gf, sf)
        if category == 'Context' and metric == 'Sv%':
            if sa <= 0:
                return 100.0 if ga <= 0 else 0.0
            return 100.0 * (1.0 - (ga / sa))
        if category == 'Context' and metric == 'PDO':
            sh_oi = _pct(gf, sf)
            sv_oi = 100.0 if sa <= 0 and ga <= 0 else (0.0 if sa <= 0 else 100.0 * (1.0 - (ga / sa)))
            return (sh_oi + sv_oi) if (sh_oi is not None and sv_oi is not None) else None
        if category == 'Context' and metric == 'GAx':
            if xgf is None:
                return None
            return _rate_from(gp, (gf - float(xgf)))
        if category == 'Context' and metric == 'GSAx':
            if xga is None:
                return None
            return _rate_from(gp, (float(xga) - ga))

        return None

    rows_out: List[Dict[str, Any]] = []
    for ab, v in base.items():
        d = {
            'team': ab,
            'teamId': int(v.get('teamId') or 0),
        }
        for mid in metric_ids:
            d[mid] = _compute_non_edge(mid, v)
        rows_out.append(d)

    # Stable ordering: team name.
    rows_out.sort(key=lambda x: str(x.get('team') or ''))
    payload = {
        'season': int(season_int) if scope == 'season' else None,
        'seasons': season_ids if scope == 'season' else [],
        'scope': scope,
        'seasonState': season_state,
        'strengthState': strength_state,
        'rates': rates,
        'metricIds': metric_ids,
        'rows': rows_out,
    }
    j = jsonify(payload)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/teams/scatter')
def api_teams_scatter():
    """League-wide scatter data for the Teams 'Charts' tab.

    Query params:
      season=<int>
      seasonState=regular|playoffs|all
      rates=Totals|PerGame
      scope=season|total
      includeHistoric=0|1
      xMetricId=<Category|Metric>
      yMetricId=<Category|Metric>

    Notes:
      - Uses NHL stats REST aggregates (non-Edge metrics only).
      - Does NOT support NHL Edge metrics.
    """
    season = str(request.args.get('season') or '').strip()
    season_state = str(request.args.get('seasonState') or 'regular').strip().lower()
    strength_state = str(request.args.get('strengthState') or '5v5').strip()
    rates = str(request.args.get('rates') or 'Totals').strip() or 'Totals'
    scope = str(request.args.get('scope') or 'season').strip().lower()
    include_historic = str(request.args.get('includeHistoric') or request.args.get('include_historic') or '').strip()
    x_metric_id = str(request.args.get('xMetricId') or request.args.get('xMetric') or '').strip()
    y_metric_id = str(request.args.get('yMetricId') or request.args.get('yMetric') or '').strip()

    season_ids = _parse_request_season_ids(season, default=20252026)
    season_int = _primary_season_id(season_ids, default=20252026) or 20252026

    if season_state not in {'regular', 'playoffs', 'all'}:
        season_state = 'regular'
    if strength_state not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        strength_state = '5v5'
    if rates not in {'Totals', 'PerGame'}:
        rates = 'Totals'
    if scope not in {'season', 'total'}:
        scope = 'season'

    if not x_metric_id or not y_metric_id:
        return jsonify({'error': 'missing_metric', 'hint': 'Provide xMetricId and yMetricId'}), 400
    if str(x_metric_id).startswith('Edge|') or str(y_metric_id).startswith('Edge|'):
        return jsonify({'error': 'edge_not_supported'}), 400

    base = _build_team_base_stats(
        scope=scope,
        season_int=season_int,
        season_ids=season_ids,
        season_state=season_state,
        strength_state=strength_state,
    )

    # Filter historic teams (based on Teams.csv Active flag) unless requested.
    show_hist = include_historic in {'1', 'true', 'True', 'yes', 'YES'}
    if not show_hist:
        active = {str(r.get('Team') or '').strip().upper() for r in (TEAM_ROWS or []) if str(r.get('Active') or '').strip() == '1'}
        base = {ab: v for ab, v in base.items() if ab in active}

    team_name_by_ab = {str(r.get('Team') or '').strip().upper(): str(r.get('Name') or '').strip() for r in (TEAM_ROWS or [])}

    def _pct(n: Optional[float], d: Optional[float]) -> Optional[float]:
        try:
            if n is None or d is None:
                return None
            if d <= 0:
                return None
            return 100.0 * (float(n) / float(d))
        except Exception:
            return None

    def _rate_from(gp: float, vv: Optional[float]) -> Optional[float]:
        if rates == 'Totals':
            return vv
        if gp <= 0:
            return None
        try:
            return float(vv) / float(gp) if vv is not None else None
        except Exception:
            return None

    def _compute_non_edge(metric_id: str, v: Dict[str, Any]) -> Optional[float]:
        gp = float(v.get('GP') or 0.0)
        cf = float(v.get('CF') or 0.0)
        ca = float(v.get('CA') or 0.0)
        ff = float(v.get('FF') or 0.0)
        fa = float(v.get('FA') or 0.0)
        sf = float(v.get('SF') or 0.0)
        sa = float(v.get('SA') or 0.0)
        gf = float(v.get('GF') or 0.0)
        ga = float(v.get('GA') or 0.0)
        xgf = v.get('xGF')
        xga = v.get('xGA')

        category = None
        metric = None
        if '|' in metric_id:
            category, metric = metric_id.split('|', 1)
        else:
            metric = metric_id

        if category == 'Edge':
            return None

        if metric == 'CA':
            return _rate_from(gp, ca)
        if metric == 'FA':
            return _rate_from(gp, fa)
        if metric == 'SA':
            return _rate_from(gp, sa)
        if metric == 'GA':
            return _rate_from(gp, ga)
        if metric == 'xGA':
            return _rate_from(gp, float(xga) if xga is not None else None)

        if metric == 'CF':
            return _rate_from(gp, cf)
        if metric == 'FF':
            return _rate_from(gp, ff)
        if metric == 'SF':
            return _rate_from(gp, sf)
        if metric == 'GF':
            return _rate_from(gp, gf)
        if metric == 'xGF':
            return _rate_from(gp, float(xgf) if xgf is not None else None)

        if metric == 'CF%':
            return _pct(cf, (cf + ca))
        if metric == 'FF%':
            return _pct(ff, (ff + fa))
        if metric == 'SF%':
            return _pct(sf, (sf + sa))
        if metric == 'GF%':
            return _pct(gf, (gf + ga))
        if metric == 'xGF%':
            if xgf is None or xga is None:
                return None
            return _pct(float(xgf), float(xgf) + float(xga))

        if metric == 'C+/-':
            return _rate_from(gp, (cf - ca))
        if metric == 'F+/-':
            return _rate_from(gp, (ff - fa))
        if metric == 'S+/-':
            return _rate_from(gp, (sf - sa))
        if metric == 'G+/-':
            return _rate_from(gp, (gf - ga))
        if metric == 'xG+/-':
            if xgf is None or xga is None:
                return None
            return _rate_from(gp, (float(xgf) - float(xga)))

        if category == 'Context' and metric == 'Sh%':
            return _pct(gf, sf)
        if category == 'Context' and metric == 'Sv%':
            if sa <= 0:
                return 100.0 if ga <= 0 else 0.0
            return 100.0 * (1.0 - (ga / sa))
        if category == 'Context' and metric == 'PDO':
            sh_oi = _pct(gf, sf)
            sv_oi = 100.0 if sa <= 0 and ga <= 0 else (0.0 if sa <= 0 else 100.0 * (1.0 - (ga / sa)))
            return (sh_oi + sv_oi) if (sh_oi is not None and sv_oi is not None) else None
        if category == 'Context' and metric == 'GAx':
            if xgf is None:
                return None
            return _rate_from(gp, (gf - float(xgf)))
        if category == 'Context' and metric == 'GSAx':
            if xga is None:
                return None
            return _rate_from(gp, (float(xga) - ga))

        return None

    points: List[Dict[str, Any]] = []
    for ab, v in base.items():
        x = _compute_non_edge(x_metric_id, v)
        y = _compute_non_edge(y_metric_id, v)
        try:
            fx = float(x) if x is not None else None
            fy = float(y) if y is not None else None
        except Exception:
            fx = None
            fy = None
        if fx is None or fy is None:
            continue
        if not (math.isfinite(fx) and math.isfinite(fy)):
            continue
        points.append({
            'team': ab,
            'name': team_name_by_ab.get(ab) or ab,
            'x': fx,
            'y': fy,
        })

    payload = {
        'season': int(season_int) if scope == 'season' else None,
        'seasons': season_ids if scope == 'season' else [],
        'scope': scope,
        'seasonState': season_state,
        'rates': rates,
        'xMetricId': x_metric_id,
        'yMetricId': y_metric_id,
        'points': points,
    }
    j = jsonify(payload)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/goalies/series')
def api_goalies_series():
    """Per-season GSAA/GSAx series for a single goalie.

    Query params:
      playerId=<int>
      seasonState=regular|playoffs|all
      strengthState=5v5|PP|SH|Other|all
      xgModel=xG_S|xG_F|xG_F2

    Notes:
      - GSAA is computed season-by-season using that season's league-average Sv% baseline.
      - GSAx is 0 before 20102011.
    """
    player_id_q = str(request.args.get('playerId') or request.args.get('player_id') or '').strip()
    season_state = str(request.args.get('seasonState') or 'regular').strip().lower()
    strength_state = str(request.args.get('strengthState') or '5v5').strip()
    xg_model = str(request.args.get('xgModel') or 'xG_F').strip()

    pid = _safe_int(player_id_q)
    if not pid or pid <= 0:
        return jsonify({'error': 'missing_playerId'}), 400

    if season_state not in {'regular', 'playoffs', 'all'}:
        season_state = 'regular'
    if strength_state not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        strength_state = '5v5'
    if xg_model not in {'xG_S', 'xG_F', 'xG_F2'}:
        xg_model = 'xG_F'

    by_pid_season, league_sa_ga = _build_goalies_career_season_matrix(
        season_state=season_state,
        strength_state=strength_state,
    )

    seasons = by_pid_season.get(int(pid)) or {}

    def _goalie_team_map(pid_i: int, ss: str) -> Dict[int, str]:
        """Best-effort {seasonId -> teamAbbrev} from NHL Stats API goalie summary (cached)."""
        try:
            ttl_s = max(60, int(os.getenv('GOALIES_TEAM_BY_SEASON_CACHE_TTL_SECONDS', str(7 * 86400)) or str(7 * 86400)))
        except Exception:
            ttl_s = 7 * 86400
        try:
            max_items = max(1, int(os.getenv('GOALIES_TEAM_BY_SEASON_MAP_CACHE_MAX_ITEMS', '128') or '128'))
        except Exception:
            max_items = 128
        ss_norm = (ss or 'regular').strip().lower()
        if ss_norm not in {'regular', 'playoffs', 'all'}:
            ss_norm = 'regular'
        ck = (int(pid_i), ss_norm)
        now2 = time.time()
        _cache_prune_ttl_and_size(_GOALIES_TEAM_BY_SEASON_MAP_CACHE, ttl_s=ttl_s, max_items=max_items)
        cached = _GOALIES_TEAM_BY_SEASON_MAP_CACHE.get(ck)
        if cached and (now2 - float(cached[0])) < float(ttl_s):
            return cached[1] or {}

        def _toi_to_seconds(x: Any) -> int:
            try:
                if isinstance(x, (int, float)):
                    return int(x)
                s = str(x or '').strip()
                if not s:
                    return 0
                if ':' in s:
                    parts = s.split(':')
                    if len(parts) == 2:
                        return int(parts[0]) * 60 + int(parts[1])
                    if len(parts) == 3:
                        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                return int(float(s))
            except Exception:
                return 0

        # Fetch from NHL stats API.
        try:
            if ss_norm == 'regular':
                cay = f'gameTypeId=2 and playerId={int(pid_i)}'
            elif ss_norm == 'playoffs':
                cay = f'gameTypeId=3 and playerId={int(pid_i)}'
            else:
                cay = f'(gameTypeId=2 or gameTypeId=3) and playerId={int(pid_i)}'
            url = 'https://api.nhle.com/stats/rest/en/goalie/summary'
            r = requests.get(
                url,
                params={'limit': -1, 'start': 0, 'cayenneExp': cay},
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=20,
                allow_redirects=True,
            )
            if r.status_code == 200:
                data = r.json() if r.content else {}
                rows = data.get('data') if isinstance(data, dict) else None
                if isinstance(rows, list) and rows:
                    best: Dict[int, Tuple[int, str]] = {}
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        sid = row.get('seasonId')
                        season_id = _safe_int(sid)
                        if not season_id:
                            continue
                        season_id_i = int(season_id)

                        team_raw = row.get('teamAbbrev') or row.get('teamAbbrevs') or row.get('currentTeamAbbrev') or ''
                        team_abbrev = ''
                        if isinstance(team_raw, list) and team_raw:
                            team_abbrev = str(team_raw[0] or '').strip().upper()
                        else:
                            team_abbrev = str(team_raw or '').strip().upper()
                        if '/' in team_abbrev:
                            team_abbrev = team_abbrev.split('/')[0].strip().upper()
                        if not team_abbrev:
                            continue

                        gp = row.get('gamesPlayed') or row.get('games') or 0
                        toi = row.get('timeOnIce') or row.get('toi') or 0
                        weight = 0
                        try:
                            weight = int(gp) * 100000 + _toi_to_seconds(toi)
                        except Exception:
                            weight = _toi_to_seconds(toi)

                        prev = best.get(season_id_i)
                        if not prev or weight > int(prev[0]):
                            best[season_id_i] = (int(weight), team_abbrev)

                    team_map: Dict[int, str] = {sid: t for sid, (_, t) in best.items()}
                    _cache_set_multi_bounded(_GOALIES_TEAM_BY_SEASON_MAP_CACHE, ck, team_map, ttl_s=ttl_s, max_items=max_items)
                    return team_map
        except Exception:
            pass

        try:
            _cache_set_multi_bounded(_GOALIES_TEAM_BY_SEASON_MAP_CACHE, ck, {}, ttl_s=ttl_s, max_items=max_items)
        except Exception:
            pass
        return {}

    def _goalie_primary_team_for_season(pid_i: int, season_i: int, ss: str) -> str:
        try:
            m = _goalie_team_map(pid_i, ss)
            return str(m.get(int(season_i)) or '').strip().upper()
        except Exception:
            return ''

    def _xga(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('xGA_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('xGA_F2') or 0.0)
        return float(v.get('xGA_S') or 0.0)

    def _sv_frac(ga: float, att: float) -> float:
        if att <= 0:
            return 1.0 if ga <= 0 else 0.0
        return 1.0 - (ga / att)

    out: List[Dict[str, Any]] = []
    for season_id in sorted(int(s) for s in seasons.keys()):
        v = seasons.get(int(season_id)) or {}
        try:
            sa = float(v.get('SA') or 0.0)
            ga = float(v.get('GA') or 0.0)
        except Exception:
            sa = 0.0
            ga = 0.0

        tot_sa, tot_ga = league_sa_ga.get(int(season_id), (0.0, 0.0))
        avg_sv_s = _sv_frac(float(tot_ga or 0.0), float(tot_sa or 0.0))
        sv_s = _sv_frac(float(ga or 0.0), float(sa or 0.0))
        gsaa = (sv_s - avg_sv_s) * float(sa or 0.0)

        gsax = 0.0
        if int(season_id) >= 20102011:
            try:
                xga = float(_xga(v) or 0.0)
                gsax = float(xga or 0.0) - float(ga or 0.0)
            except Exception:
                gsax = 0.0

        out.append({
            'season': int(season_id),
            'team': _goalie_primary_team_for_season(int(pid), int(season_id), season_state),
            'GSAA': float(gsaa),
            'GSAx': float(gsax),
        })

    j = jsonify({
        'playerId': int(pid),
        'seasonState': season_state,
        'strengthState': strength_state,
        'xgModel': xg_model,
        'seasons': out,
    })
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/skaters/table', methods=['GET', 'POST'])
def api_skaters_table():
    """Bulk table metrics for a set of playerIds using the same slicers as the Card tab.

    Query params:
      season=<int>
      seasonState=regular|playoffs|all
      strengthState=5v5|PP|SH|Other|all
      xgModel=xG_S|xG_F|xG_F2
      rates=Totals|Per60|PerGame
      scope=season|career
      minGP=<int>
      minTOI=<float minutes>
      playerIds=<comma separated ints>
      metricIds=<comma separated Category|Metric ids>

    Notes:
      - Computes SeasonStats + RAPM + Context (QoT/QoC/ZS).
      - Does NOT compute NHL Edge metrics in bulk (those require per-player upstream calls).
    """
    body: Optional[Dict[str, Any]] = None
    try:
        if request.method == 'POST':
            maybe = request.get_json(silent=True)
            if isinstance(maybe, dict):
                body = maybe
    except Exception:
        body = None

    def _get(key: str, default: Any = None) -> Any:
        try:
            if isinstance(body, dict) and key in body and body.get(key) is not None:
                return body.get(key)
        except Exception:
            pass
        return request.args.get(key, default)

    season = str(_get('season') or '').strip()
    season_state = str(_get('seasonState', 'regular') or 'regular').strip().lower()
    strength_state = str(_get('strengthState', '5v5') or '5v5').strip()
    xg_model = str(_get('xgModel', 'xG_F') or 'xG_F').strip()
    rates = str(_get('rates') or _get('ratesTotals') or 'Totals').strip() or 'Totals'
    scope = str(_get('scope', 'season') or 'season').strip().lower()

    metric_ids_val = _get('metricIds') or _get('metrics')
    player_ids_val = _get('playerIds') or _get('player_ids')

    min_gp = _safe_int(_get('minGP') or _get('minGp') or _get('min_gp') or 0) or 0
    min_toi_raw = _get('minTOI') or _get('minToi') or _get('min_toi') or 0
    try:
        min_toi = float(_parse_locale_float(min_toi_raw) or 0.0)
    except Exception:
        min_toi = 0.0
    if min_gp < 0:
        min_gp = 0
    if min_toi < 0:
        min_toi = 0.0

    player_ids: List[int] = []
    if isinstance(player_ids_val, list):
        for v in player_ids_val:
            pid_i = _safe_int(v)
            if pid_i and pid_i > 0:
                player_ids.append(int(pid_i))
    else:
        player_ids_raw = str(player_ids_val or '').strip()
        if not player_ids_raw:
            return jsonify({'error': 'missing_playerIds'}), 400
        for part in player_ids_raw.split(','):
            part = str(part or '').strip()
            if not part:
                continue
            pid_i = _safe_int(part)
            if pid_i and pid_i > 0:
                player_ids.append(int(pid_i))
    # De-dupe while preserving order
    seen: set[int] = set()
    player_ids = [pid for pid in player_ids if not (pid in seen or seen.add(pid))]
    if not player_ids:
        return jsonify({'error': 'empty_playerIds'}), 400

    season_ids = _parse_request_season_ids(season, default=20252026)
    season_int = _primary_season_id(season_ids, default=20252026) or 20252026

    if season_state not in {'regular', 'playoffs', 'all'}:
        season_state = 'regular'
    if strength_state not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        strength_state = '5v5'
    if xg_model not in {'xG_S', 'xG_F', 'xG_F2'}:
        xg_model = 'xG_F'
    if rates not in {'Totals', 'Per60', 'PerGame'}:
        rates = 'Totals'
    if scope not in {'season', 'career'}:
        scope = 'season'

    metric_ids: List[str] = []
    if isinstance(metric_ids_val, list):
        metric_ids = [str(s).strip() for s in metric_ids_val if s is not None and str(s).strip()]
    else:
        metric_ids_raw = str(metric_ids_val or '').strip()
        if metric_ids_raw:
            metric_ids = [s.strip() for s in metric_ids_raw.split(',') if s and s.strip()]
    if not metric_ids:
        # Best-effort defaults: use card default metrics in definition order.
        defs0 = _load_card_metrics_defs()
        for m in (defs0.get('metrics') or []):
            try:
                if isinstance(m, dict) and m.get('default') and m.get('id'):
                    metric_ids.append(str(m.get('id')))
            except Exception:
                continue

    # Aggregate by player under the requested filters (cached).
    agg, _pos_group_by_pid = _build_seasonstats_agg(
        scope=scope,
        season_int=season_int,
        season_ids=season_ids,
        season_state=season_state,
        strength_state=strength_state,
    )

    # Apply min requirements.
    if min_gp > 0 or min_toi > 0:
        eligible = {
            pid_k
            for pid_k, d in agg.items()
            if float(d.get('GP') or 0) >= float(min_gp) and float(d.get('TOI') or 0.0) >= float(min_toi)
        }
        agg = {pid_k: d for pid_k, d in agg.items() if pid_k in eligible}

    # Only keep requested playerIds that are present after filtering.
    # This makes minGP/minTOI (and missing SeasonStats) actually remove rows.
    try:
        eligible_pids = set(int(k) for k in agg.keys())
        player_ids = [int(pid) for pid in player_ids if int(pid) in eligible_pids]
    except Exception:
        pass

    # Determine RAPM/Context needs.
    needs_rapm = any(('|RAPM ' in mid) for mid in metric_ids)
    needs_ctx = any((mid in {'Context|QoT', 'Context|QoC', 'Context|ZS'}) for mid in metric_ids)

    want_pid_set: set[int] = set(int(pid) for pid in player_ids)

    def _norm_rates_totals(v: Any) -> str:
        s = str(v or '').strip().lower()
        if s.startswith('tot'):
            return 'Totals'
        if s.startswith('rate'):
            return 'Rates'
        return str(v or '').strip() or 'Rates'

    want_strength = strength_state if strength_state in {'5v5', 'PP', 'SH'} else '5v5'
    want_rapm_rates = 'Totals' if rates == 'Totals' else 'Rates'

    rapm_by_pid: Dict[int, List[Dict[str, Any]]] = {}
    if needs_rapm:
        rapm_rows = _load_rapm_static_csv()

        for r in rapm_rows:
            try:
                if not _row_season_in_selected(r, season_ids):
                    continue
                pid_i = _safe_int(r.get('PlayerID'))
                if not pid_i or pid_i <= 0:
                    continue
                if int(pid_i) not in want_pid_set:
                    continue
                rapm_by_pid.setdefault(int(pid_i), []).append(r)
            except Exception:
                continue

    ctx_by_pid: Dict[int, List[Dict[str, Any]]] = {}
    if needs_ctx:
        ctx_rows = _load_context_static_csv()

        for r in ctx_rows:
            try:
                if not _row_season_in_selected(r, season_ids):
                    continue
                pid_i = _safe_int(r.get('PlayerID'))
                if not pid_i or pid_i <= 0:
                    continue
                if int(pid_i) not in want_pid_set:
                    continue
                ctx_by_pid.setdefault(int(pid_i), []).append(r)
            except Exception:
                continue

    def _pick_rapm_row(pid_i: int) -> Optional[Dict[str, Any]]:
        rows = rapm_by_pid.get(int(pid_i)) or []
        if not rows:
            return None
        # Prefer exact strength + rates.
        for r in rows:
            try:
                if str(r.get('StrengthState') or '').strip() == want_strength and _norm_rates_totals(r.get('Rates_Totals') or r.get('Rates/Totals') or r.get('RatesTotals')) == want_rapm_rates:
                    return r
            except Exception:
                continue
        # Fallback by rates.
        for r in rows:
            try:
                if _norm_rates_totals(r.get('Rates_Totals') or r.get('Rates/Totals') or r.get('RatesTotals')) == want_rapm_rates:
                    return r
            except Exception:
                continue
        return rows[0]

    def _pick_ctx_row(pid_i: int) -> Optional[Dict[str, Any]]:
        rows = ctx_by_pid.get(int(pid_i)) or []
        if not rows:
            return None
        for r in rows:
            try:
                if str(r.get('StrengthState') or '').strip() == want_strength:
                    return r
            except Exception:
                continue
        for r in rows:
            try:
                if str(r.get('StrengthState') or '').strip() == '5v5':
                    return r
            except Exception:
                continue
        return rows[0]

    def _pct(n: Optional[float], d: Optional[float]) -> Optional[float]:
        try:
            if n is None or d is None:
                return None
            if d <= 0:
                return None
            return 100.0 * (float(n) / float(d))
        except Exception:
            return None

    def _attempts(v: Dict[str, Any]) -> float:
        vv = v.get('iShots') if xg_model == 'xG_S' else v.get('iFenwick')
        return float(vv or 0.0)

    def _ixg(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('ixG_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('ixG_F2') or 0.0)
        return float(v.get('ixG_S') or 0.0)

    def _xgf(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('xGF_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('xGF_F2') or 0.0)
        return float(v.get('xGF_S') or 0.0)

    def _xga(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('xGA_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('xGA_F2') or 0.0)
        return float(v.get('xGA_S') or 0.0)

    def _rate_from(gp: float, toi: float, vv: Optional[float]) -> Optional[float]:
        if rates == 'Totals':
            return vv
        denom = None
        if rates == 'PerGame':
            denom = gp if gp > 0 else None
        elif rates == 'Per60':
            denom = (toi / 60.0) if toi > 0 else None
        if vv is None or denom is None or denom <= 0:
            return None
        try:
            return float(vv) / float(denom)
        except Exception:
            return None

    def _compute_metric(metric_id: str, v: Optional[Dict[str, Any]], pid_i: int) -> Optional[float]:
        if v is None:
            return None
        gp = float(v.get('GP') or 0.0)
        toi = float(v.get('TOI') or 0.0)
        igoals = float(v.get('iGoals') or 0.0)
        a1 = float(v.get('Assists1') or 0.0)
        a2 = float(v.get('Assists2') or 0.0)
        pts = igoals + a1 + a2
        att = _attempts(v)
        ixg = _ixg(v)

        cf = float(v.get('CF') or 0.0)
        ca = float(v.get('CA') or 0.0)
        ff = float(v.get('FF') or 0.0)
        fa = float(v.get('FA') or 0.0)
        sf = float(v.get('SF') or 0.0)
        sa = float(v.get('SA') or 0.0)
        gf = float(v.get('GF') or 0.0)
        ga = float(v.get('GA') or 0.0)
        xgf = _xgf(v)
        xga = _xga(v)

        pim_taken = float(v.get('PIM_taken') or 0.0)
        pim_drawn = float(v.get('PIM_drawn') or 0.0)
        pim_for = float(v.get('PIM_for') or 0.0)
        pim_against = float(v.get('PIM_against') or 0.0)
        hits = float(v.get('Hits') or 0.0)
        takeaways = float(v.get('Takeaways') or 0.0)
        giveaways = float(v.get('Giveaways') or 0.0)

        category = None
        metric = None
        if '|' in metric_id:
            category, metric = metric_id.split('|', 1)
        else:
            metric = metric_id

        # NHL Edge is not computed in bulk.
        if category == 'Edge':
            return None

        # RAPM + Context from preloaded maps
        if metric and str(metric).startswith('RAPM '):
            row = _pick_rapm_row(pid_i)
            if not row:
                return None
            base = str(metric).replace('RAPM', '', 1).strip()
            col = None
            if base in {'CF', 'CA', 'GF', 'GA', 'xGF', 'xGA'}:
                col = base
            elif base == 'C+/-':
                col = 'C_plusminus'
            elif base == 'G+/-':
                col = 'G_plusminus'
            elif base == 'xG+/-':
                col = 'xG_plusminus'
            if not col:
                return None
            val = _parse_locale_float(row.get(col))
            return float(val) if val is not None else None

        if category == 'Context' and metric in {'QoT', 'QoC', 'ZS'}:
            row = _pick_ctx_row(pid_i)
            if not row:
                return None
            col2 = None
            if metric == 'QoT':
                col2 = 'QoT_blend_xG67_G33'
            elif metric == 'QoC':
                col2 = 'QoC_blend_xG67_G33'
            elif metric == 'ZS':
                col2 = 'ZS_Difficulty'
            val2 = _parse_locale_float(row.get(col2)) if col2 else None
            return float(val2) if val2 is not None else None

        if metric == 'GP':
            return gp
        if metric == 'TOI':
            return toi

        if metric == 'iGoals':
            return _rate_from(gp, toi, igoals)
        if metric == 'Assists1':
            return _rate_from(gp, toi, a1)
        if metric == 'Assists2':
            return _rate_from(gp, toi, a2)
        if metric == 'Points':
            return _rate_from(gp, toi, pts)

        if metric in {'iShots', 'iFenwick', 'iShots or iFenwick'}:
            vv = float(v.get('iShots') or 0.0) if xg_model == 'xG_S' else float(v.get('iFenwick') or 0.0)
            return _rate_from(gp, toi, vv)

        if metric in {'ixG', 'Individual xG'}:
            return _rate_from(gp, toi, ixg)

        if category == 'Shooting' and metric in {'Sh% or FSh%', 'Sh%'}:
            return _pct(igoals, att)
        if category == 'Shooting' and metric in {'xSh% or xFS%', 'xSh% or xFSh%', 'xSh%'}:
            return _pct(ixg, att)
        if category == 'Shooting' and metric in {'dSh% or dFSh%'}:
            sh = _pct(igoals, att)
            xsh = _pct(ixg, att)
            return (sh - xsh) if (sh is not None and xsh is not None) else None
        if metric == 'GAx' and category == 'Shooting':
            return _rate_from(gp, toi, (igoals - ixg))

        # On-ice totals
        if metric == 'CF':
            return _rate_from(gp, toi, cf)
        if metric == 'CA':
            return _rate_from(gp, toi, ca)
        if metric == 'FF':
            return _rate_from(gp, toi, ff)
        if metric == 'FA':
            return _rate_from(gp, toi, fa)
        if metric == 'SF':
            return _rate_from(gp, toi, sf)
        if metric == 'SA':
            return _rate_from(gp, toi, sa)
        if metric == 'GF':
            return _rate_from(gp, toi, gf)
        if metric == 'GA':
            return _rate_from(gp, toi, ga)
        if metric == 'xGF':
            return _rate_from(gp, toi, xgf)
        if metric == 'xGA':
            return _rate_from(gp, toi, xga)

        # On-ice percentages / differentials
        if metric == 'CF%':
            return _pct(cf, (cf + ca))
        if metric == 'FF%':
            return _pct(ff, (ff + fa))
        if metric == 'SF%':
            return _pct(sf, (sf + sa))
        if metric == 'GF%':
            return _pct(gf, (gf + ga))
        if metric == 'xGF%':
            return _pct(xgf, (xgf + xga))
        if metric == 'C+/-':
            return _rate_from(gp, toi, (cf - ca))
        if metric == 'F+/-':
            return _rate_from(gp, toi, (ff - fa))
        if metric == 'S+/-':
            return _rate_from(gp, toi, (sf - sa))
        if metric == 'G+/-':
            return _rate_from(gp, toi, (gf - ga))
        if metric == 'xG+/-':
            return _rate_from(gp, toi, (xgf - xga))

        if category == 'Context' and metric == 'Sh%':
            return _pct(gf, sf)
        if category == 'Context' and metric == 'Sv%':
            if sa <= 0:
                return 100.0 if ga <= 0 else 0.0
            return 100.0 * (1.0 - (ga / sa))
        if category == 'Context' and metric == 'PDO':
            sh_oi = _pct(gf, sf)
            sv_oi = 100.0 if sa <= 0 and ga <= 0 else (0.0 if sa <= 0 else 100.0 * (1.0 - (ga / sa)))
            return (sh_oi + sv_oi) if (sh_oi is not None and sv_oi is not None) else None
        if category == 'Context' and metric == 'GAx':
            return _rate_from(gp, toi, (gf - xgf))
        if category == 'Context' and metric == 'GSAx':
            return _rate_from(gp, toi, (xga - ga))

        if category == 'Penalties' and metric == 'PIM_taken':
            return _rate_from(gp, toi, pim_taken)
        if category == 'Penalties' and metric == 'PIM_drawn':
            return _rate_from(gp, toi, pim_drawn)
        if category == 'Penalties' and metric == 'PIM+/-':
            return _rate_from(gp, toi, (pim_drawn - pim_taken))
        if category == 'Penalties' and metric == 'PIM_For':
            return _rate_from(gp, toi, pim_for)
        if category == 'Penalties' and metric == 'PIM_Against':
            return _rate_from(gp, toi, pim_against)
        if category == 'Penalties' and metric == 'oiPIM+/-':
            return _rate_from(gp, toi, (pim_for - pim_against))

        if category == 'Other' and metric == 'Hits':
            return _rate_from(gp, toi, hits)
        if category == 'Other' and metric == 'Takeaways':
            return _rate_from(gp, toi, takeaways)
        if category == 'Other' and metric == 'Giveaways':
            return _rate_from(gp, toi, giveaways)

        if metric and metric in v:
            try:
                return _rate_from(gp, toi, float(v.get(metric) or 0.0))
            except Exception:
                return None

        return None

    out_players: List[Dict[str, Any]] = []
    for pid_i in player_ids:
        v = agg.get(int(pid_i))
        mm: Dict[str, Optional[float]] = {}
        for mid in metric_ids:
            mm[mid] = _compute_metric(mid, v, int(pid_i))
        out_players.append({'playerId': int(pid_i), 'metrics': mm})

    label_attempts = 'iShots' if xg_model == 'xG_S' else 'iFenwick'
    label_sh = 'Sh%' if xg_model == 'xG_S' else 'FSh%'
    label_xsh = 'xSh%' if xg_model == 'xG_S' else 'xFSh%'
    label_dsh = 'dSh%' if xg_model == 'xG_S' else 'dFSh%'

    payload = {
        'season': int(season_int),
        'scope': scope,
        'seasonState': season_state,
        'strengthState': strength_state,
        'xgModel': xg_model,
        'rates': rates,
        'minGP': int(min_gp),
        'minTOI': float(min_toi),
        'playerIds': player_ids,
        'metricIds': metric_ids,
        'labels': {
            'Attempts': label_attempts,
            'Sh': label_sh,
            'xSh': label_xsh,
            'dSh': label_dsh,
        },
        'players': out_players,
    }
    j = jsonify(payload)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/goalies/table', methods=['GET', 'POST'])
def api_goalies_table():
    """Bulk table metrics for goalies using the same slicers as the Goalies page."""
    body: Optional[Dict[str, Any]] = None
    try:
        if request.method == 'POST':
            maybe = request.get_json(silent=True)
            if isinstance(maybe, dict):
                body = maybe
    except Exception:
        body = None

    def _get(key: str, default: Any = None) -> Any:
        try:
            if isinstance(body, dict) and key in body and body.get(key) is not None:
                return body.get(key)
        except Exception:
            pass
        return request.args.get(key, default)

    season = str(_get('season') or '').strip()
    season_state = str(_get('seasonState', 'regular') or 'regular').strip().lower()
    strength_state = str(_get('strengthState', '5v5') or '5v5').strip()
    xg_model = str(_get('xgModel', 'xG_F') or 'xG_F').strip()
    rates = str(_get('rates') or _get('ratesTotals') or 'Totals').strip() or 'Totals'
    scope = str(_get('scope', 'season') or 'season').strip().lower()

    metric_ids_val = _get('metricIds') or _get('metrics')
    player_ids_val = _get('playerIds') or _get('player_ids')

    min_gp = _safe_int(_get('minGP') or _get('minGp') or _get('min_gp') or 0) or 0
    min_toi_raw = _get('minTOI') or _get('minToi') or _get('min_toi') or 0
    try:
        min_toi = float(_parse_locale_float(min_toi_raw) or 0.0)
    except Exception:
        min_toi = 0.0
    if min_gp < 0:
        min_gp = 0
    if min_toi < 0:
        min_toi = 0.0

    player_ids: List[int] = []
    if isinstance(player_ids_val, list):
        for v in player_ids_val:
            pid_i = _safe_int(v)
            if pid_i and pid_i > 0:
                player_ids.append(int(pid_i))
    else:
        player_ids_raw = str(player_ids_val or '').strip()
        if not player_ids_raw:
            return jsonify({'error': 'missing_playerIds'}), 400
        for part in player_ids_raw.split(','):
            part = str(part or '').strip()
            if not part:
                continue
            pid_i = _safe_int(part)
            if pid_i and pid_i > 0:
                player_ids.append(int(pid_i))
    seen: set[int] = set()
    player_ids = [pid for pid in player_ids if not (pid in seen or seen.add(pid))]
    if not player_ids:
        return jsonify({'error': 'empty_playerIds'}), 400

    season_ids = _parse_request_season_ids(season, default=20252026)
    season_int = _primary_season_id(season_ids, default=20252026) or 20252026

    if season_state not in {'regular', 'playoffs', 'all'}:
        season_state = 'regular'
    if strength_state not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        strength_state = '5v5'
    if xg_model not in {'xG_S', 'xG_F', 'xG_F2'}:
        xg_model = 'xG_F'
    if rates not in {'Totals', 'Per60', 'PerGame'}:
        rates = 'Totals'
    if scope not in {'season', 'career'}:
        scope = 'season'

    metric_ids: List[str] = []
    if isinstance(metric_ids_val, list):
        metric_ids = [str(s).strip() for s in metric_ids_val if s is not None and str(s).strip()]
    else:
        metric_ids_raw = str(metric_ids_val or '').strip()
        if metric_ids_raw:
            metric_ids = [s.strip() for s in metric_ids_raw.split(',') if s and s.strip()]
    if not metric_ids:
        defs0 = _load_card_metrics_defs('goalies')
        metric_ids = [str(m.get('id')) for m in (defs0.get('metrics') or []) if isinstance(m, dict) and m.get('id')]

    agg, _pos_group_by_pid = _build_goalies_seasonstats_agg(
        scope=scope,
        season_int=season_int,
        season_ids=season_ids,
        season_state=season_state,
        strength_state=strength_state,
    )

    if min_gp > 0 or min_toi > 0:
        eligible = {pid_k for pid_k, d in agg.items() if float(d.get('GP') or 0) >= float(min_gp) and float(d.get('TOI') or 0.0) >= float(min_toi)}
        agg = {pid_k: d for pid_k, d in agg.items() if pid_k in eligible}

    try:
        eligible_pids = set(int(k) for k in agg.keys())
        player_ids = [int(pid) for pid in player_ids if int(pid) in eligible_pids]
    except Exception:
        pass

    def _xga(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('xGA_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('xGA_F2') or 0.0)
        return float(v.get('xGA_S') or 0.0)

    def _rate_from(gp: float, toi: float, vv: Optional[float]) -> Optional[float]:
        if rates == 'Totals':
            return vv
        denom = None
        if rates == 'PerGame':
            denom = gp if gp > 0 else None
        elif rates == 'Per60':
            denom = (toi / 60.0) if toi > 0 else None
        if vv is None or denom is None or denom <= 0:
            return None
        try:
            return float(vv) / float(denom)
        except Exception:
            return None

    def _sv_frac(ga: float, att: float) -> float:
        if att <= 0:
            return 1.0 if ga <= 0 else 0.0
        return 1.0 - (ga / att)

    total_sa = 0.0
    total_ga = 0.0
    for _pid_i, vv in agg.items():
        try:
            total_sa += float(vv.get('SA') or 0.0)
            total_ga += float(vv.get('GA') or 0.0)
        except Exception:
            continue
    avg_sv = _sv_frac(float(total_ga or 0.0), float(total_sa or 0.0))

    career_gsaa_by_pid: Dict[int, float] = {}
    career_gsax_by_pid: Dict[int, float] = {}
    if scope == 'career' and any(str(mid) in {'Results|GSAA', 'Results|GSAx'} for mid in metric_ids):
        try:
            by_pid_season, league_sa_ga = _build_goalies_career_season_matrix(
                season_state=season_state,
                strength_state=strength_state,
            )

            pid_set = set(int(pid) for pid in player_ids)
            for pid_i in pid_set:
                seasons = by_pid_season.get(int(pid_i)) or {}
                gsaa_sum = 0.0
                gsax_sum = 0.0
                for s_id, srow in seasons.items():
                    try:
                        sa_s = float(srow.get('SA') or 0.0)
                        ga_s = float(srow.get('GA') or 0.0)
                    except Exception:
                        continue

                    tot_sa, tot_ga = league_sa_ga.get(int(s_id), (0.0, 0.0))
                    avg_sv_s = _sv_frac(float(tot_ga or 0.0), float(tot_sa or 0.0))
                    sv_s = _sv_frac(ga_s, sa_s)
                    gsaa_sum += (sv_s - avg_sv_s) * float(sa_s or 0.0)

                    if int(s_id) >= 20102011:
                        try:
                            xga_s = _xga(srow)
                            gsax_sum += float(xga_s or 0.0) - float(ga_s or 0.0)
                        except Exception:
                            continue

                career_gsaa_by_pid[int(pid_i)] = float(gsaa_sum)
                career_gsax_by_pid[int(pid_i)] = float(gsax_sum)
        except Exception:
            career_gsaa_by_pid = {}
            career_gsax_by_pid = {}

    def _compute_metric(metric_id: str, pid_i: int, v: Optional[Dict[str, Any]]) -> Optional[float]:
        if v is None:
            return None
        gp = float(v.get('GP') or 0.0)
        toi = float(v.get('TOI') or 0.0)
        fa = float(v.get('FA') or 0.0)
        sa = float(v.get('SA') or 0.0)
        ga = float(v.get('GA') or 0.0)
        xga = _xga(v)

        category = None
        metric = None
        if '|' in metric_id:
            category, metric = metric_id.split('|', 1)
        else:
            metric = metric_id

        if category == 'Workload' and metric == 'FA':
            return _rate_from(gp, toi, fa)
        if category == 'Workload' and metric == 'SA':
            return _rate_from(gp, toi, sa)
        if category == 'Workload' and metric == 'xGA':
            return _rate_from(gp, toi, xga)
        if category == 'Workload' and metric == 'GA':
            return _rate_from(gp, toi, ga)
        if category == 'Save Percentage' and metric == 'Sv% or FSv%':
            denom = sa if xg_model == 'xG_S' else fa
            return 100.0 * _sv_frac(ga, denom)
        if category == 'Save Percentage' and metric == 'xSv% or xFSv%':
            denom = sa if xg_model == 'xG_S' else fa
            return 100.0 * _sv_frac(xga, denom)
        if category == 'Save Percentage' and metric == 'dSv% or dFSv%':
            denom = sa if xg_model == 'xG_S' else fa
            sv = 100.0 * _sv_frac(ga, denom)
            xsv = 100.0 * _sv_frac(xga, denom)
            return (sv - xsv)
        if category == 'Results' and metric == 'GSAx':
            if scope == 'career':
                return _rate_from(gp, toi, float(career_gsax_by_pid.get(int(pid_i), 0.0)))
            if int(season_int or 0) < 20102011:
                return _rate_from(gp, toi, 0.0)
            return _rate_from(gp, toi, (xga - ga))
        if category == 'Results' and metric == 'GSAA':
            if scope == 'career':
                return _rate_from(gp, toi, float(career_gsaa_by_pid.get(int(pid_i), 0.0)))
            sv = _sv_frac(ga, sa)
            gsaa = (sv - avg_sv) * sa
            return _rate_from(gp, toi, gsaa)

        return None

    out_players: List[Dict[str, Any]] = []
    for pid_i in player_ids:
        v = agg.get(int(pid_i))
        mm: Dict[str, Optional[float]] = {}
        for mid in metric_ids:
            mm[mid] = _compute_metric(mid, int(pid_i), v)
        out_players.append({'playerId': int(pid_i), 'metrics': mm})

    label_attempts = 'SA' if xg_model == 'xG_S' else 'FA'
    label_sv = 'Sv%' if xg_model == 'xG_S' else 'FSv%'
    label_xsv = 'xSv%' if xg_model == 'xG_S' else 'xFSv%'
    label_dsv = 'dSv%' if xg_model == 'xG_S' else 'dFSv%'

    payload = {
        'season': int(season_int),
        'scope': scope,
        'seasonState': season_state,
        'strengthState': strength_state,
        'xgModel': xg_model,
        'rates': rates,
        'minGP': int(min_gp),
        'minTOI': float(min_toi),
        'playerIds': player_ids,
        'metricIds': metric_ids,
        'labels': {
            'Attempts': label_attempts,
            'Sv': label_sv,
            'xSv': label_xsv,
            'dSv': label_dsv,
        },
        'players': out_players,
    }
    j = jsonify(payload)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/skaters/edge')
def api_skaters_edge():
    """Return a compact set of NHL Edge metrics for the Skaters 'Edge' tab.

    Query params:
      season=20252026
      playerId=<int>
      seasonState=regular|playoffs

    Notes:
      - NHL Edge data availability starts at 20212022.
      - Percentiles are returned by NHL Edge (0..1) and converted to 0..100.
      - Strength filter is applied only to distance + zone time metrics.
    """
    season = str(request.args.get('season') or '').strip()
    player_id_q = str(request.args.get('playerId') or request.args.get('player_id') or '').strip()
    season_state = str(request.args.get('seasonState') or 'regular').strip().lower()

    pid = _safe_int(player_id_q)
    if not pid or pid <= 0:
        return jsonify({'error': 'missing_playerId'}), 400

    season_ids = _parse_request_season_ids(season, default=20252026)
    season_int = _primary_season_id(season_ids, default=20252026) or 20252026

    if season_state not in {'regular', 'playoffs'}:
        season_state = 'regular'

    edge_seasons = [int(s) for s in season_ids if int(s) >= 20212022]

    # NHL Edge data begins in 20212022.
    if not edge_seasons:
        j0 = jsonify({
            'playerId': int(pid),
            'season': int(season_int),
            'seasons': [int(s) for s in season_ids],
            'seasonState': season_state,
            'gameType': _edge_game_type(season_state),
            'available': False,
            'reason': 'edge_unavailable_before_20212022',
            'shotSpeed': {},
            'skatingSpeed': {},
            'zoneTime': {},
            'skatingDistance': {},
        })
        try:
            j0.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
        return j0

    game_type = _edge_game_type(season_state)
    base = 'https://api-web.nhle.com/v1/edge'

    season_payloads: List[Dict[str, Any]] = []
    for edge_season in edge_seasons:
        urls = {
            'shotSpeed': f'{base}/skater-shot-speed-detail/{int(pid)}/{int(edge_season)}/{int(game_type)}',
            'skatingSpeed': f'{base}/skater-skating-speed-detail/{int(pid)}/{int(edge_season)}/{int(game_type)}',
            'zoneTime': f'{base}/skater-zone-time/{int(pid)}/{int(edge_season)}/{int(game_type)}',
            'skatingDistance': f'{base}/skater-skating-distance-detail/{int(pid)}/{int(edge_season)}/{int(game_type)}',
        }
        season_payloads.append({
            'season': int(edge_season),
            'shot': _edge_get_cached_json(urls['shotSpeed']) or {},
            'skate': _edge_get_cached_json(urls['skatingSpeed']) or {},
            'zone': _edge_get_cached_json(urls['zoneTime']) or {},
            'dist': _edge_get_cached_json(urls['skatingDistance']) or {},
        })

    def pack(payload: Dict[str, Any], metric_key: str, strength_code: Optional[str] = None) -> Dict[str, Any]:
        v, p, a = _edge_extract_value_pct_avg(payload, metric_key, strength_code)
        return {'value': v, 'pct': p, 'avg': a}

    def _merge_metric_dict(items: Sequence[Dict[str, Any]], *, value_mode: str = 'avg') -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        value_mode_norm = str(value_mode or 'avg').strip().lower()
        for key in ('value', 'pct', 'avg'):
            vals = []
            for item in items:
                v = _safe_float(item.get(key))
                if v is not None and math.isfinite(float(v)):
                    vals.append(float(v))
            if not vals:
                out[key] = None
                continue
            if key == 'value':
                if value_mode_norm == 'sum':
                    out[key] = sum(vals)
                    continue
                if value_mode_norm == 'max':
                    out[key] = max(vals)
                    continue
            out[key] = sum(vals) / len(vals)
        return out

    def _merge_metrics_block(
        blocks: Sequence[Dict[str, Dict[str, Any]]],
        *,
        value_modes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        keys: set[str] = set()
        for b in blocks:
            keys.update([str(k) for k in (b or {}).keys()])
        out: Dict[str, Dict[str, Any]] = {}
        modes = value_modes or {}
        for k in keys:
            out[str(k)] = _merge_metric_dict(
                [dict((b or {}).get(k) or {}) for b in blocks],
                value_mode=str(modes.get(str(k)) or 'avg'),
            )
        return out

    shot_blocks = []
    skate_blocks = []
    zone_blocks: Dict[str, List[Dict[str, Dict[str, Any]]]] = {'all': [], 'es': [], 'pp': [], 'pk': []}
    dist_blocks: Dict[str, List[Dict[str, Dict[str, Any]]]] = {'all': [], 'es': [], 'pp': [], 'pk': []}

    for payload in season_payloads:
        payload_shot = payload.get('shot') or {}
        payload_skate = payload.get('skate') or {}
        payload_zone = payload.get('zone') or {}
        payload_dist = payload.get('dist') or {}

        shot_blocks.append({
            'topShotSpeed': pack(payload_shot, 'topShotSpeed'),
            'avgShotSpeed': pack(payload_shot, 'avgShotSpeed'),
            'shotAttempts70to80': pack(payload_shot, 'shotAttempts70to80'),
            'shotAttempts80to90': pack(payload_shot, 'shotAttempts80to90'),
            'shotAttempts90to100': pack(payload_shot, 'shotAttempts90to100'),
            'shotAttemptsOver100': pack(payload_shot, 'shotAttemptsOver100'),
        })

        skate_blocks.append({
            'maxSkatingSpeed': pack(payload_skate, 'maxSkatingSpeed'),
            'bursts18to20': pack(payload_skate, 'bursts18to20'),
            'bursts20to22': pack(payload_skate, 'bursts20to22'),
            'burstsOver22': pack(payload_skate, 'burstsOver22'),
        })

        for sc in ('all', 'es', 'pp', 'pk'):
            zone_blocks[sc].append({
                'offensiveZonePctg': pack(payload_zone, 'offensiveZonePctg', sc),
                'neutralZonePctg': pack(payload_zone, 'neutralZonePctg', sc),
                'defensiveZonePctg': pack(payload_zone, 'defensiveZonePctg', sc),
            })
            dist_blocks[sc].append({
                'distanceTotal': pack(payload_dist, 'distanceTotal', sc),
                'distancePer60': pack(payload_dist, 'distancePer60', sc),
            })

    shot_metrics = _merge_metrics_block(
        shot_blocks,
        value_modes={
            'topShotSpeed': 'max',
            'avgShotSpeed': 'avg',
            'shotAttempts70to80': 'sum',
            'shotAttempts80to90': 'sum',
            'shotAttempts90to100': 'sum',
            'shotAttemptsOver100': 'sum',
        },
    )
    skating_speed_metrics = _merge_metrics_block(
        skate_blocks,
        value_modes={
            'maxSkatingSpeed': 'max',
            'bursts18to20': 'sum',
            'bursts20to22': 'sum',
            'burstsOver22': 'sum',
        },
    )
    zone_time_by_strength: Dict[str, Any] = {
        sc: _merge_metrics_block(zone_blocks[sc])
        for sc in ('all', 'es', 'pp', 'pk')
    }
    skating_distance_by_strength: Dict[str, Any] = {
        sc: _merge_metrics_block(
            dist_blocks[sc],
            value_modes={'distanceTotal': 'sum', 'distancePer60': 'avg'},
        )
        for sc in ('all', 'es', 'pp', 'pk')
    }

    payload = {
        'playerId': int(pid),
        'season': int(season_int),
        'seasons': edge_seasons,
        'seasonState': season_state,
        'gameType': int(game_type),
        'available': True,
        'shotSpeed': shot_metrics,
        'skatingSpeed': skating_speed_metrics,
        'zoneTime': zone_time_by_strength,
        'skatingDistance': skating_distance_by_strength,
    }
    j = jsonify(payload)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/skaters/scatter')
def api_skaters_scatter():
    """League-wide scatter data for the Skaters 'Charts' tab.

    Query params (match Card slicers):
      season=20252026
      seasonState=regular|playoffs|all
      strengthState=5v5|PP|SH|Other|all
      xgModel=xG_S|xG_F|xG_F2
      rates=Totals|Per60|PerGame
      scope=season|career
      minGP=<int>
      minTOI=<float>

    Scatter params:
      xMetricId=<Category|Metric>
      yMetricId=<Category|Metric>

    Notes:
      - Uses SeasonStats aggregates for all players.
      - Supports RAPM and Context (QoT/QoC/ZS) metrics via static CSVs and/or Sheets for 20252026.
      - Does NOT support NHL Edge metrics.
    """
    season = str(request.args.get('season') or '').strip()
    season_state = str(request.args.get('seasonState') or 'regular').strip().lower()
    strength_state = str(request.args.get('strengthState') or '5v5').strip()
    xg_model = str(request.args.get('xgModel') or 'xG_F').strip()
    rates = str(request.args.get('rates') or 'Totals').strip()
    scope = str(request.args.get('scope') or 'season').strip().lower()
    x_metric_id = str(request.args.get('xMetricId') or request.args.get('xMetric') or '').strip()
    y_metric_id = str(request.args.get('yMetricId') or request.args.get('yMetric') or '').strip()

    try:
        min_gp = int(float(str(request.args.get('minGP') or '0').strip() or '0'))
    except Exception:
        min_gp = 0
    try:
        min_toi = float(str(request.args.get('minTOI') or '0').strip() or '0')
    except Exception:
        min_toi = 0.0

    season_ids = _parse_request_season_ids(season, default=20252026)
    season_int = _primary_season_id(season_ids, default=20252026) or 20252026

    if season_state not in {'regular', 'playoffs', 'all'}:
        season_state = 'regular'
    if strength_state not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        strength_state = '5v5'
    if xg_model not in {'xG_S', 'xG_F', 'xG_F2'}:
        xg_model = 'xG_F'
    if rates not in {'Totals', 'Per60', 'PerGame'}:
        rates = 'Totals'
    if scope not in {'season', 'career'}:
        scope = 'season'

    if not x_metric_id or not y_metric_id:
        return jsonify({'error': 'missing_metric', 'hint': 'Provide xMetricId and yMetricId'}), 400
    if str(x_metric_id).startswith('Edge|') or str(y_metric_id).startswith('Edge|'):
        return jsonify({'error': 'edge_not_supported'}), 400

    try:
        scatter_ttl_s = max(15, int(os.getenv('SKATERS_SCATTER_CACHE_TTL_SECONDS', '300') or '300'))
    except Exception:
        scatter_ttl_s = 300
    try:
        scatter_max_items = max(8, int(os.getenv('SKATERS_SCATTER_CACHE_MAX_ITEMS', '128') or '128'))
    except Exception:
        scatter_max_items = 128

    scatter_cache_key = (
        'v2',
        tuple(_normalize_season_id_list(season_ids)),
        str(scope),
        str(season_state),
        str(strength_state),
        str(xg_model),
        str(rates),
        int(min_gp),
        float(min_toi),
        str(x_metric_id),
        str(y_metric_id),
    )
    _cache_prune_ttl_and_size(_SKATERS_SCATTER_CACHE, ttl_s=scatter_ttl_s, max_items=scatter_max_items)
    scatter_cached = _cache_get(_SKATERS_SCATTER_CACHE, scatter_cache_key, scatter_ttl_s)
    if isinstance(scatter_cached, dict):
        j_cached = jsonify(scatter_cached)
        try:
            j_cached.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
        return j_cached

    agg, _pos_group_by_pid = _build_seasonstats_agg(
        scope=scope,
        season_int=season_int,
        season_ids=season_ids,
        season_state=season_state,
        strength_state=strength_state,
    )

    # Apply minimum requirements.
    if min_gp > 0 or min_toi > 0:
        eligible = {pid_k for pid_k, d in agg.items() if float(d.get('GP') or 0) >= float(min_gp) and float(d.get('TOI') or 0.0) >= float(min_toi)}
        agg = {pid_k: d for pid_k, d in agg.items() if pid_k in eligible}

    def _pct(n: Optional[float], d: Optional[float]) -> Optional[float]:
        try:
            if n is None or d is None:
                return None
            if d <= 0:
                return None
            return 100.0 * (float(n) / float(d))
        except Exception:
            return None

    def _attempts(v: Dict[str, Any]) -> float:
        vv = v.get('iShots') if xg_model == 'xG_S' else v.get('iFenwick')
        return float(vv or 0.0)

    def _ixg(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('ixG_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('ixG_F2') or 0.0)
        return float(v.get('ixG_S') or 0.0)

    def _xgf(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('xGF_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('xGF_F2') or 0.0)
        return float(v.get('xGF_S') or 0.0)

    def _xga(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('xGA_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('xGA_F2') or 0.0)
        return float(v.get('xGA_S') or 0.0)

    def _rate_from(gp: float, toi: float, vv: Optional[float]) -> Optional[float]:
        if rates == 'Totals':
            return vv
        denom = None
        if rates == 'PerGame':
            denom = gp if gp > 0 else None
        elif rates == 'Per60':
            denom = (toi / 60.0) if toi > 0 else None
        if vv is None or denom is None or denom <= 0:
            return None
        try:
            return float(vv) / float(denom)
        except Exception:
            return None

    def _norm_rates_totals(v: Any) -> str:
        s = str(v or '').strip().lower()
        if s.startswith('tot'):
            return 'Totals'
        if s.startswith('rate'):
            return 'Rates'
        return str(v or '').strip() or 'Rates'

    want_strength = strength_state if strength_state in {'5v5', 'PP', 'SH'} else '5v5'
    want_rapm_rates = 'Totals' if rates == 'Totals' else 'Rates'

    # Optional RAPM/context maps for league-wide lookup.
    rapm_by_pid: Dict[int, Dict[str, Any]] = {}
    ctx_by_pid: Dict[int, Dict[str, Any]] = {}

    needs_rapm = ('|RAPM ' in x_metric_id) or ('|RAPM ' in y_metric_id)
    needs_ctx = (x_metric_id in {'Context|QoT', 'Context|QoC', 'Context|ZS'}) or (y_metric_id in {'Context|QoT', 'Context|QoC', 'Context|ZS'})

    if needs_rapm:
        rapm_rows = _load_rapm_static_csv() or []

        for r in rapm_rows:
            try:
                if not _row_season_in_selected(r, season_ids):
                    continue
                st = str(r.get('StrengthState') or '').strip()
                rt = _norm_rates_totals(r.get('Rates_Totals') or r.get('Rates/Totals') or r.get('RatesTotals'))
                if st != want_strength or rt != want_rapm_rates:
                    continue
                pid_r = int(str(r.get('PlayerID') or '').strip())
                if pid_r <= 0:
                    continue
                rapm_by_pid[pid_r] = r
            except Exception:
                continue

    if needs_ctx:
        ctx_rows = _load_context_static_csv() or []

        for r in ctx_rows:
            try:
                if not _row_season_in_selected(r, season_ids):
                    continue
                st = str(r.get('StrengthState') or '').strip()
                if st != want_strength:
                    continue
                pid_r = int(str(r.get('PlayerID') or '').strip())
                if pid_r <= 0:
                    continue
                ctx_by_pid[pid_r] = r
            except Exception:
                continue

    def _compute_metric(metric_id: str, v: Dict[str, Any], player_id: int) -> Optional[float]:
        gp = float(v.get('GP') or 0.0)
        toi = float(v.get('TOI') or 0.0)
        igoals = float(v.get('iGoals') or 0.0)
        a1 = float(v.get('Assists1') or 0.0)
        a2 = float(v.get('Assists2') or 0.0)
        pts = igoals + a1 + a2
        att = _attempts(v)
        ixg = _ixg(v)

        cf = float(v.get('CF') or 0.0)
        ca = float(v.get('CA') or 0.0)
        ff = float(v.get('FF') or 0.0)
        fa = float(v.get('FA') or 0.0)
        sf = float(v.get('SF') or 0.0)
        sa = float(v.get('SA') or 0.0)
        gf = float(v.get('GF') or 0.0)
        ga = float(v.get('GA') or 0.0)
        xgf = _xgf(v)
        xga = _xga(v)

        pim_taken = float(v.get('PIM_taken') or 0.0)
        pim_drawn = float(v.get('PIM_drawn') or 0.0)
        pim_for = float(v.get('PIM_for') or 0.0)
        pim_against = float(v.get('PIM_against') or 0.0)
        hits = float(v.get('Hits') or 0.0)
        takeaways = float(v.get('Takeaways') or 0.0)
        giveaways = float(v.get('Giveaways') or 0.0)

        category = None
        metric = None
        if '|' in metric_id:
            category, metric = metric_id.split('|', 1)
        else:
            metric = metric_id

        # No Edge support here.
        if category == 'Edge':
            return None

        # League-wide RAPM.
        if metric and str(metric).startswith('RAPM '):
            rrow = rapm_by_pid.get(int(player_id))
            if not rrow:
                return None
            base = str(metric).replace('RAPM', '', 1).strip()
            col = None
            if base in {'CF', 'CA', 'GF', 'GA', 'xGF', 'xGA'}:
                col = base
            elif base == 'C+/-':
                col = 'C_plusminus'
            elif base == 'G+/-':
                col = 'G_plusminus'
            elif base == 'xG+/-':
                col = 'xG_plusminus'
            if not col:
                return None
            val = _parse_locale_float(rrow.get(col))
            return float(val) if val is not None else None

        # League-wide Context (QoT/QoC/ZS).
        if category == 'Context' and metric in {'QoT', 'QoC', 'ZS'}:
            crow = ctx_by_pid.get(int(player_id))
            if not crow:
                return None
            col2 = None
            if metric == 'QoT':
                col2 = 'QoT_blend_xG67_G33'
            elif metric == 'QoC':
                col2 = 'QoC_blend_xG67_G33'
            elif metric == 'ZS':
                col2 = 'ZS_Difficulty'
            val2 = _parse_locale_float(crow.get(col2)) if col2 else None
            return float(val2) if val2 is not None else None

        if metric == 'GP':
            return gp
        if metric == 'TOI':
            return toi

        if metric == 'iGoals':
            return _rate_from(gp, toi, igoals)
        if metric == 'Assists1':
            return _rate_from(gp, toi, a1)
        if metric == 'Assists2':
            return _rate_from(gp, toi, a2)
        if metric == 'Points':
            return _rate_from(gp, toi, pts)

        if metric in {'iShots', 'iFenwick', 'iShots or iFenwick'}:
            vv = float(v.get('iShots') or 0.0) if xg_model == 'xG_S' else float(v.get('iFenwick') or 0.0)
            return _rate_from(gp, toi, vv)

        if metric in {'ixG', 'Individual xG'}:
            return _rate_from(gp, toi, ixg)

        if category == 'Shooting' and metric in {'Sh% or FSh%', 'Sh%'}:
            return _pct(igoals, att)
        if category == 'Shooting' and metric in {'xSh% or xFS%', 'xSh%'}:
            return _pct(ixg, att)
        if category == 'Shooting' and metric in {'dSh% or dFSh%'}:
            sh = _pct(igoals, att)
            xsh = _pct(ixg, att)
            return (sh - xsh) if (sh is not None and xsh is not None) else None
        if metric == 'GAx' and category == 'Shooting':
            return _rate_from(gp, toi, (igoals - ixg))

        if metric == 'CF':
            return _rate_from(gp, toi, cf)
        if metric == 'CA':
            return _rate_from(gp, toi, ca)
        if metric == 'FF':
            return _rate_from(gp, toi, ff)
        if metric == 'FA':
            return _rate_from(gp, toi, fa)
        if metric == 'SF':
            return _rate_from(gp, toi, sf)
        if metric == 'SA':
            return _rate_from(gp, toi, sa)
        if metric == 'GF':
            return _rate_from(gp, toi, gf)
        if metric == 'GA':
            return _rate_from(gp, toi, ga)
        if metric == 'xGF':
            return _rate_from(gp, toi, xgf)
        if metric == 'xGA':
            return _rate_from(gp, toi, xga)

        if metric == 'CF%':
            return _pct(cf, (cf + ca))
        if metric == 'FF%':
            return _pct(ff, (ff + fa))
        if metric == 'SF%':
            return _pct(sf, (sf + sa))
        if metric == 'GF%':
            return _pct(gf, (gf + ga))
        if metric == 'xGF%':
            return _pct(xgf, (xgf + xga))
        if metric == 'C+/-':
            return _rate_from(gp, toi, (cf - ca))
        if metric == 'F+/-':
            return _rate_from(gp, toi, (ff - fa))
        if metric == 'S+/-':
            return _rate_from(gp, toi, (sf - sa))
        if metric == 'G+/-':
            return _rate_from(gp, toi, (gf - ga))
        if metric == 'xG+/-':
            return _rate_from(gp, toi, (xgf - xga))

        if category == 'Context' and metric == 'Sh%':
            return _pct(gf, sf)
        if category == 'Context' and metric == 'Sv%':
            if sa <= 0:
                return 100.0 if ga <= 0 else 0.0
            return 100.0 * (1.0 - (ga / sa))
        if category == 'Context' and metric == 'PDO':
            sh_oi = _pct(gf, sf)
            sv_oi = 100.0 if sa <= 0 and ga <= 0 else (0.0 if sa <= 0 else 100.0 * (1.0 - (ga / sa)))
            return (sh_oi + sv_oi) if (sh_oi is not None and sv_oi is not None) else None
        if category == 'Context' and metric == 'GAx':
            return _rate_from(gp, toi, (gf - xgf))
        if category == 'Context' and metric == 'GSAx':
            return _rate_from(gp, toi, (xga - ga))

        if category == 'Penalties' and metric == 'PIM_taken':
            return _rate_from(gp, toi, pim_taken)
        if category == 'Penalties' and metric == 'PIM_drawn':
            return _rate_from(gp, toi, pim_drawn)
        if category == 'Penalties' and metric == 'PIM+/-':
            return _rate_from(gp, toi, (pim_drawn - pim_taken))
        if category == 'Penalties' and metric == 'PIM_For':
            return _rate_from(gp, toi, pim_for)
        if category == 'Penalties' and metric == 'PIM_Against':
            return _rate_from(gp, toi, pim_against)
        if category == 'Penalties' and metric == 'oiPIM+/-':
            return _rate_from(gp, toi, (pim_for - pim_against))

        if category == 'Other' and metric == 'Hits':
            return _rate_from(gp, toi, hits)
        if category == 'Other' and metric == 'Takeaways':
            return _rate_from(gp, toi, takeaways)
        if category == 'Other' and metric == 'Giveaways':
            return _rate_from(gp, toi, giveaways)

        # If the metric names a column directly, use it.
        if metric and metric in v:
            try:
                return _rate_from(gp, toi, float(v.get(metric) or 0.0))
            except Exception:
                return None

        return None

    # Player/team labels via season-aware roster mapping.
    roster_map: Dict[int, Dict[str, Any]] = {}
    name_map: Dict[int, str] = {}
    try:
        roster_map = _load_all_rosters_for_seasons_cached(season_ids) or {}
    except Exception:
        roster_map = {}
    try:
        name_map = _load_player_names_for_seasons(season_ids) or {}
    except Exception:
        name_map = {}

    pts_out: List[Dict[str, Any]] = []
    for pid_i, v in agg.items():
        try:
            xv = _compute_metric(x_metric_id, v, int(pid_i))
            yv = _compute_metric(y_metric_id, v, int(pid_i))
            if xv is None or yv is None:
                continue
            xf = float(xv)
            yf = float(yv)
            if not math.isfinite(xf) or not math.isfinite(yf):
                continue
            info = roster_map.get(int(pid_i)) or {}
            team = str(info.get('team') or '').strip().upper()
            name = str(info.get('name') or '').strip()
            if not name:
                name = str(name_map.get(int(pid_i)) or '').strip()
            if not name:
                name = str(pid_i)
            pts_out.append({
                'playerId': int(pid_i),
                'name': name,
                'team': team,
                'x': xf,
                'y': yf,
                'gp': float(v.get('GP') or 0.0),
                'toi': float(v.get('TOI') or 0.0),
            })
        except Exception:
            continue

    label_attempts = 'iShots' if xg_model == 'xG_S' else 'iFenwick'
    label_sh = 'Sh%' if xg_model == 'xG_S' else 'FSh%'
    label_xsh = 'xSh%' if xg_model == 'xG_S' else 'xFSh%'
    label_dsh = 'dSh%' if xg_model == 'xG_S' else 'dFSh%'

    payload = {
        'season': season_int,
        'scope': scope,
        'seasonState': season_state,
        'strengthState': strength_state,
        'xgModel': xg_model,
        'rates': rates,
        'minGP': int(min_gp),
        'minTOI': float(min_toi),
        'xMetricId': x_metric_id,
        'yMetricId': y_metric_id,
        'labels': {
            'Attempts': label_attempts,
            'Sh': label_sh,
            'xSh': label_xsh,
            'dSh': label_dsh,
        },
        'points': pts_out,
    }
    _cache_set_multi_bounded(_SKATERS_SCATTER_CACHE, scatter_cache_key, payload, ttl_s=scatter_ttl_s, max_items=scatter_max_items)
    j = jsonify(payload)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/goalies/scatter')
def api_goalies_scatter():
    """League-wide scatter data for the Goalies 'Charts' tab."""
    season = str(request.args.get('season') or '').strip()
    season_state = str(request.args.get('seasonState') or 'regular').strip().lower()
    strength_state = str(request.args.get('strengthState') or '5v5').strip()
    xg_model = str(request.args.get('xgModel') or 'xG_F').strip()
    rates = str(request.args.get('rates') or 'Totals').strip()
    scope = str(request.args.get('scope') or 'season').strip().lower()
    x_metric_id = str(request.args.get('xMetricId') or request.args.get('xMetric') or '').strip()
    y_metric_id = str(request.args.get('yMetricId') or request.args.get('yMetric') or '').strip()

    try:
        min_gp = int(float(str(request.args.get('minGP') or '0').strip() or '0'))
    except Exception:
        min_gp = 0
    try:
        min_toi = float(str(request.args.get('minTOI') or '0').strip() or '0')
    except Exception:
        min_toi = 0.0

    season_ids = _parse_request_season_ids(season, default=20252026)
    season_int = _primary_season_id(season_ids, default=20252026) or 20252026

    if season_state not in {'regular', 'playoffs', 'all'}:
        season_state = 'regular'
    if strength_state not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        strength_state = '5v5'
    if xg_model not in {'xG_S', 'xG_F', 'xG_F2'}:
        xg_model = 'xG_F'
    if rates not in {'Totals', 'Per60', 'PerGame'}:
        rates = 'Totals'
    if scope not in {'season', 'career'}:
        scope = 'season'

    if not x_metric_id or not y_metric_id:
        return jsonify({'error': 'missing_metric', 'hint': 'Provide xMetricId and yMetricId'}), 400

    try:
        scatter_ttl_s = max(30, int(os.getenv('GOALIES_SCATTER_CACHE_TTL_SECONDS', '180') or '180'))
    except Exception:
        scatter_ttl_s = 180
    try:
        scatter_max_items = max(1, int(os.getenv('GOALIES_SCATTER_CACHE_MAX_ITEMS', '128') or '128'))
    except Exception:
        scatter_max_items = 128
    scatter_cache_key = (
        tuple(_normalize_season_id_list(season_ids)),
        str(scope),
        str(season_state),
        str(strength_state),
        str(xg_model),
        str(rates),
        int(min_gp),
        float(min_toi),
        str(x_metric_id),
        str(y_metric_id),
    )
    _cache_prune_ttl_and_size(_GOALIES_SCATTER_CACHE, ttl_s=scatter_ttl_s, max_items=scatter_max_items)
    scatter_cached = _cache_get(_GOALIES_SCATTER_CACHE, scatter_cache_key, scatter_ttl_s)
    if scatter_cached is not None:
        j = jsonify(scatter_cached)
        try:
            j.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
        return j

    agg, _pos_group_by_pid = _build_goalies_seasonstats_agg(
        scope=scope,
        season_int=season_int,
        season_ids=season_ids,
        season_state=season_state,
        strength_state=strength_state,
    )

    if min_gp > 0 or min_toi > 0:
        eligible = {pid_k for pid_k, d in agg.items() if float(d.get('GP') or 0) >= float(min_gp) and float(d.get('TOI') or 0.0) >= float(min_toi)}
        agg = {pid_k: d for pid_k, d in agg.items() if pid_k in eligible}

    def _xga(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('xGA_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('xGA_F2') or 0.0)
        return float(v.get('xGA_S') or 0.0)

    def _rate_from(gp: float, toi: float, vv: Optional[float]) -> Optional[float]:
        if rates == 'Totals':
            return vv
        denom = None
        if rates == 'PerGame':
            denom = gp if gp > 0 else None
        elif rates == 'Per60':
            denom = (toi / 60.0) if toi > 0 else None
        if vv is None or denom is None or denom <= 0:
            return None
        try:
            return float(vv) / float(denom)
        except Exception:
            return None

    def _sv_frac(ga: float, att: float) -> float:
        if att <= 0:
            return 1.0 if ga <= 0 else 0.0
        return 1.0 - (ga / att)

    total_sa = 0.0
    total_ga = 0.0
    for _pid_i, vv in agg.items():
        try:
            total_sa += float(vv.get('SA') or 0.0)
            total_ga += float(vv.get('GA') or 0.0)
        except Exception:
            continue
    avg_sv = _sv_frac(float(total_ga or 0.0), float(total_sa or 0.0))

    career_gsaa_by_pid: Dict[int, float] = {}
    career_gsax_by_pid: Dict[int, float] = {}
    if scope == 'career' and (x_metric_id in {'Results|GSAA', 'Results|GSAx'} or y_metric_id in {'Results|GSAA', 'Results|GSAx'}):
        try:
            by_pid_season, league_sa_ga = _build_goalies_career_season_matrix(
                season_state=season_state,
                strength_state=strength_state,
            )

            for pid_i in agg.keys():
                seasons = by_pid_season.get(int(pid_i)) or {}
                gsaa_sum = 0.0
                gsax_sum = 0.0
                for s_id, srow in seasons.items():
                    try:
                        sa_s = float(srow.get('SA') or 0.0)
                        ga_s = float(srow.get('GA') or 0.0)
                    except Exception:
                        continue

                    tot_sa, tot_ga = league_sa_ga.get(int(s_id), (0.0, 0.0))
                    avg_sv_s = _sv_frac(float(tot_ga or 0.0), float(tot_sa or 0.0))
                    sv_s = _sv_frac(ga_s, sa_s)
                    gsaa_sum += (sv_s - avg_sv_s) * float(sa_s or 0.0)

                    if int(s_id) >= 20102011:
                        try:
                            xga_s = _xga(srow)
                            gsax_sum += float(xga_s or 0.0) - float(ga_s or 0.0)
                        except Exception:
                            continue

                career_gsaa_by_pid[int(pid_i)] = float(gsaa_sum)
                career_gsax_by_pid[int(pid_i)] = float(gsax_sum)
        except Exception:
            career_gsaa_by_pid = {}
            career_gsax_by_pid = {}

    def _compute_metric(metric_id: str, pid_i: int, v: Dict[str, Any]) -> Optional[float]:
        gp = float(v.get('GP') or 0.0)
        toi = float(v.get('TOI') or 0.0)
        fa = float(v.get('FA') or 0.0)
        sa = float(v.get('SA') or 0.0)
        ga = float(v.get('GA') or 0.0)
        xga = _xga(v)

        category = None
        metric = None
        if '|' in metric_id:
            category, metric = metric_id.split('|', 1)
        else:
            metric = metric_id

        if category == 'Workload' and metric == 'FA':
            return _rate_from(gp, toi, fa)
        if category == 'Workload' and metric == 'SA':
            return _rate_from(gp, toi, sa)
        if category == 'Workload' and metric == 'xGA':
            return _rate_from(gp, toi, xga)
        if category == 'Workload' and metric == 'GA':
            return _rate_from(gp, toi, ga)
        if category == 'Save Percentage' and metric == 'Sv% or FSv%':
            denom = sa if xg_model == 'xG_S' else fa
            return 100.0 * _sv_frac(ga, denom)
        if category == 'Save Percentage' and metric == 'xSv% or xFSv%':
            denom = sa if xg_model == 'xG_S' else fa
            return 100.0 * _sv_frac(xga, denom)
        if category == 'Save Percentage' and metric == 'dSv% or dFSv%':
            denom = sa if xg_model == 'xG_S' else fa
            sv = 100.0 * _sv_frac(ga, denom)
            xsv = 100.0 * _sv_frac(xga, denom)
            return (sv - xsv)
        if category == 'Results' and metric == 'GSAx':
            if scope == 'career':
                return _rate_from(gp, toi, float(career_gsax_by_pid.get(int(pid_i), 0.0)))
            if int(season_int or 0) < 20102011:
                return _rate_from(gp, toi, 0.0)
            return _rate_from(gp, toi, (xga - ga))
        if category == 'Results' and metric == 'GSAA':
            if scope == 'career':
                return _rate_from(gp, toi, float(career_gsaa_by_pid.get(int(pid_i), 0.0)))
            sv = _sv_frac(ga, sa)
            gsaa = (sv - avg_sv) * sa
            return _rate_from(gp, toi, gsaa)
        return None

    def _ensure_league_goalie_map() -> Dict[int, Dict[str, Any]]:
        try:
            ttl_s = max(60, int(os.getenv('GOALIES_PLAYERS_CACHE_TTL_SECONDS', '21600') or '21600'))
        except Exception:
            ttl_s = 21600
        try:
            max_items = max(1, int(os.getenv('GOALIES_PLAYERS_CACHE_MAX_ITEMS', '12') or '12'))
        except Exception:
            max_items = 12
        ck = (tuple(season_ids), '__LEAGUE__', season_state)
        now2 = time.time()
        try:
            _cache_prune_ttl_and_size(_GOALIES_PLAYERS_CACHE, ttl_s=ttl_s, max_items=max_items)
            cached_players = _cache_get(_GOALIES_PLAYERS_CACHE, ck, int(ttl_s))
            if cached_players is not None:
                return {int(r.get('playerId') or 0): r for r in (cached_players or []) if int(r.get('playerId') or 0) > 0}
        except Exception:
            pass

        players: List[Dict[str, Any]] = []
        try:
            if season_state == 'regular':
                cay = f'seasonId={int(season_int)} and gameTypeId=2'
            elif season_state == 'playoffs':
                cay = f'seasonId={int(season_int)} and gameTypeId=3'
            else:
                cay = f'seasonId={int(season_int)} and (gameTypeId=2 or gameTypeId=3)'
            url = 'https://api.nhle.com/stats/rest/en/goalie/summary'
            r = requests.get(
                url,
                params={'limit': -1, 'start': 0, 'cayenneExp': cay},
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=25,
                allow_redirects=True,
            )
            if r.status_code == 200:
                data = r.json() if r.content else {}
                rows = data.get('data') if isinstance(data, dict) else None
                if isinstance(rows, list):
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        pid = _safe_int(row.get('playerId') or row.get('goalieId') or row.get('id'))
                        if not pid or pid <= 0:
                            continue
                        name = str(row.get('goalieFullName') or row.get('playerFullName') or row.get('skaterFullName') or '').strip() or str(pid)
                        team_raw = row.get('teamAbbrev') or row.get('teamAbbrevs') or row.get('currentTeamAbbrev') or ''
                        team_abbrev = str(team_raw or '').strip().upper()
                        if '/' in team_abbrev:
                            team_abbrev = team_abbrev.split('/')[0].strip().upper()
                        rec: Dict[str, Any] = {'playerId': int(pid), 'name': name, 'pos': 'G'}
                        if team_abbrev:
                            rec['team'] = team_abbrev
                        players.append(rec)
        except Exception:
            players = []

        try:
            _cache_set_multi_bounded(_GOALIES_PLAYERS_CACHE, ck, players, ttl_s=ttl_s, max_items=max_items)
        except Exception:
            pass
        return {int(r.get('playerId') or 0): r for r in players if int(r.get('playerId') or 0) > 0}

    goalie_info_by_pid = _ensure_league_goalie_map()

    pts_out: List[Dict[str, Any]] = []
    for pid_i, v in agg.items():
        try:
            xv = _compute_metric(x_metric_id, int(pid_i), v)
            yv = _compute_metric(y_metric_id, int(pid_i), v)
            if xv is None or yv is None:
                continue
            xf = float(xv)
            yf = float(yv)
            if not math.isfinite(xf) or not math.isfinite(yf):
                continue
            info = goalie_info_by_pid.get(int(pid_i)) or {}
            team = str(info.get('team') or '').strip().upper()
            name = str(info.get('name') or '').strip()
            if not name:
                name = str(pid_i)
            pts_out.append({
                'playerId': int(pid_i),
                'name': name,
                'team': team,
                'x': xf,
                'y': yf,
                'gp': float(v.get('GP') or 0.0),
                'toi': float(v.get('TOI') or 0.0),
            })
        except Exception:
            continue

    label_attempts = 'SA' if xg_model == 'xG_S' else 'FA'
    label_sv = 'Sv%' if xg_model == 'xG_S' else 'FSv%'
    label_xsv = 'xSv%' if xg_model == 'xG_S' else 'xFSv%'
    label_dsv = 'dSv%' if xg_model == 'xG_S' else 'dFSv%'

    payload = {
        'season': season_int,
        'scope': scope,
        'seasonState': season_state,
        'strengthState': strength_state,
        'xgModel': xg_model,
        'rates': rates,
        'minGP': int(min_gp),
        'minTOI': float(min_toi),
        'xMetricId': x_metric_id,
        'yMetricId': y_metric_id,
        'labels': {
            'Attempts': label_attempts,
            'Sv': label_sv,
            'xSv': label_xsv,
            'dSv': label_dsv,
        },
        'points': pts_out,
    }
    _cache_set_multi_bounded(_GOALIES_SCATTER_CACHE, scatter_cache_key, payload, ttl_s=scatter_ttl_s, max_items=scatter_max_items)
    j = jsonify(payload)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/rapm/scale')
def api_rapm_scale():
    """League min/max scales for the Skaters RAPM chart.

    Query params:
      season=20252026
      rates=Rates|Totals
      metric=corsi|xg|goals

    Returns ranges for:
      - fivev5: differential (C+/xG+/G+)
      - pp: PP offense (PP_CF/PP_xGF/PP_GF) from StrengthState=PP rows
      - sh: SH defense (-SH_CA/-SH_xGA/-SH_GA) from StrengthState=SH rows
    """
    season = str(request.args.get('season') or '').strip()
    season_ids = _parse_request_season_ids(season)
    season_set = set(_normalize_season_id_list(season_ids))
    season_primary = _primary_season_id(season_ids)
    rates = str(request.args.get('rates') or 'Rates').strip() or 'Rates'
    metric = str(request.args.get('metric') or 'corsi').strip().lower() or 'corsi'
    player_id_q = str(request.args.get('playerId') or request.args.get('player_id') or '').strip()
    try:
        player_id_int = int(player_id_q) if player_id_q else None
    except Exception:
        player_id_int = None
    if metric not in {'corsi', 'xg', 'goals'}:
        metric = 'corsi'

    cache_key = (tuple(_normalize_season_id_list(season_ids)), rates, metric)
    try:
        ttl_s = max(30, int(os.getenv('RAPM_SCALE_CACHE_TTL_SECONDS', '300') or '300'))
    except Exception:
        ttl_s = 300
    try:
        max_items = max(1, int(os.getenv('RAPM_SCALE_CACHE_MAX_ITEMS', '24') or '24'))
    except Exception:
        max_items = 24
    now = time.time()

    try:
        _cache_prune_ttl_and_size(_RAPM_SCALE_CACHE, ttl_s=ttl_s, max_items=max_items)
    except Exception:
        pass
    cached = _RAPM_SCALE_CACHE.get(cache_key)
    if cached and (now - cached[0]) < ttl_s:
        payload, dists = cached[1], cached[2]
        if player_id_int is not None:
            payload = dict(payload)
            try:
                payload['playerId'] = player_id_int
                payload['player'] = _compute_player_scale_payload(player_id_int, dists)
            except Exception:
                payload['playerId'] = player_id_int
                payload['player'] = {'error': 'player_calc_failed'}
        j = jsonify(payload)
        try:
            j.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
        return j

    # Eligibility thresholds (minutes)
    MIN_5V5 = 100.0
    MIN_PP = 40.0
    MIN_SH = 40.0

    # Load rows (Supabase → CSV)
    rows = _load_rapm_static_csv()

    # Load context minutes (Supabase → CSV)
    ctx_rows = _load_context_static_csv()

    minutes_by_pid_strength: Dict[Tuple[int, str], float] = {}
    minutes_by_pid_strength_season: Dict[Tuple[int, str, int], float] = {}
    for r in ctx_rows:
        try:
            if season_set and not _row_season_in_selected(r, season_ids):
                continue
            pid = int(str(r.get('PlayerID') or '').strip())
            st = str(r.get('StrengthState') or '').strip()
            ssn = _safe_int(r.get('Season'))
            mins = _parse_locale_float(r.get('Minutes'))
            if mins is None:
                continue
            minutes_by_pid_strength[(pid, st)] = float(minutes_by_pid_strength.get((pid, st), 0.0)) + float(mins)
            if ssn:
                kss = (pid, st, int(ssn))
                minutes_by_pid_strength_season[kss] = float(minutes_by_pid_strength_season.get(kss, 0.0)) + float(mins)
        except Exception:
            continue

    def _rt(v: Any) -> str:
        s = str(v or '').strip().lower()
        if s.startswith('tot'):
            return 'Totals'
        if s.startswith('rate'):
            return 'Rates'
        return str(v or '').strip()

    # Columns for the requested metric
    if metric == 'xg':
        diff_col = 'xG_plusminus'
        pp_col = 'PP_xGF'
        sh_col = 'SH_xGA'
        pp_base = 'xGF'
        sh_base = 'xGA'
    elif metric == 'goals':
        diff_col = 'G_plusminus'
        pp_col = 'PP_GF'
        sh_col = 'SH_GA'
        pp_base = 'GF'
        sh_base = 'GA'
    else:
        diff_col = 'C_plusminus'
        pp_col = 'PP_CF'
        sh_col = 'SH_CA'
        pp_base = 'CF'
        sh_base = 'CA'

    # Build per-player values; apply eligibility by minutes
    five_by_pid: Dict[int, float] = {}
    pp_by_pid: Dict[int, float] = {}
    sh_by_pid: Dict[int, float] = {}
    five_off_by_pid: Dict[int, float] = {}
    five_def_by_pid: Dict[int, float] = {}
    five_w_by_pid: Dict[int, float] = {}
    pp_w_by_pid: Dict[int, float] = {}
    sh_w_by_pid: Dict[int, float] = {}

    # When Rates are requested but data only has Totals, convert Totals → Rates
    want_rates = _rt(rates) == 'Rates'
    has_rates_rows = any(
        _rt(r.get('Rates_Totals') or r.get('Rates/Totals') or r.get('RatesTotals')) == 'Rates'
        for r in rows
    )
    # If we want Rates and no Rates rows exist, use Totals rows with conversion
    use_totals_as_rates = want_rates and not has_rates_rows
    use_weighted_rate_average = want_rates and not use_totals_as_rates

    for r in rows:
        try:
            if season_set and not _row_season_in_selected(r, season_ids):
                continue
            row_rt = _rt(r.get('Rates_Totals') or r.get('Rates/Totals') or r.get('RatesTotals'))
            if use_totals_as_rates:
                if row_rt != 'Totals':
                    continue
            else:
                if row_rt != _rt(rates):
                    continue
            pid = int(str(r.get('PlayerID') or '').strip())
            st = str(r.get('StrengthState') or '').strip()
            season_row = _safe_int(r.get('Season'))
            row_minutes = float(minutes_by_pid_strength_season.get((pid, st, int(season_row)), 0.0)) if season_row else 0.0
            row_weight = row_minutes if row_minutes > 0 else 1.0

            # Rate conversion factor: Rates = Totals × (60 / minutes)
            rate_factor = 1.0
            if use_totals_as_rates:
                mins = minutes_by_pid_strength.get((pid, st))
                if mins and mins > 0:
                    rate_factor = 60.0 / mins
                else:
                    continue  # can't convert without minutes

            if st == '5v5':
                vdiff = _parse_locale_float(r.get(diff_col))
                if vdiff is not None:
                    if use_weighted_rate_average:
                        five_by_pid[pid] = float(five_by_pid.get(pid, 0.0)) + (float(vdiff) * rate_factor * row_weight)
                        five_w_by_pid[pid] = float(five_w_by_pid.get(pid, 0.0)) + row_weight
                    else:
                        five_by_pid[pid] = float(five_by_pid.get(pid, 0.0)) + (float(vdiff) * rate_factor)

                voff = _parse_locale_float(r.get('CF' if metric == 'corsi' else ('xGF' if metric == 'xg' else 'GF')))
                vdef_raw = _parse_locale_float(r.get('CA' if metric == 'corsi' else ('xGA' if metric == 'xg' else 'GA')))
                if voff is not None:
                    if use_weighted_rate_average:
                        five_off_by_pid[pid] = float(five_off_by_pid.get(pid, 0.0)) + (float(voff) * rate_factor * row_weight)
                    else:
                        five_off_by_pid[pid] = float(five_off_by_pid.get(pid, 0.0)) + (float(voff) * rate_factor)
                if vdef_raw is not None:
                    if use_weighted_rate_average:
                        five_def_by_pid[pid] = float(five_def_by_pid.get(pid, 0.0)) - (float(vdef_raw) * rate_factor * row_weight)
                    else:
                        five_def_by_pid[pid] = float(five_def_by_pid.get(pid, 0.0)) - (float(vdef_raw) * rate_factor)

                # Fallback PP/SH columns on 5v5 rows.
                vpp = _parse_locale_float(r.get(pp_col))
                if vpp is not None:
                    if use_weighted_rate_average:
                        pp_by_pid[pid] = float(pp_by_pid.get(pid, 0.0)) + (float(vpp) * rate_factor * row_weight)
                        pp_w_by_pid[pid] = float(pp_w_by_pid.get(pid, 0.0)) + row_weight
                    else:
                        pp_by_pid[pid] = float(pp_by_pid.get(pid, 0.0)) + (float(vpp) * rate_factor)
                vsh = _parse_locale_float(r.get(sh_col))
                if vsh is not None:
                    if use_weighted_rate_average:
                        sh_by_pid[pid] = float(sh_by_pid.get(pid, 0.0)) - (float(vsh) * rate_factor * row_weight)
                        sh_w_by_pid[pid] = float(sh_w_by_pid.get(pid, 0.0)) + row_weight
                    else:
                        sh_by_pid[pid] = float(sh_by_pid.get(pid, 0.0)) - (float(vsh) * rate_factor)

            elif st == 'PP':
                vpp = _parse_locale_float(r.get(pp_col))
                if vpp is None:
                    vpp = _parse_locale_float(r.get(pp_base))
                if vpp is not None:
                    if use_weighted_rate_average:
                        pp_by_pid[pid] = float(pp_by_pid.get(pid, 0.0)) + (float(vpp) * rate_factor * row_weight)
                        pp_w_by_pid[pid] = float(pp_w_by_pid.get(pid, 0.0)) + row_weight
                    else:
                        pp_by_pid[pid] = float(pp_by_pid.get(pid, 0.0)) + (float(vpp) * rate_factor)

            elif st == 'SH':
                vsh = _parse_locale_float(r.get(sh_col))
                if vsh is None:
                    vsh = _parse_locale_float(r.get(sh_base))
                if vsh is not None:
                    if use_weighted_rate_average:
                        sh_by_pid[pid] = float(sh_by_pid.get(pid, 0.0)) - (float(vsh) * rate_factor * row_weight)
                        sh_w_by_pid[pid] = float(sh_w_by_pid.get(pid, 0.0)) + row_weight
                    else:
                        sh_by_pid[pid] = float(sh_by_pid.get(pid, 0.0)) - (float(vsh) * rate_factor)
        except Exception:
            continue

    if use_weighted_rate_average:
        for pid, total in list(five_by_pid.items()):
            w = float(five_w_by_pid.get(pid, 0.0))
            if w > 0:
                five_by_pid[pid] = float(total) / w
        for pid, total in list(five_off_by_pid.items()):
            w = float(five_w_by_pid.get(pid, 0.0))
            if w > 0:
                five_off_by_pid[pid] = float(total) / w
        for pid, total in list(five_def_by_pid.items()):
            w = float(five_w_by_pid.get(pid, 0.0))
            if w > 0:
                five_def_by_pid[pid] = float(total) / w
        for pid, total in list(pp_by_pid.items()):
            w = float(pp_w_by_pid.get(pid, 0.0))
            if w > 0:
                pp_by_pid[pid] = float(total) / w
        for pid, total in list(sh_by_pid.items()):
            w = float(sh_w_by_pid.get(pid, 0.0))
            if w > 0:
                sh_by_pid[pid] = float(total) / w

    def _eligible(pid: int, strength: str) -> bool:
        mins = minutes_by_pid_strength.get((pid, strength))
        if mins is None:
            return False
        if strength == '5v5':
            return mins >= MIN_5V5
        if strength == 'PP':
            return mins >= MIN_PP
        if strength == 'SH':
            return mins >= MIN_SH
        return False

    five_vals = [v for pid, v in five_by_pid.items() if _eligible(pid, '5v5')]
    pp_vals = [v for pid, v in pp_by_pid.items() if _eligible(pid, 'PP')]
    sh_vals = [v for pid, v in sh_by_pid.items() if _eligible(pid, 'SH')]

    five_off_vals = [v for pid, v in five_off_by_pid.items() if _eligible(pid, '5v5')]
    five_def_vals = [v for pid, v in five_def_by_pid.items() if _eligible(pid, '5v5')]

    def _minmax(vals: List[float]) -> Dict[str, Any]:
        if not vals:
            return {'min': None, 'max': None}
        lo, hi = float(min(vals)), float(max(vals))
        # Make symmetric around zero and pad 10 % so bars don't touch edges
        ext = max(abs(lo), abs(hi)) * 1.10
        return {'min': -ext, 'max': ext}

    payload = {
        'season': season_primary,
        'rates': _rt(rates),
        'metric': metric,
        'source': 'supabase',
        'contextSource': 'supabase',
        'thresholds': {'fivev5': MIN_5V5, 'pp': MIN_PP, 'sh': MIN_SH},
        'fivev5': _minmax(five_vals),
        'pp': _minmax(pp_vals),
        'sh': _minmax(sh_vals),
    }

    # Build distributions for percentile calcs (sorted for bisect)
    dists: Dict[str, Any] = {
        'fivev5_diff': sorted(five_vals),
        'fivev5_off': sorted(five_off_vals),
        'fivev5_def': sorted(five_def_vals),
        'pp_off': sorted(pp_vals),
        'sh_def': sorted(sh_vals),
        'minutes': minutes_by_pid_strength,
        'values': {
            'fivev5_diff': five_by_pid,
            'fivev5_off': five_off_by_pid,
            'fivev5_def': five_def_by_pid,
            'pp_off': pp_by_pid,
            'sh_def': sh_by_pid,
        },
        'thresholds': {'fivev5': MIN_5V5, 'pp': MIN_PP, 'sh': MIN_SH},
    }

    def _bisect_pct(sorted_vals: List[float], v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        if not sorted_vals:
            return None
        import bisect
        idx = bisect.bisect_right(sorted_vals, v)
        return 100.0 * (idx / float(len(sorted_vals)))

    def _player_payload(pid: int) -> Dict[str, Any]:
        mins5 = minutes_by_pid_strength.get((pid, '5v5'))
        minsp = minutes_by_pid_strength.get((pid, 'PP'))
        minss = minutes_by_pid_strength.get((pid, 'SH'))
        elig5 = mins5 is not None and mins5 >= MIN_5V5
        eligp = minsp is not None and minsp >= MIN_PP
        eligs = minss is not None and minss >= MIN_SH
        v5off = five_off_by_pid.get(pid)
        v5def = five_def_by_pid.get(pid)
        v5diff = five_by_pid.get(pid)
        vpp = pp_by_pid.get(pid)
        vsh = sh_by_pid.get(pid)
        return {
            'minutes': {'5v5': mins5, 'PP': minsp, 'SH': minss},
            'eligible': {'5v5': elig5, 'PP': eligp, 'SH': eligs},
            'percentiles': {
                '5v5_off': _bisect_pct(dists['fivev5_off'], v5off) if elig5 else None,
                '5v5_def': _bisect_pct(dists['fivev5_def'], v5def) if elig5 else None,
                '5v5_diff': _bisect_pct(dists['fivev5_diff'], v5diff) if elig5 else None,
                'pp_off': _bisect_pct(dists['pp_off'], vpp) if eligp else None,
                'sh_def': _bisect_pct(dists['sh_def'], vsh) if eligs else None,
            },
        }

    # Expose player percentiles/eligibility when requested
    if player_id_int is not None:
        payload = dict(payload)
        payload['playerId'] = player_id_int
        payload['player'] = _player_payload(player_id_int)

    try:
        payload_base = payload if player_id_int is None else {k: v for k, v in payload.items() if k not in {'player', 'playerId'}}
        _cache_set_multi_bounded(_RAPM_SCALE_CACHE, cache_key, payload_base, dists, ttl_s=ttl_s, max_items=max_items)
    except Exception:
        pass
    j = jsonify(payload)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


def _compute_player_scale_payload(player_id: int, dists: Dict[str, Any]) -> Dict[str, Any]:
    """Compute eligibility + percentiles for one player from cached distributions."""
    mins = dists.get('minutes') or {}
    thresholds = dists.get('thresholds') or {}
    vmap = (dists.get('values') or {})
    try:
        MIN_5V5 = float(thresholds.get('fivev5', 100.0))
        MIN_PP = float(thresholds.get('pp', 40.0))
        MIN_SH = float(thresholds.get('sh', 40.0))
    except Exception:
        MIN_5V5, MIN_PP, MIN_SH = 100.0, 40.0, 40.0

    mins5 = mins.get((player_id, '5v5'))
    minsp = mins.get((player_id, 'PP'))
    minss = mins.get((player_id, 'SH'))
    elig5 = mins5 is not None and mins5 >= MIN_5V5
    eligp = minsp is not None and minsp >= MIN_PP
    eligs = minss is not None and minss >= MIN_SH

    import bisect
    def _pct(key: str, val: Optional[float]) -> Optional[float]:
        if val is None:
            return None
        arr = dists.get(key) or []
        if not arr:
            return None
        idx = bisect.bisect_right(arr, val)
        return 100.0 * (idx / float(len(arr)))

    v5off = (vmap.get('fivev5_off') or {}).get(player_id)
    v5def = (vmap.get('fivev5_def') or {}).get(player_id)
    v5diff = (vmap.get('fivev5_diff') or {}).get(player_id)
    vpp = (vmap.get('pp_off') or {}).get(player_id)
    vsh = (vmap.get('sh_def') or {}).get(player_id)

    return {
        'minutes': {'5v5': mins5, 'PP': minsp, 'SH': minss},
        'eligible': {'5v5': elig5, 'PP': eligp, 'SH': eligs},
        'percentiles': {
            '5v5_off': _pct('fivev5_off', v5off) if elig5 else None,
            '5v5_def': _pct('fivev5_def', v5def) if elig5 else None,
            '5v5_diff': _pct('fivev5_diff', v5diff) if elig5 else None,
            'pp_off': _pct('pp_off', vpp) if eligp else None,
            'sh_def': _pct('sh_def', vsh) if eligs else None,
        },
    }


@main_bp.route('/api/rapm/career')
def api_rapm_career():
    """Career RAPM series for a single player.

    Query params:
      playerId=<int>
      rates=Rates|Totals
      metric=corsi|xg|goals
      strength=5v5|PP|SH

    Output includes per-season values + per-season percentiles filtered by minutes thresholds.
    Scales are league-aware (min/max across eligible players per season).
    """
    pid_q = str(request.args.get('playerId') or request.args.get('player_id') or '').strip()
    try:
        pid = int(pid_q)
    except Exception:
        return jsonify({'error': 'missing_playerId'}), 400

    rates = str(request.args.get('rates') or 'Rates').strip() or 'Rates'
    metric = str(request.args.get('metric') or 'corsi').strip().lower() or 'corsi'
    strength = str(request.args.get('strength') or '5v5').strip() or '5v5'
    if metric not in {'corsi', 'xg', 'goals'}:
        metric = 'corsi'
    if strength not in {'All', '5v5', 'PP', 'SH'}:
        strength = '5v5'

    def _rt(v: Any) -> str:
        s = str(v or '').strip().lower()
        if s.startswith('tot'):
            return 'Totals'
        if s.startswith('rate'):
            return 'Rates'
        return str(v or '').strip()

    # thresholds (minutes)
    MIN_5V5 = 100.0
    MIN_PP = 40.0
    MIN_SH = 40.0

    def _season_int(v: Any) -> Optional[int]:
        """Parse a season into an int like 20252026.

        Accepts common formats from CSV/Sheets:
        - 20252026
        - "20252026"
        - "2025-2026" / "2025/2026"
        - "2025-26" / "2025/26"
        """
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        # Fast path: all digits
        if s.isdigit():
            try:
                n = int(s)
                return n if n >= 10000000 else None
            except Exception:
                return None

        # Normalize separators
        s2 = re.sub(r"[^0-9]", "", s)
        if len(s2) == 8:
            try:
                return int(s2)
            except Exception:
                return None

        # Handle YYYY-YY / YYYY/YY
        m = re.match(r"^(\d{4})\D+(\d{2})$", s)
        if m:
            try:
                a = int(m.group(1))
                b = int(m.group(2))
                end = (a // 100) * 100 + b
                return int(f"{a}{end:04d}")
            except Exception:
                return None
        return None

    # Build (and cache) league distributions/scales by season for this rates/metric/strength.
    cache_key = (_rt(rates), metric, strength)
    try:
        ttl_s = max(30, int(os.getenv('RAPM_CAREER_CACHE_TTL_SECONDS', '300') or '300'))
    except Exception:
        ttl_s = 300
    try:
        max_items = max(1, int(os.getenv('RAPM_CAREER_CACHE_MAX_ITEMS', '24') or '24'))
    except Exception:
        max_items = 24
    now = time.time()

    league = None
    try:
        _cache_prune_ttl_and_size(_RAPM_CAREER_CACHE, ttl_s=ttl_s, max_items=max_items)
        league = _cache_get(_RAPM_CAREER_CACHE, cache_key, int(ttl_s))
    except Exception:
        league = None

    if league is None:
        # Load RAPM rows from Supabase-backed static loader.
        rapm_rows = _load_rapm_static_csv() or []

        # Load context rows from Supabase-backed static loader.
        ctx_rows = _load_context_static_csv() or []

        minutes_by_season_pid_strength: Dict[Tuple[int, int, str], float] = {}
        for r in ctx_rows:
            try:
                season_int = _season_int(r.get('Season'))
                if season_int is None:
                    continue
                pid_i = int(str(r.get('PlayerID') or '').strip())
                st = str(r.get('StrengthState') or '').strip()
                mins = _parse_locale_float(r.get('Minutes'))
                if mins is None:
                    continue
                minutes_by_season_pid_strength[(season_int, pid_i, st)] = float(mins)
            except Exception:
                continue

        def _eligible(season_int: int, pid_i: int, st: str) -> bool:
            mins = minutes_by_season_pid_strength.get((season_int, pid_i, st))
            if mins is None:
                return False
            if st == '5v5':
                return mins >= MIN_5V5
            if st == 'PP':
                return mins >= MIN_PP
            if st == 'SH':
                return mins >= MIN_SH
            return False

        # Column names per metric
        if metric == 'xg':
            diff_col = 'xG_plusminus'
            off_col = 'xGF'
            def_col = 'xGA'
            pp_col = 'PP_xGF'
            pp_base = 'xGF'
            sh_col = 'SH_xGA'
            sh_base = 'xGA'
            z_off = 'xGF_zscore'
            z_def = 'xGA_zscore'
            z_diff = 'xG_plusminus_zscore'
            z_pp = 'PP_xGF_zscore'
            z_sh = 'SH_xGA_zscore'
        elif metric == 'goals':
            diff_col = 'G_plusminus'
            off_col = 'GF'
            def_col = 'GA'
            pp_col = 'PP_GF'
            pp_base = 'GF'
            sh_col = 'SH_GA'
            sh_base = 'GA'
            z_off = 'GF_zscore'
            z_def = 'GA_zscore'
            z_diff = 'G_plusminus_zscore'
            z_pp = 'PP_GF_zscore'
            z_sh = 'SH_GA_zscore'
        else:
            diff_col = 'C_plusminus'
            off_col = 'CF'
            def_col = 'CA'
            pp_col = 'PP_CF'
            pp_base = 'CF'
            sh_col = 'SH_CA'
            sh_base = 'CA'
            z_off = 'CF_zscore'
            z_def = 'CA_zscore'
            z_diff = 'C_plusminus_zscore'
            z_pp = 'PP_CF_zscore'
            z_sh = 'SH_CA_zscore'

        # Aggregate league distributions by season for percentiles + z-score stats
        dist_by_season: Dict[int, Dict[str, List[float]]] = {}
        scale_by_season: Dict[int, Dict[str, Dict[str, Optional[float]]]] = {}
        stats_by_season: Dict[int, Dict[str, Dict[str, Optional[float]]]] = {}

        def _push(season_int: int, key: str, v: Optional[float]):
            if v is None:
                return
            dist_by_season.setdefault(season_int, {}).setdefault(key, []).append(float(v))

        # For totals we need to combine strengths per player within a season.
        contrib: Dict[Tuple[int, int], Dict[str, float]] = {}

        # First pass: collect values per season/player/strength
        for r in rapm_rows:
            try:
                season_int = _season_int(r.get('Season'))
                if season_int is None:
                    continue
                if _rt(r.get('Rates_Totals') or r.get('Rates/Totals') or r.get('RatesTotals')) != _rt(rates):
                    continue
                pid_i = int(str(r.get('PlayerID') or '').strip())
                st = str(r.get('StrengthState') or '').strip()

                if st == '5v5' and _eligible(season_int, pid_i, '5v5'):
                    vdiff = _parse_locale_float(r.get(diff_col))
                    voff = _parse_locale_float(r.get(off_col))
                    vdef_raw = _parse_locale_float(r.get(def_col))
                    vdef = (-vdef_raw) if vdef_raw is not None else None
                    _push(season_int, '5v5_diff', vdiff)
                    _push(season_int, '5v5_off', voff)
                    _push(season_int, '5v5_def', vdef)

                    if voff is not None and vdef is not None:
                        contrib.setdefault((season_int, pid_i), {})['5v5_total'] = float(voff) + float(vdef)

                elif st == 'PP' and _eligible(season_int, pid_i, 'PP'):
                    vpp = _parse_locale_float(r.get(pp_col))
                    if vpp is None:
                        vpp = _parse_locale_float(r.get(pp_base))
                    _push(season_int, 'pp_off', vpp)

                    if vpp is not None:
                        contrib.setdefault((season_int, pid_i), {})['pp_off'] = float(vpp)

                elif st == 'SH' and _eligible(season_int, pid_i, 'SH'):
                    vsh = _parse_locale_float(r.get(sh_col))
                    if vsh is None:
                        vsh = _parse_locale_float(r.get(sh_base))
                    vsh2 = (-vsh) if vsh is not None else None
                    _push(season_int, 'sh_def', vsh2)

                    if vsh2 is not None:
                        contrib.setdefault((season_int, pid_i), {})['sh_def'] = float(vsh2)
            except Exception:
                continue

        # Second pass: totals distributions
        for (season_int, _pid_i), d in contrib.items():
            v5 = d.get('5v5_total')
            if v5 is not None:
                _push(season_int, '5v5_total', v5)
            all_total = 0.0
            any_part = False
            for k in ('5v5_total', 'pp_off', 'sh_def'):
                if k in d:
                    all_total += float(d[k])
                    any_part = True
            if any_part:
                _push(season_int, 'all_total', all_total)

        # Compute per-season min/max for scaling
        for season_int, d in dist_by_season.items():
            out: Dict[str, Dict[str, Optional[float]]] = {}
            for k, vals in d.items():
                if not vals:
                    out[k] = {'min': None, 'max': None}
                else:
                    out[k] = {'min': float(min(vals)), 'max': float(max(vals))}
            scale_by_season[season_int] = out

        # Compute per-season mean/std for z-score (for derived totals)
        for season_int, d in dist_by_season.items():
            out2: Dict[str, Dict[str, Optional[float]]] = {}
            for k, vals in d.items():
                if not vals:
                    out2[k] = {'mean': None, 'std': None}
                    continue
                try:
                    n = float(len(vals))
                    mean = float(sum(vals) / n)
                    var = float(sum((x - mean) ** 2 for x in vals) / n)
                    std = float(var ** 0.5)
                    out2[k] = {'mean': mean, 'std': std}
                except Exception:
                    out2[k] = {'mean': None, 'std': None}
            stats_by_season[season_int] = out2

        # Sort distributions for percentile calc
        for season_int, d in dist_by_season.items():
            for k in list(d.keys()):
                d[k] = sorted(d[k])

        league = {
            'dist': dist_by_season,
            'scale': scale_by_season,
            'stats': stats_by_season,
            'minutes': minutes_by_season_pid_strength,
            'thresholds': {'fivev5': MIN_5V5, 'pp': MIN_PP, 'sh': MIN_SH},
            'cols': {
                'diff': diff_col,
                'off': off_col,
                'def': def_col,
                'pp': pp_col,
                'pp_base': pp_base,
                'sh': sh_col,
                'sh_base': sh_base,
                'z_off': z_off,
                'z_def': z_def,
                'z_diff': z_diff,
                'z_pp': z_pp,
                'z_sh': z_sh,
            },
        }
        try:
            _cache_set_multi_bounded(_RAPM_CAREER_CACHE, cache_key, league, ttl_s=ttl_s, max_items=max_items)
        except Exception:
            pass

    # Pull this player's per-season series from RAPM rows.
    rapm_rows = _load_rapm_static_csv() or []

    cols = league.get('cols') or {}
    dist = league.get('dist') or {}
    scale = league.get('scale') or {}
    stats = league.get('stats') or {}
    minutes_map = league.get('minutes') or {}
    thresholds = league.get('thresholds') or {'fivev5': 100.0, 'pp': 40.0, 'sh': 40.0}

    def _mins(season_int: int, st: str) -> Optional[float]:
        return minutes_map.get((season_int, pid, st))

    def _elig(season_int: int, st: str) -> bool:
        m = _mins(season_int, st)
        if m is None:
            return False
        if st == '5v5':
            return m >= float(thresholds.get('fivev5', 100.0))
        if st == 'PP':
            return m >= float(thresholds.get('pp', 40.0))
        if st == 'SH':
            return m >= float(thresholds.get('sh', 40.0))
        return False

    import bisect
    def _pct(season_int: int, key: str, v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        arr = (dist.get(season_int) or {}).get(key) or []
        if not arr:
            return None
        idx = bisect.bisect_right(arr, v)
        return 100.0 * (idx / float(len(arr)))

    def _z(season_int: int, key: str, v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        mm = (stats.get(season_int) or {}).get(key) or {}
        mean = mm.get('mean')
        std = mm.get('std')
        if mean is None or std is None:
            return None
        try:
            std_f = float(std)
            if std_f == 0:
                return None
            return (float(v) - float(mean)) / std_f
        except Exception:
            return None

    series: Dict[int, Dict[str, Any]] = {}
    for r in rapm_rows:
        try:
            if _rt(r.get('Rates_Totals') or r.get('Rates/Totals') or r.get('RatesTotals')) != _rt(rates):
                continue
            if int(str(r.get('PlayerID') or '').strip()) != pid:
                continue
            season_int = _season_int(r.get('Season'))
            if season_int is None:
                continue
            st = str(r.get('StrengthState') or '').strip()
            if st not in {'5v5', 'PP', 'SH'}:
                continue

            item = series.setdefault(season_int, {'Season': season_int})
            if st == '5v5':
                off_key = str(cols.get('off') or '')
                def_key = str(cols.get('def') or '')
                diff_key = str(cols.get('diff') or '')
                z_off_key = str(cols.get('z_off') or '')
                z_def_key = str(cols.get('z_def') or '')
                z_diff_key = str(cols.get('z_diff') or '')

                voff = _parse_locale_float(r.get(off_key))
                vdef_raw = _parse_locale_float(r.get(def_key))
                vdef = (-vdef_raw) if vdef_raw is not None else None
                vdiff = _parse_locale_float(r.get(diff_key))
                item['5v5_off'] = voff
                item['5v5_def'] = vdef
                item['5v5_diff'] = vdiff
                zoff = _parse_locale_float(r.get(z_off_key))
                zdef_raw = _parse_locale_float(r.get(z_def_key))
                zdiff = _parse_locale_float(r.get(z_diff_key))
                item['5v5_off_z'] = zoff
                item['5v5_def_z'] = (-zdef_raw) if zdef_raw is not None else None
                item['5v5_diff_z'] = zdiff
            elif st == 'PP':
                pp_key = str(cols.get('pp') or '')
                pp_base_key = str(cols.get('pp_base') or '')
                z_pp_key = str(cols.get('z_pp') or '')
                vpp = _parse_locale_float(r.get(pp_key))
                if vpp is None:
                    vpp = _parse_locale_float(r.get(pp_base_key))
                item['pp_off'] = vpp
                item['pp_off_z'] = _parse_locale_float(r.get(z_pp_key))
            elif st == 'SH':
                sh_key = str(cols.get('sh') or '')
                sh_base_key = str(cols.get('sh_base') or '')
                z_sh_key = str(cols.get('z_sh') or '')
                vsh = _parse_locale_float(r.get(sh_key))
                if vsh is None:
                    vsh = _parse_locale_float(r.get(sh_base_key))
                item['sh_def'] = (-vsh) if vsh is not None else None
                zsh_raw = _parse_locale_float(r.get(z_sh_key))
                item['sh_def_z'] = (-zsh_raw) if zsh_raw is not None else None
        except Exception:
            continue

    seasons = sorted(series.keys())
    points: List[Dict[str, Any]] = []
    for season_int in seasons:
        row = series.get(season_int) or {'Season': season_int}
        mins5 = _mins(season_int, '5v5')
        minsp = _mins(season_int, 'PP')
        minss = _mins(season_int, 'SH')
        elig5 = _elig(season_int, '5v5')
        eligp = _elig(season_int, 'PP')
        eligs = _elig(season_int, 'SH')
        p: Dict[str, Any] = {
            'Season': season_int,
            'minutes': {'5v5': mins5, 'PP': minsp, 'SH': minss},
            'eligible': {'5v5': elig5, 'PP': eligp, 'SH': eligs},
        }
        if elig5:
            p['5v5_off'] = row.get('5v5_off')
            p['5v5_def'] = row.get('5v5_def')
            p['5v5_diff'] = row.get('5v5_diff')
            p['5v5_off_z'] = row.get('5v5_off_z')
            p['5v5_def_z'] = row.get('5v5_def_z')
            p['5v5_diff_z'] = row.get('5v5_diff_z')
            p['5v5_off_pct'] = _pct(season_int, '5v5_off', row.get('5v5_off'))
            p['5v5_def_pct'] = _pct(season_int, '5v5_def', row.get('5v5_def'))
            p['5v5_diff_pct'] = _pct(season_int, '5v5_diff', row.get('5v5_diff'))

            try:
                v5off = row.get('5v5_off')
                v5def = row.get('5v5_def')
                if v5off is not None and v5def is not None:
                    vtot = float(v5off) + float(v5def)
                    p['5v5_total'] = vtot
                    p['5v5_total_pct'] = _pct(season_int, '5v5_total', vtot)
                    p['5v5_total_z'] = _z(season_int, '5v5_total', vtot)
            except Exception:
                pass
        if eligp:
            p['pp_off'] = row.get('pp_off')
            p['pp_off_z'] = row.get('pp_off_z')
            p['pp_off_pct'] = _pct(season_int, 'pp_off', row.get('pp_off'))
        if eligs:
            p['sh_def'] = row.get('sh_def')
            p['sh_def_z'] = row.get('sh_def_z')
            p['sh_def_pct'] = _pct(season_int, 'sh_def', row.get('sh_def'))

        # All-strength total for this season: sum any eligible components.
        try:
            total_all = 0.0
            any_part = False
            if p.get('5v5_total') is not None:
                total_all += float(p['5v5_total'])
                any_part = True
            if eligp:
                vpp_any = p.get('pp_off')
                if vpp_any is not None:
                    total_all += float(vpp_any)
                    any_part = True
            if eligs:
                vsh_any = p.get('sh_def')
                if vsh_any is not None:
                    total_all += float(vsh_any)
                    any_part = True
            if any_part:
                p['all_total'] = total_all
                p['all_total_pct'] = _pct(season_int, 'all_total', total_all)
                p['all_total_z'] = _z(season_int, 'all_total', total_all)
        except Exception:
            pass
        points.append(p)

    # Global scale across seasons (league min/max per season, then overall min/max)
    def _minmax_over_seasons(key: str) -> Dict[str, Any]:
        mins: List[float] = []
        maxs: List[float] = []
        for season_int, s in scale.items():
            mm = (s or {}).get(key) or {}
            vmin = mm.get('min')
            vmax = mm.get('max')
            if vmin is not None:
                try:
                    mins.append(float(vmin))
                except Exception:
                    pass
            if vmax is not None:
                try:
                    maxs.append(float(vmax))
                except Exception:
                    pass
        return {'min': (min(mins) if mins else None), 'max': (max(maxs) if maxs else None)}

    if strength == 'All':
        league_scale = _minmax_over_seasons('all_total')
    elif strength == '5v5':
        # 5v5 chart uses Off/Def/Total; keep scale wide enough for all three.
        mm1 = _minmax_over_seasons('5v5_off')
        mm2 = _minmax_over_seasons('5v5_def')
        mm3 = _minmax_over_seasons('5v5_total')
        mins = [x for x in [mm1.get('min'), mm2.get('min'), mm3.get('min')] if x is not None]
        maxs = [x for x in [mm1.get('max'), mm2.get('max'), mm3.get('max')] if x is not None]
        league_scale = {'min': (min(mins) if mins else None), 'max': (max(maxs) if maxs else None)}
    elif strength == 'PP':
        league_scale = _minmax_over_seasons('pp_off')
    else:
        league_scale = _minmax_over_seasons('sh_def')

    payload = {
        'playerId': pid,
        'rates': _rt(rates),
        'metric': metric,
        'strength': strength,
        'thresholds': thresholds,
        'seasons': seasons,
        'points': points,
        'scale': league_scale,
    }
    j = jsonify(payload)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/seasons/<team_code>')
def api_seasons(team_code: str):
    """Return seasons for a given team using stored team season stats first.
    Shape: [{ "season": 20242025, "gameTypes": ["2", "3"] }, ...]
    """
    team = (team_code or '').upper().strip()
    if not team:
        return jsonify([])

    try:
        ttl_s = max(60, int(os.getenv('SEASONS_CACHE_TTL_SECONDS', '3600') or '3600'))
    except Exception:
        ttl_s = 3600
    cached = _TEAM_SEASONS_CACHE.get(team)
    now = time.time()
    if cached and (now - float(cached[0])) < float(ttl_s):
        return jsonify(cached[1])

    def _season_state_to_game_type(v: Any) -> Optional[str]:
        raw = str(v or '').strip().lower()
        if raw in {'2', 'reg', 'regular', 'regularseason', 'regular_season'}:
            return '2'
        if raw in {'3', 'po', 'playoffs', 'playoff'}:
            return '3'
        return None

    try:
        season_map: Dict[int, set] = {}
        for row in _iter_teamseasonstats_static_rows():
            row_team = str(row.get('Team') or row.get('team') or '').strip().upper()
            if row_team != team:
                continue
            season_i = _safe_int(row.get('Season') if 'Season' in row else row.get('season'))
            if not season_i:
                continue
            game_type = _season_state_to_game_type(row.get('SeasonState') if 'SeasonState' in row else row.get('season_state'))
            if not game_type:
                continue
            season_map.setdefault(int(season_i), set()).add(game_type)
    except Exception:
        pass

    # Always supplement with NHL API to discover seasons not yet in our data.
    url = f'https://api-web.nhle.com/v1/club-stats-season/{team}'
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                for it in data:
                    if not isinstance(it, dict):
                        continue
                    season_val = it.get('season')
                    gtypes = it.get('gameTypes')
                    try:
                        s_int = int(season_val) if season_val is not None else None
                    except Exception:
                        s_int = None
                    if s_int is None or not isinstance(gtypes, list):
                        continue
                    gts = {str(gt) for gt in gtypes if gt is not None}
                    if s_int not in season_map:
                        season_map[s_int] = gts
                    else:
                        season_map[s_int] |= gts
    except Exception:
        pass

    if season_map:
        out = [
            {'season': season_i, 'gameTypes': sorted(list(game_types))}
            for season_i, game_types in season_map.items()
        ]
        out.sort(key=lambda x: x['season'], reverse=True)
        _TEAM_SEASONS_CACHE[team] = (now, out)
        return jsonify(out)

    _TEAM_SEASONS_CACHE[team] = (now, [])
    return jsonify([])


@main_bp.route('/api/standings/<int:season>')
def api_standings(season: int):
    """Proxy NHL standings for a season, pass-through relevant fields."""
    def _current_season_id(now: Optional[datetime] = None) -> int:
        d = now or datetime.utcnow()
        y = d.year
        # NHL seasons typically begin around Sep/Oct; use Sep (9) as boundary
        if d.month >= 9:
            start_y = y
            end_y = y + 1
        else:
            start_y = y - 1
            end_y = y
        return start_y * 10000 + end_y

    def _normalize(data: Any) -> Any:
        if isinstance(data, dict) and 'standings' in data:
            return {'standings': data.get('standings') or []}
        if isinstance(data, list):
            return {'standings': data}
        return {'standings': []}

    # Try a series of endpoints known to serve standings across seasons
    urls: List[str] = []
    # If current season, 'now' is reliable
    if season == _current_season_id():
        urls.append('https://api-web.nhle.com/v1/standings/now')
    # If we have a last-date for this season, prefer date-based endpoints
    last_date = LAST_DATES.get(season)
    if last_date:
        urls.append(f'https://api-web.nhle.com/v1/standings/{last_date}')
        urls.append(f'https://api-web.nhle.com/v1/standings/{last_date}?gameType=2')
    # Season-coded variants
    urls.extend([
        f'https://api-web.nhle.com/v1/standings/{season}',
        f'https://api-web.nhle.com/v1/standings/{season}?gameType=2',
        f'https://api-web.nhle.com/v1/standings-season/{season}',
        f'https://api-web.nhle.com/v1/standings-season/{season}?gameType=2',
        f'https://api-web.nhle.com/v1/standings?season={season}',
        f'https://api-web.nhle.com/v1/standings?season={season}&gameType=2',
    ])

    last_status: Optional[int] = None
    try:
        for url in urls:
            try:
                r = requests.get(url, timeout=25)
            except Exception:
                continue
            last_status = r.status_code
            if r.status_code == 200:
                try:
                    data = r.json()
                except Exception:
                    continue
                return jsonify(_normalize(data))
        # If none succeeded and it's the current season, try 'now' one last time
        if season == _current_season_id():
            try:
                r2 = requests.get('https://api-web.nhle.com/v1/standings/now', timeout=25)
                if r2.status_code == 200:
                    return jsonify(_normalize(r2.json()))
            except Exception:
                pass
        # Final fallback: use stats REST standings by season (gameTypeId=2 regular season)
        try:
            stats_url = (
                'https://api.nhle.com/stats/rest/en/team/standings'
                '?isAggregate=false&reportType=basic&isGame=true&reportName=teamstandings'
                f'&cayenneExp=seasonId={season}%20and%20gameTypeId=2'
            )
            rs = requests.get(stats_url, timeout=30)
            last_status = rs.status_code
            if rs.status_code == 200:
                js = rs.json()
                rows = js.get('data') if isinstance(js, dict) else None
                out = []
                if isinstance(rows, list):
                    # Build team logo lookup
                    logo_by_abbrev = {}
                    try:
                        for tr in TEAM_ROWS:
                            ab = (tr.get('Team') or '').upper()
                            logo_by_abbrev[ab] = tr.get('Logo') or ''
                    except Exception:
                        logo_by_abbrev = {}
                    for rrow in rows:
                        try:
                            ab = (rrow.get('teamAbbrev') or rrow.get('teamAbbrevDefault') or '').upper()
                            gp = rrow.get('gamesPlayed') or rrow.get('gp') or 0
                            pts = rrow.get('points') or rrow.get('pts') or 0
                            w = rrow.get('wins') or rrow.get('w') or 0
                            l = rrow.get('losses') or rrow.get('l') or 0
                            otl = rrow.get('otLosses') or rrow.get('otl') or rrow.get('overtimeLosses') or 0
                            ties = rrow.get('ties') or 0
                            gf = rrow.get('goalsFor') or rrow.get('gf') or 0
                            ga = rrow.get('goalsAgainst') or rrow.get('ga') or 0
                            diff = (gf or 0) - (ga or 0)
                            ppct = rrow.get('pointsPercentage') or rrow.get('pointPctg')
                            if ppct is None:
                                try:
                                    ppct = (float(pts) / (2.0 * float(gp))) if gp else 0.0
                                except Exception:
                                    ppct = 0.0
                            l10w = rrow.get('lastTenWins') or rrow.get('l10Wins') or 0
                            l10l = rrow.get('lastTenLosses') or rrow.get('l10Losses') or 0
                            l10o = rrow.get('lastTenOtLosses') or rrow.get('l10OtLosses') or 0
                            # Streak
                            streak_code = rrow.get('streakCode') or rrow.get('streakType') or ''
                            streak_num = rrow.get('streakNumber') or rrow.get('streakCount') or 0
                            # Grouping
                            div_name = rrow.get('divisionName') or rrow.get('divisionAbbrev') or ''
                            conf_name = rrow.get('conferenceName') or rrow.get('conferenceAbbrev') or ''
                            out.append({
                                'teamAbbrev': ab,
                                'divisionName': div_name,
                                'conferenceName': conf_name,
                                'gamesPlayed': gp,
                                'points': pts,
                                'wins': w,
                                'losses': l,
                                'ties': ties,
                                'otLosses': otl,
                                'goalFor': gf,
                                'goalAgainst': ga,
                                'goalDifferential': diff,
                                'pointPctg': ppct,
                                'l10Wins': l10w,
                                'l10Losses': l10l,
                                'l10OtLosses': l10o,
                                'streakCode': (str(streak_code)[:1]).upper() if streak_code else '',
                                'streakCount': streak_num or 0,
                                'teamLogo': logo_by_abbrev.get(ab, ''),
                            })
                        except Exception:
                            continue
                return jsonify({'standings': out})
        except Exception:
            pass
        return jsonify({'error': 'Upstream error', 'status': last_status or 502}), 502
    except Exception:
        return jsonify({'error': 'Failed to fetch standings'}), 502


@main_bp.route('/api/live-games')
def api_live_games():
    """Return list of live games using NHL schedule/now endpoint.
    Filters to gameState indicating in-progress; if none, returns empty list.
    Shape: { games: [ { id, gameState, startTimeUTC, venue, awayTeam, homeTeam, periodDescriptor? } ] }
    """
    url = 'https://api-web.nhle.com/v1/schedule/now'
    try:
        r = requests.get(url, timeout=20)
    except Exception:
        return jsonify({'games': [], 'error': 'Fetch failed'}), 502
    if r.status_code != 200:
        return jsonify({'games': [], 'error': 'Upstream error', 'status': r.status_code}), 502
    try:
        js = r.json()
    except Exception:
        return jsonify({'games': []})
    live_states = {'LIVE', 'INPROGRESS', 'CRIT', 'OT', 'SHOOTOUT'}
    out = []
    for wk in (js.get('gameWeek') or []):
        for g in (wk.get('games') or []):
            st = str(g.get('gameState') or '').upper()
            if st in live_states:
                out.append({
                    'id': g.get('id'),
                    'season': g.get('season'),
                    'gameType': g.get('gameType'),
                    'startTimeUTC': g.get('startTimeUTC'),
                    'gameState': g.get('gameState'),
                    'venue': g.get('venue'),
                    'awayTeam': g.get('awayTeam'),
                    'homeTeam': g.get('homeTeam'),
                    'periodDescriptor': g.get('periodDescriptor'),
                })
    return jsonify({'games': out})


@main_bp.route('/admin/prestart-snapshots')
def admin_prestart_snapshots():
    """Admin endpoint to preview or download the prestart snapshots CSV.
    Query params:
      - mode: 'preview' (default) | 'download'
      - limit: number of rows to return for preview (default: 100, returns last N rows)
      - gameId: optional filter for a specific GameID (int) for preview
    """
    guard = _require_admin_api()
    if guard is not None:
        return guard
    mode = str(request.args.get('mode', 'preview')).strip().lower()
    path = _prestart_csv_path()
    # Download mode
    if mode == 'download':
        try:
            from flask import send_file  # local import to avoid top-level issues
            if not os.path.exists(path):
                return jsonify({'error': 'file_not_found', 'path': path}), 404
            resp = send_file(path, as_attachment=True, download_name=os.path.basename(path))
            try:
                resp.headers['Cache-Control'] = 'no-store'
            except Exception:
                pass
            return resp
        except Exception:
            return jsonify({'error': 'download_failed'}), 500
    # Preview mode
    try:
        limit = int(request.args.get('limit', '100'))
    except Exception:
        limit = 100
    try:
        game_id_filter = request.args.get('gameId')
        game_id_val = int(game_id_filter) if game_id_filter is not None else None
    except Exception:
        game_id_val = None
    if not os.path.exists(path):
        return jsonify({'exists': False, 'path': path, 'rows': [], 'total': 0})
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, 'r', encoding='utf-8', newline='') as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                if game_id_val is not None:
                    try:
                        gid = int(str(r.get('GameID') or '').strip())
                        if gid != game_id_val:
                            continue
                    except Exception:
                        continue
                rows.append(r)
    except Exception:
        return jsonify({'exists': True, 'path': path, 'rows': [], 'total': 0, 'error': 'read_failed'}), 500
    total = len(rows)
    if limit > 0 and total > limit:
        rows = rows[-limit:]
    return jsonify({'exists': True, 'path': path, 'total': total, 'limit': limit, 'rows': rows})


@main_bp.route('/favicon.png')
def favicon_png():
    """Serve favicon.png placed at project root.
    We look in CWD and repo root for a favicon.png and stream it; fallback 404.
    """
    from flask import send_file
    paths = [
        os.path.join(os.getcwd(), 'favicon.png'),
        os.path.join(os.path.dirname(__file__), '..', 'favicon.png'),
    ]
    for p in paths:
        try:
            p2 = os.path.abspath(p)
            if os.path.exists(p2):
                return send_file(p2, mimetype='image/png')
        except Exception:
            continue
    return ('', 404)


@main_bp.route('/api/diag/models')
def api_diag_models():
    """Diagnostics for model loading in production.
    Returns Python and package versions, model directory, and available model files.
    """
    import sys
    info: Dict[str, Any] = {}
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_dir = os.path.join(project_root, 'Model')
        files = []
        try:
            files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pkl')]) if os.path.isdir(model_dir) else []
        except Exception:
            files = []
        # Versions
        def _ver(mod_name: str) -> Optional[str]:
            try:
                mod = __import__(mod_name)
                return getattr(mod, '__version__', 'unknown')
            except Exception:
                return None
        info = {
            'python': sys.version,
            'versions': {
                'numpy': _ver('numpy'),
                'pandas': _ver('pandas'),
                'sklearn': _ver('sklearn'),
                'xgboost': _ver('xgboost'),
                'joblib': _ver('joblib'),
            },
            'model_dir': model_dir,
            'model_count': len(files),
            'models': files,
        }
    except Exception:
        info = {'error': 'diagnostics_failed'}
    return jsonify(info)


_TEAM_LOGO_PROXY_CACHE: Dict[str, Tuple[float, bytes]] = {}
_PLAYER_HEADSHOT_PROXY_CACHE: Dict[str, Tuple[float, bytes]] = {}


def _team_logo_source_url(team_abbrev: str) -> Optional[str]:
    a = (team_abbrev or '').strip().upper()
    if not a:
        return None
    # Prefer Teams.csv mapping when present
    try:
        for row in TEAM_ROWS:
            if (row.get('Team') or '').strip().upper() == a:
                u = (row.get('Logo') or '').strip()
                if u:
                    return u
    except Exception:
        pass
    # Fallback to current NHL assets
    return f'https://assets.nhle.com/logos/nhl/svg/{a}_light.svg'


def _normalize_svg_dimensions(svg_text: str) -> str:
    """Ensure SVG has width/height when only viewBox is provided.

    Some browsers are picky when drawing SVGs to canvas without intrinsic dimensions.
    """
    try:
        head = svg_text[:2048]
        if 'width=' in head and 'height=' in head:
            return svg_text
        m = re.search(r'viewBox\s*=\s*"\s*[-\d\.]+\s+[-\d\.]+\s+([\d\.]+)\s+([\d\.]+)\s*"', head, re.IGNORECASE)
        if not m:
            return svg_text
        w = m.group(1)
        h = m.group(2)

        def repl(match: re.Match) -> str:
            attrs = match.group(1) or ''
            if 'width=' in attrs.lower() or 'height=' in attrs.lower():
                return match.group(0)
            return f'<svg{attrs} width="{w}" height="{h}">' 

        return re.sub(r'<svg\b([^>]*)>', repl, svg_text, count=1, flags=re.IGNORECASE)
    except Exception:
        return svg_text


def _player_headshot_source_urls(player_id: int, season: str, team_abbrev: str) -> List[str]:
    pid_i = _safe_int(player_id)
    if not pid_i or pid_i <= 0:
        return []
    urls: List[str] = []
    season_s = str(season or '').strip()
    team_s = str(team_abbrev or '').strip().upper()
    if re.fullmatch(r'\d{8}', season_s) and re.fullmatch(r'[A-Z]{2,4}', team_s):
        urls.append(f'https://assets.nhle.com/mugs/nhl/{season_s}/{team_s}/{int(pid_i)}.png')
    urls.append(f'https://assets.nhle.com/mugs/nhl/latest/{int(pid_i)}.png')
    return urls


@main_bp.route('/api/team-logo/<team_abbrev>.svg')
def api_team_logo_svg(team_abbrev: str):
    """Proxy team SVG logos as same-origin to make canvas rendering reliable."""
    a = (team_abbrev or '').strip().upper()
    if not a or not re.fullmatch(r'[A-Z]{2,4}', a):
        return ('', 404)

    # Cache for 30 days in-process
    ttl_s = 30 * 24 * 3600
    try:
        max_items = max(1, int(os.getenv('TEAM_LOGO_PROXY_CACHE_MAX_ITEMS', '64') or '64'))
    except Exception:
        max_items = 64
    now = time.time()
    try:
        _cache_prune_ttl_and_size(_TEAM_LOGO_PROXY_CACHE, ttl_s=ttl_s, max_items=max_items)
        cached = _cache_get(_TEAM_LOGO_PROXY_CACHE, a, int(ttl_s))
        if cached:
            resp = make_response(cached)
            resp.headers['Content-Type'] = 'image/svg+xml'
            resp.headers['Cache-Control'] = 'public, max-age=86400'
            return resp
    except Exception:
        pass

    src = _team_logo_source_url(a)
    if not src:
        return ('', 404)

    # Whitelist host/path to avoid SSRF from Teams.csv edits
    try:
        from urllib.parse import urlparse
        pu = urlparse(src)
        if pu.scheme not in ('http', 'https') or pu.netloc.lower() not in ('assets.nhle.com',):
            src = f'https://assets.nhle.com/logos/nhl/svg/{a}_light.svg'
    except Exception:
        src = f'https://assets.nhle.com/logos/nhl/svg/{a}_light.svg'

    try:
        r = requests.get(src, timeout=10)
        if r.status_code != 200 or not (r.content or b''):
            return ('', 404)
        raw = r.content
        try:
            txt = raw.decode('utf-8', errors='replace')
            txt = _normalize_svg_dimensions(txt)
            data = txt.encode('utf-8')
        except Exception:
            data = raw
        try:
            _cache_set_multi_bounded(_TEAM_LOGO_PROXY_CACHE, a, data, ttl_s=ttl_s, max_items=max_items)
        except Exception:
            pass
        resp = make_response(data)
        resp.headers['Content-Type'] = 'image/svg+xml'
        resp.headers['Cache-Control'] = 'public, max-age=86400'
        return resp
    except Exception:
        return ('', 404)


@main_bp.route('/api/player-headshot/<int:player_id>.png')
def api_player_headshot_png(player_id: int):
    """Proxy NHL player mug PNGs as same-origin for canvas/export reliability."""
    pid_i = _safe_int(player_id)
    if not pid_i or pid_i <= 0:
        return ('', 404)

    season = str(request.args.get('season') or '').strip()
    team_abbrev = str(request.args.get('team') or '').strip().upper()
    cache_key = f'{int(pid_i)}|{season}|{team_abbrev}'
    ttl_s = 30 * 24 * 3600
    try:
        max_items = max(1, int(os.getenv('PLAYER_HEADSHOT_PROXY_CACHE_MAX_ITEMS', '256') or '256'))
    except Exception:
        max_items = 256

    try:
        _cache_prune_ttl_and_size(_PLAYER_HEADSHOT_PROXY_CACHE, ttl_s=ttl_s, max_items=max_items)
        cached = _cache_get(_PLAYER_HEADSHOT_PROXY_CACHE, cache_key, int(ttl_s))
        if cached:
            resp = make_response(cached)
            resp.headers['Content-Type'] = 'image/png'
            resp.headers['Cache-Control'] = 'public, max-age=86400'
            return resp
    except Exception:
        pass

    for src in _player_headshot_source_urls(int(pid_i), season, team_abbrev):
        try:
            from urllib.parse import urlparse
            pu = urlparse(src)
            if pu.scheme not in ('http', 'https') or pu.netloc.lower() not in ('assets.nhle.com',):
                continue
            r = requests.get(src, timeout=10)
            if r.status_code != 200 or not (r.content or b''):
                continue
            data = bytes(r.content)
            try:
                _cache_set_multi_bounded(_PLAYER_HEADSHOT_PROXY_CACHE, cache_key, data, ttl_s=ttl_s, max_items=max_items)
            except Exception:
                pass
            resp = make_response(data)
            resp.headers['Content-Type'] = 'image/png'
            resp.headers['Cache-Control'] = 'public, max-age=86400'
            return resp
        except Exception:
            continue

    return ('', 404)


# ── Supabase read helpers & column maps ──────────────────────────

def _sb_read(table: str, *, columns: str = "*",
             filters: Optional[Dict[str, str]] = None,
             col_map: Optional[Dict[str, str]] = None,
             order: Optional[str] = None,
             limit: int = 0) -> Optional[List[Dict[str, Any]]]:
    """Read rows from Supabase REST API.  Returns *None* on any failure
    so callers can fall back to CSV / Sheets transparently."""
    if not _SUPABASE_OK:
        return None
    try:
        sb = _sb_client()
        PAGE = 1000
        all_rows: List[Dict[str, Any]] = []
        offset = 0
        while True:
            q = sb.table(table).select(columns).range(offset, offset + PAGE - 1)
            if order:
                q = q.order(order)
            if filters:
                for col, expr in filters.items():
                    op, val = expr.split(".", 1)
                    if op == 'in':
                        val_list = [v.strip() for v in val.strip('()').split(',') if v.strip()]
                        q = q.in_(col, val_list)
                    else:
                        q = getattr(q, op)(col, val)
            batch = q.execute().data
            all_rows.extend(batch)
            if len(batch) < PAGE or (limit and len(all_rows) >= limit):
                break
            offset += PAGE
        if limit:
            all_rows = all_rows[:limit]
        if col_map:
            all_rows = [{col_map.get(k, k): v for k, v in r.items()} for r in all_rows]
        return all_rows
    except Exception:
        return None


_PLAYER_NAMES_CACHE: Dict[int, Tuple[float, Dict[int, str]]] = {}  # season -> (ts, {player_id: name})

def _load_player_names_db(season: int) -> Dict[int, str]:
    """Load player_id->name map from the players table for the given season."""
    now = time.time()
    cached = _PLAYER_NAMES_CACHE.get(season)
    if cached and (now - cached[0]) < 21600:
        return cached[1]
    rows = _sb_read('players', columns='player_id,player',
                     filters={'season': f'eq.{season}'})
    out: Dict[int, str] = {}
    if rows:
        for r in rows:
            pid = _safe_int(r.get('player_id'))
            name = str(r.get('player') or '').strip()
            if pid and name:
                out[pid] = name
    _PLAYER_NAMES_CACHE[season] = (now, out)
    return out


# Reverse column maps: Supabase snake_case → original CSV/Sheets column names
_COL_MAP_TEAMS = {
    "team": "Team", "team_id": "TeamID", "name": "Name",
    "logo": "Logo", "color": "Color", "active": "Active",
}
_COL_MAP_LAST_DATES = {"season": "Season", "last_date": "Last_Date"}
_COL_MAP_BOX_IDS = {
    "x": "x", "y": "y", "box_id": "BoxID",
    "box_id_rev": "BoxID_rev", "box_size": "Boxsize",
}
_COL_MAP_SEASON_STATS = {
    "season": "Season", "season_state": "SeasonState",
    "strength_state": "StrengthState", "player_id": "PlayerID",
    "position": "Position", "gp": "GP", "plus_minus": "plusMinus",
    "blocked_shots": "blockedShots", "toi": "TOI",
    "i_goals": "iGoals", "assists1": "Assists1", "assists2": "Assists2",
    "i_corsi": "iCorsi", "i_fenwick": "iFenwick", "i_shots": "iShots",
    "ixg_f": "ixG_F", "ixg_s": "ixG_S", "ixg_f2": "ixG_F2",
    "pim_taken": "PIM_taken", "pim_drawn": "PIM_drawn",
    "hits": "Hits", "takeaways": "Takeaways", "giveaways": "Giveaways",
    "so_goal": "SO_Goal", "so_attempt": "SO_Attempt",
    "ca": "CA", "cf": "CF", "fa": "FA", "ff": "FF",
    "sa": "SA", "sf": "SF", "ga": "GA", "gf": "GF",
    "xga_f": "xGA_F", "xgf_f": "xGF_F", "xga_s": "xGA_S", "xgf_s": "xGF_S",
    "xga_f2": "xGA_F2", "xgf_f2": "xGF_F2",
    "pim_for": "PIM_for", "pim_against": "PIM_against",
}
_COL_MAP_SEASON_STATS_TEAMS = {
    "season": "Season", "season_state": "SeasonState",
    "strength_state": "StrengthState", "team": "Team",
    "gp": "GP", "toi": "TOI",
    "cf": "CF", "ca": "CA", "ff": "FF", "fa": "FA",
    "sf": "SF", "sa": "SA", "gf": "GF", "ga": "GA",
    "xgf_f": "xGF_F", "xga_f": "xGA_F", "xgf_s": "xGF_S", "xga_s": "xGA_S",
    "xgf_f2": "xGF_F2", "xga_f2": "xGA_F2",
    "pim_for": "PIM_for", "pim_against": "PIM_against",
}
_COL_MAP_PLAYER_PROJECTIONS = {
    "player_id": "PlayerID", "position": "Position", "game_no": "Game_No",
    "age": "Age", "rookie": "Rookie", "evo": "EVO", "evd": "EVD",
    "pp": "PP", "sh": "SH", "gsax": "GSAx",
}
_COL_MAP_RAPM_CORE = {
    "player_id": "PlayerID", "season": "Season",
    "strength_state": "StrengthState", "rates_totals": "Rates_Totals",
    "cf": "CF", "ca": "CA", "gf": "GF", "ga": "GA",
    "xgf": "xGF", "xga": "xGA",
    "pen_taken": "PEN_taken", "pen_drawn": "PEN_drawn",
    "c_plusminus": "C_plusminus", "g_plusminus": "G_plusminus",
    "xg_plusminus": "xG_plusminus", "pen_plusminus": "PEN_plusminus",
    "alpha_cf": "Alpha_CF", "alpha_gf": "Alpha_GF",
    "alpha_xgf": "Alpha_xGF", "alpha_pen": "Alpha_PEN",
}
_COL_MAP_RAPM_CONTEXT = {
    "player_id": "PlayerID", "season": "Season",
    "strength_state": "StrengthState", "minutes": "Minutes",
    "qot_blend_xg67_g33": "QoT_blend_xG67_G33",
    "qoc_blend_xg67_g33": "QoC_blend_xG67_G33",
    "zs_difficulty": "ZS_Difficulty",
}

# Columns that should be divided by (minutes/60) when converting Totals → Rates
_RAPM_VALUE_COLS = [
    'CF', 'CA', 'GF', 'GA', 'xGF', 'xGA',
    'PEN_taken', 'PEN_drawn',
    'C_plusminus', 'G_plusminus', 'xG_plusminus', 'PEN_plusminus',
    'PP_CF', 'PP_GF', 'PP_xGF', 'SH_CA', 'SH_GA', 'SH_xGA',
]


def _synthesize_rates_rows(
    totals_rows: List[Dict[str, Any]],
    ctx_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Create Rates rows from Totals rows by dividing value columns by (minutes/60).

    Z-score columns are copied from Totals (approximate).
    """
    mins_map: Dict[Tuple[str, str, str], float] = {}
    for cr in ctx_rows:
        key = (
            str(cr.get('PlayerID') or '').strip(),
            str(cr.get('Season') or '').strip(),
            str(cr.get('StrengthState') or '').strip(),
        )
        m = _parse_locale_float(cr.get('Minutes'))
        if m is not None and m > 0:
            mins_map[key] = float(m)

    rates: List[Dict[str, Any]] = []
    for r in totals_rows:
        rt = str(r.get('Rates_Totals') or r.get('Rates/Totals') or '').strip().lower()
        if not rt.startswith('tot'):
            continue
        key = (
            str(r.get('PlayerID') or '').strip(),
            str(r.get('Season') or '').strip(),
            str(r.get('StrengthState') or '').strip(),
        )
        mins = mins_map.get(key)
        if not mins or mins <= 0:
            continue
        factor = 60.0 / mins
        rate_row = dict(r)
        rate_row['Rates_Totals'] = 'Rates'
        for col in _RAPM_VALUE_COLS:
            v = _parse_locale_float(rate_row.get(col))
            if v is not None:
                rate_row[col] = v * factor
        rates.append(rate_row)
    return rates


def _sb_read_rapm(*, filters: Optional[Dict[str, str]] = None) -> Optional[List[Dict[str, Any]]]:
    """Read RAPM rows from Supabase, unpacking JSONB columns (stddev/zscore/pp_sh)
    back to flat dicts with original column names."""
    raw = _sb_read("rapm", filters=filters)
    if raw is None:
        return None
    out: List[Dict[str, Any]] = []
    for row in raw:
        r: Dict[str, Any] = {}
        for k, v in row.items():
            if k in ('stddev', 'zscore', 'pp_sh'):
                if isinstance(v, dict):
                    r.update(v)  # keys are already original names from seed
            else:
                r[_COL_MAP_RAPM_CORE.get(k, k)] = v
        out.append(r)
    return out


# Load Teams.csv (used for theming and lookups in templates)
def _load_teams_csv() -> List[Dict[str, str]]:
    # Try Supabase first
    rows = _sb_read("teams", col_map=_COL_MAP_TEAMS)
    if rows is not None:
        # Supabase returns Active as boolean; normalise to '0'/'1' strings
        for r in rows:
            v = r.get('Active')
            if isinstance(v, bool):
                r['Active'] = '1' if v else '0'
            elif v is not None:
                r['Active'] = str(v)
        return rows
    # Fallback to CSV
    paths = [
        os.path.join(os.path.dirname(__file__), '..', 'Teams.csv'),
        os.path.join(os.getcwd(), 'Teams.csv'),
    ]
    for p in paths:
        try:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    return [row for row in reader]
        except Exception:
            continue
    return []


TEAM_ROWS: List[Dict[str, str]] = _load_teams_csv()

# Load Last_date.csv mapping Season -> Last_Date (YYYY-MM-DD)
def _load_last_dates() -> Dict[int, str]:
    # Try Supabase first
    sb_rows = _sb_read("last_dates", col_map=_COL_MAP_LAST_DATES)
    if sb_rows is not None:
        out: Dict[int, str] = {}
        for r in sb_rows:
            try:
                s = r.get('Season')
                d = r.get('Last_Date')
                if s and d:
                    out[int(str(s).strip())] = str(d).strip()
            except Exception:
                continue
        return out
    # Fallback to CSV
    paths = [
        os.path.join(os.getcwd(), 'Last_date.csv'),
        os.path.join(os.path.dirname(__file__), '..', 'Last_date.csv'),
    ]
    out2: Dict[int, str] = {}
    for p in paths:
        try:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8', newline='') as f:
                    rdr = csv.DictReader(f)
                    for row in rdr:
                        try:
                            s = row.get('Season')
                            d = row.get('Last_Date')
                            if s and d:
                                out2[int(str(s).strip())] = str(d).strip()
                        except Exception:
                            continue
                break
        except Exception:
            continue
    return out2

LAST_DATES: Dict[int, str] = _load_last_dates()

# Lazy-loaded cache for BoxID lookups by (x,y)
_BOXID_MAP: Optional[Dict[Tuple[int, int], Tuple[str, str, int]]] = None

def _get_boxid_map() -> Dict[Tuple[int, int], Tuple[str, str, int]]:
    global _BOXID_MAP
    if _BOXID_MAP is not None:
        # If the cached map appears degenerate (e.g., only x==0 keys due to BOM/header issues), rebuild it
        try:
            if any(k[0] != 0 for k in _BOXID_MAP.keys()):
                return _BOXID_MAP
        except Exception:
            if _BOXID_MAP:
                return _BOXID_MAP
        _BOXID_MAP = None
    # Try Supabase first
    sb_rows = _sb_read("box_ids", col_map=_COL_MAP_BOX_IDS)
    if sb_rows is not None:
        mapping: Dict[Tuple[int, int], Tuple[str, str, int]] = {}
        for r in sb_rows:
            try:
                xi = int(float(r.get('x') or 0))
                yi = int(float(r.get('y') or 0))
                bid = str(r.get('BoxID') or '').strip() or None
                bre = str(r.get('BoxID_rev') or '').strip() or None
                bsi_raw = r.get('Boxsize')
                bsi = int(bsi_raw) if bsi_raw is not None else None
                if bid and bre and bsi is not None:
                    mapping[(xi, yi)] = (bid, bre, bsi)
            except Exception:
                continue
        _BOXID_MAP = mapping
        return mapping
    # Fallback to CSV
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'BoxID.csv'),
        os.path.join(os.getcwd(), 'BoxID.csv'),
    ]
    mapping2: Dict[Tuple[int, int], Tuple[str, str, int]] = {}
    for p in candidate_paths:
        try:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8', newline='') as f:
                    rdr = csv.DictReader(f)
                    # Build a case-insensitive and BOM-tolerant field map
                    raw_fields = rdr.fieldnames or []
                    fns = [fn.strip() for fn in raw_fields]
                    def norm_field(s: str) -> str:
                        return s.lstrip('\ufeff').lower().strip()
                    lower_map = {norm_field(fn): fn for fn in fns}
                    col_x = lower_map.get('x')
                    col_y = lower_map.get('y')
                    col_boxid = lower_map.get('boxid')
                    col_boxid_rev = lower_map.get('boxid_rev')
                    # Boxsize/BoxSize
                    col_boxsize = lower_map.get('boxsize')
                    for row in rdr:
                        try:
                            # Access with original name; if missing, attempt BOM-stripped variants
                            xi_raw = row.get(col_x) if col_x else (row.get('x') or row.get('\ufeffx'))
                            yi_raw = row.get(col_y) if col_y else row.get('y')
                            bid_raw = row.get(col_boxid) if col_boxid else (row.get('BoxID') or row.get('boxid'))
                            bre_raw = row.get(col_boxid_rev) if col_boxid_rev else (row.get('BoxID_rev') or row.get('boxid_rev'))
                            bsz_raw = row.get(col_boxsize) if col_boxsize else (row.get('Boxsize') or row.get('BoxSize') or row.get('boxsize'))
                            xi = int(float(xi_raw or 0))
                            yi = int(float(yi_raw or 0))
                            bid = str(bid_raw or '').strip() or None
                            bre = str(bre_raw or '').strip() or None
                            bsi = int(str(bsz_raw).strip()) if (bsz_raw is not None and str(bsz_raw).strip().lstrip('-').isdigit()) else None
                            if bid and bre and bsi is not None:
                                mapping2[(xi, yi)] = (bid, bre, bsi)
                        except Exception:
                            continue
                break
        except Exception:
            continue
    _BOXID_MAP = mapping2
    return mapping2


# --- Projections helpers ---
def _static_path(*parts: str) -> str:
    try:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))
        return os.path.join(base, *parts)
    except Exception:
        return os.path.join(os.getcwd(), *parts)


def _iter_csv_dict_rows(path: str, *, delimiter: str = ',', encoding: str = 'utf-8-sig') -> Iterator[Dict[str, Any]]:
    """Yield DictReader rows without materializing the full file in memory."""
    try:
        if not path or (not os.path.exists(path)):
            return
        with open(path, 'r', encoding=encoding, newline='') as f:
            rdr = csv.DictReader(f, delimiter=delimiter)
            for row in rdr:
                if isinstance(row, dict):
                    yield row
    except Exception:
        return


def _load_rapm_player_rows_static(player_id: int, season: Optional[int]) -> List[Dict[str, Any]]:
    """Load RAPM rows for a single player (TTL cached). Tries Supabase, falls back to CSV."""
    global _RAPM_PLAYER_STATIC_CACHE
    try:
        ttl_s = max(30, int(os.getenv('RAPM_PLAYER_STATIC_CACHE_TTL_SECONDS', '600') or '600'))
    except Exception:
        ttl_s = 600
    try:
        max_items = max(1, int(os.getenv('RAPM_PLAYER_STATIC_CACHE_MAX_ITEMS', '512') or '512'))
    except Exception:
        max_items = 512
    key = (int(player_id), int(season) if season is not None else None)
    now = time.time()
    _cache_prune_ttl_and_size(_RAPM_PLAYER_STATIC_CACHE, ttl_s=ttl_s, max_items=max_items)
    cached = _RAPM_PLAYER_STATIC_CACHE.get(key)
    if cached and (now - cached[0]) < ttl_s:
        return cached[1]

    # Try Supabase first
    sb_filters: Dict[str, str] = {"player_id": f"eq.{int(player_id)}"}
    if season is not None:
        sb_filters["season"] = f"eq.{int(season)}"
    sb_rows = _sb_read_rapm(filters=sb_filters)
    if sb_rows is not None:
        # Supabase only stores Totals rows; synthesize Rates from context minutes
        has_rates = any(
            str(r.get('Rates_Totals') or '').strip().lower().startswith('rate')
            for r in sb_rows
        )
        if not has_rates and sb_rows:
            ctx_filters: Dict[str, str] = {"player_id": f"eq.{int(player_id)}"}
            if season is not None:
                ctx_filters["season"] = f"eq.{int(season)}"
            ctx_rows = _sb_read("rapm_context", filters=ctx_filters, col_map=_COL_MAP_RAPM_CONTEXT)
            if ctx_rows:
                sb_rows = sb_rows + _synthesize_rates_rows(sb_rows, ctx_rows)
        _cache_set_multi_bounded(_RAPM_PLAYER_STATIC_CACHE, key, sb_rows, ttl_s=ttl_s, max_items=max_items)
        return sb_rows

    # Fallback to CSV
    path = _static_path('rapm', 'rapm.csv')
    out: List[Dict[str, Any]] = []
    pid_s = str(int(player_id))
    for r in _iter_csv_dict_rows(path, delimiter=',', encoding='utf-8-sig'):
        try:
            if str(r.get('PlayerID') or '').strip() != pid_s:
                continue
            if season is not None:
                try:
                    if int(str(r.get('Season') or '').strip()) != int(season):
                        continue
                except Exception:
                    continue
            out.append(r)
        except Exception:
            continue

    _cache_set_multi_bounded(_RAPM_PLAYER_STATIC_CACHE, key, out, ttl_s=ttl_s, max_items=max_items)
    return out


def _load_context_player_rows_static(player_id: int, season: Optional[int]) -> List[Dict[str, Any]]:
    """Load Context rows for a single player (TTL cached). Tries Supabase, falls back to CSV."""
    global _CONTEXT_PLAYER_STATIC_CACHE
    try:
        ttl_s = max(30, int(os.getenv('CONTEXT_PLAYER_STATIC_CACHE_TTL_SECONDS', '600') or '600'))
    except Exception:
        ttl_s = 600
    try:
        max_items = max(1, int(os.getenv('CONTEXT_PLAYER_STATIC_CACHE_MAX_ITEMS', '512') or '512'))
    except Exception:
        max_items = 512
    key = (int(player_id), int(season) if season is not None else None)
    now = time.time()
    _cache_prune_ttl_and_size(_CONTEXT_PLAYER_STATIC_CACHE, ttl_s=ttl_s, max_items=max_items)
    cached = _CONTEXT_PLAYER_STATIC_CACHE.get(key)
    if cached and (now - cached[0]) < ttl_s:
        return cached[1]

    # Try Supabase first
    sb_filters: Dict[str, str] = {"player_id": f"eq.{int(player_id)}"}
    if season is not None:
        sb_filters["season"] = f"eq.{int(season)}"
    sb_rows = _sb_read("rapm_context", filters=sb_filters, col_map=_COL_MAP_RAPM_CONTEXT)
    if sb_rows is not None:
        _cache_set_multi_bounded(_CONTEXT_PLAYER_STATIC_CACHE, key, sb_rows, ttl_s=ttl_s, max_items=max_items)
        return sb_rows

    # Fallback to CSV
    path = _static_path('rapm', 'context.csv')
    out: List[Dict[str, Any]] = []
    pid_s = str(int(player_id))
    for r in _iter_csv_dict_rows(path, delimiter=',', encoding='utf-8-sig'):
        try:
            if str(r.get('PlayerID') or '').strip() != pid_s:
                continue
            if season is not None:
                try:
                    if int(str(r.get('Season') or '').strip()) != int(season):
                        continue
                except Exception:
                    continue
            out.append(r)
        except Exception:
            continue

    _cache_set_multi_bounded(_CONTEXT_PLAYER_STATIC_CACHE, key, out, ttl_s=ttl_s, max_items=max_items)
    return out


def _normalize_season_id_list(values: Any, *, default: Optional[int] = None) -> List[int]:
    if isinstance(values, (list, tuple, set)):
        raw_values = list(values)
    else:
        raw_values = str(values or '').split(',')

    out: List[int] = []
    seen: set[int] = set()
    for raw in raw_values:
        season_i = _safe_int(raw)
        if not season_i or season_i <= 0:
            continue
        season_i = int(season_i)
        if season_i in seen:
            continue
        seen.add(season_i)
        out.append(season_i)

    out.sort()
    if out:
        return out

    default_i = _safe_int(default)
    if default_i and default_i > 0:
        return [int(default_i)]
    return []


def _parse_request_season_ids(value: Any, *, default: Optional[int] = None) -> List[int]:
    return _normalize_season_id_list(value, default=default)


def _primary_season_id(values: Any, *, default: Optional[int] = None) -> Optional[int]:
    season_ids = _normalize_season_id_list(values, default=default)
    if season_ids:
        return int(season_ids[-1])
    return None


def _row_season_in_selected(row: Dict[str, Any], season_ids: Sequence[int]) -> bool:
    season_set = set(_normalize_season_id_list(season_ids))
    if not season_set:
        return True
    row_season = _safe_int(row.get('Season') or row.get('season'))
    return bool(row_season and int(row_season) in season_set)


def _aggregate_numeric_row_fields(
    rows: Sequence[Dict[str, Any]],
    columns: Sequence[str],
    *,
    mode: str = 'avg',
    weight_field: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    if len(rows) == 1:
        return dict(rows[0])

    out = dict(rows[-1])
    mode_norm = str(mode or 'avg').strip().lower()

    for col in columns:
        vals: List[float] = []
        weighted_vals: List[Tuple[float, float]] = []
        for row in rows:
            val = _parse_locale_float(row.get(col))
            if val is None:
                continue
            fv = float(val)
            if not math.isfinite(fv):
                continue
            vals.append(fv)
            if weight_field:
                weight = _parse_locale_float(row.get(weight_field))
                fw = float(weight) if weight is not None else 0.0
                if math.isfinite(fw) and fw > 0:
                    weighted_vals.append((fv, fw))
        if not vals:
            out[col] = None
            continue
        if mode_norm == 'sum':
            out[col] = sum(vals)
            continue
        if mode_norm == 'weighted' and weighted_vals:
            total_weight = sum(weight for _, weight in weighted_vals)
            out[col] = (sum(value * weight for value, weight in weighted_vals) / total_weight) if total_weight > 0 else None
            continue
        out[col] = sum(vals) / float(len(vals))

    return out


def _load_all_rosters_for_seasons_cached(season_ids: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    roster_map: Dict[int, Dict[str, Any]] = {}
    for season_id in _normalize_season_id_list(season_ids):
        try:
            season_roster = _load_all_rosters_for_season_cached(int(season_id)) or {}
        except Exception:
            season_roster = {}
        for pid, info in season_roster.items():
            try:
                pid_i = int(pid)
            except Exception:
                continue
            roster_map[pid_i] = info
    return roster_map


def _load_player_names_for_seasons(season_ids: Sequence[int]) -> Dict[int, str]:
    names: Dict[int, str] = {}
    for season_id in _normalize_season_id_list(season_ids):
        try:
            db_names = _load_player_names_db(int(season_id)) or {}
        except Exception:
            db_names = {}
        for pid, name in db_names.items():
            try:
                pid_i = int(pid)
            except Exception:
                continue
            if name:
                names[pid_i] = str(name)
        if db_names:
            continue
        try:
            roster = _load_all_rosters_for_season_cached(int(season_id)) or {}
        except Exception:
            roster = {}
        for pid, info in roster.items():
            try:
                pid_i = int(pid)
            except Exception:
                continue
            name = str((info or {}).get('name') or '').strip()
            if name:
                names[pid_i] = name
    return names


def _load_player_info_for_seasons(season_ids: Sequence[int]) -> Dict[str, Dict[str, str]]:
    info_by_pid: Dict[str, Dict[str, str]] = {}
    normalized = _normalize_season_id_list(season_ids)
    for season_id in normalized:
        try:
            rows = _sb_read(
                'players',
                columns='player_id,player,position',
                filters={'season': f'eq.{int(season_id)}'},
            )
        except Exception:
            rows = []
        for row in rows or []:
            pid = _safe_int(row.get('player_id'))
            if not pid or pid <= 0:
                continue
            key = str(pid)
            rec = info_by_pid.get(key)
            if rec is None:
                rec = {'name': '', 'position': ''}
                info_by_pid[key] = rec
            name = str(row.get('player') or '').strip()
            pos = str(row.get('position') or '').strip().upper()
            if name and not rec['name']:
                rec['name'] = name
            if pos and not rec['position']:
                rec['position'] = pos
    if info_by_pid:
        return info_by_pid

    roster_names = _load_player_names_for_seasons(normalized)
    for pid, name in roster_names.items():
        info_by_pid[str(pid)] = {'name': name, 'position': ''}
    return info_by_pid


def _load_player_info_targeted(player_ids: Sequence) -> Dict[str, Dict[str, str]]:
    """Fetch name/position for a specific set of player IDs in one small query.

    Much faster than _load_player_info_for_seasons when only a handful of players
    are needed (e.g. the 5 selected players, or a team roster of ~25 players).
    Falls back gracefully to an empty dict if the table is unreachable.
    """
    if not player_ids:
        return {}
    pid_list = sorted(set(int(p) for p in player_ids if _safe_int(p)))
    if not pid_list:
        return {}
    pid_filter = ','.join(str(p) for p in pid_list)
    rows = _sb_read('players', columns='player_id,player,position',
                    filters={'player_id': f'in.({pid_filter})'})
    info: Dict[str, Dict[str, str]] = {}
    for row in rows or []:
        pid = _safe_int(row.get('player_id'))
        if not pid:
            continue
        key = str(pid)
        if key not in info:
            info[key] = {
                'name': str(row.get('player') or '').strip(),
                'position': str(row.get('position') or '').strip().upper(),
            }
    return info


def _get_team_pids_for_seasons(team: str, season_ids: Sequence[int]) -> set:
    """Return the set of player IDs who played for *team* across the given seasons.

    Uses the game_data table which is already filtered by team+season, so this
    is a small, fast query even for multiple seasons.
    """
    team_pids: set = set()
    for season_id in season_ids:
        gd_rows = _sb_read(
            'game_data',
            columns='player_id',
            filters={'season': f'eq.{int(season_id)}', 'team': f'eq.{team}'},
        )
        if gd_rows:
            team_pids.update(int(r['player_id']) for r in gd_rows if r.get('player_id'))
    return team_pids


def _apply_lt_strength_filter(shift_rows: Sequence[Dict[str, Any]], strength: str) -> List[Dict[str, Any]]:
    rows = list(shift_rows or [])
    if not strength or strength.lower() == 'all':
        return rows
    strength_sets = {
        '5v5': {'5v5'},
        'PP': {'5v4', '5v3', '4v3'},
        'SH': {'4v5', '3v5', '3v4'},
    }
    if strength in strength_sets:
        allowed = strength_sets[strength]
        return [row for row in rows if str(row.get('strength_state', '')) in allowed]
    if strength == 'Other':
        all_special = {'5v5', '5v4', '5v3', '4v3', '4v5', '3v5', '3v4'}
        return [row for row in rows if str(row.get('strength_state', '')) not in all_special]
    return rows


def _accumulate_line_tool_combo(acc: Dict[str, Any], row: Dict[str, Any]) -> None:
    acc['gp'] += int(row.get('gp') or 0)
    acc['toi'] += float(row.get('toi') or 0.0)
    for key in ('cf', 'ca', 'ff', 'fa', 'sf', 'sa', 'gf', 'ga'):
        acc[key] += int(row.get(key) or 0)
    acc['xgf'] += float(row.get('xgf') or 0.0)
    acc['xga'] += float(row.get('xga') or 0.0)


def _finalize_line_tool_combo(team: str, players: Sequence[str], acc: Dict[str, Any]) -> Dict[str, Any]:
    cf = int(acc.get('cf') or 0)
    ca = int(acc.get('ca') or 0)
    ff = int(acc.get('ff') or 0)
    fa = int(acc.get('fa') or 0)
    sf = int(acc.get('sf') or 0)
    sa = int(acc.get('sa') or 0)
    gf = int(acc.get('gf') or 0)
    ga = int(acc.get('ga') or 0)
    xgf_v = round(float(acc.get('xgf') or 0.0), 2)
    xga_v = round(float(acc.get('xga') or 0.0), 2)
    sh_pct = round(100 * gf / max(sf, 1), 1)
    sv_pct = round(100 * (1 - ga / max(sa, 1)), 1)
    return {
        'players': list(players),
        'team': team,
        'gp': int(acc.get('gp') or 0),
        'toi': round(float(acc.get('toi') or 0.0), 1),
        'cf': cf,
        'ca': ca,
        'cfPct': round(100 * cf / max(cf + ca, 1), 1),
        'ff': ff,
        'fa': fa,
        'ffPct': round(100 * ff / max(ff + fa, 1), 1),
        'sf': sf,
        'sa': sa,
        'sfPct': round(100 * sf / max(sf + sa, 1), 1),
        'gf': gf,
        'ga': ga,
        'gfPct': round(100 * gf / max(gf + ga, 1), 1),
        'xgf': xgf_v,
        'xga': xga_v,
        'xgfPct': round(100 * xgf_v / max(xgf_v + xga_v, 0.001), 1),
        'shPct': sh_pct,
        'svPct': sv_pct,
        'pdo': round(sh_pct + sv_pct, 1),
    }


def _merge_line_tool_detail_counts(target: Dict[str, Dict[str, Any]], source: Dict[str, Dict[str, Any]]) -> None:
    for zone_id, src in (source or {}).items():
        dest = target.setdefault(zone_id, {'count': 0, 'fenwick': 0, 'shots': 0, 'goals': 0, 'xg': 0.0})
        dest['count'] += int(src.get('count') or 0)
        dest['fenwick'] += int(src.get('fenwick') or 0)
        dest['shots'] += int(src.get('shots') or 0)
        dest['goals'] += int(src.get('goals') or 0)
        dest['xg'] += float(src.get('xg') or 0.0)


def _iter_seasonstats_static_rows(*, season: Optional[int] = None, seasons: Optional[Sequence[int]] = None, skip_season: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Yield SeasonStats rows. Tries Supabase for filtered queries; CSV for full scans."""
    seasons_norm = _normalize_season_id_list(seasons, default=season)

    # For filtered queries (small result sets), try Supabase first.
    if seasons_norm:
        sb_rows_all: List[Dict[str, Any]] = []
        missing_csv_seasons: List[int] = []
        for season_i in seasons_norm:
            sb_rows = _sb_read("season_stats", filters={"season": f"eq.{int(season_i)}"}, col_map=_COL_MAP_SEASON_STATS)
            if sb_rows is None:
                missing_csv_seasons.append(int(season_i))
                continue
            sb_rows_all.extend(sb_rows)
        if not missing_csv_seasons:
            yield from sb_rows_all
            return

        # Mixed-source fallback: keep Supabase rows for available seasons
        # and only pull missing seasons from CSV.
        if sb_rows_all:
            yield from sb_rows_all
        missing_set = set(missing_csv_seasons)
        path = _static_path('nhl_seasonstats.csv')
        for r in _iter_csv_dict_rows(path, delimiter=',', encoding='utf-8-sig'):
            try:
                s = int(str(r.get('Season') or '').strip())
            except Exception:
                continue
            if s in missing_set:
                yield r
        return

    # Full scan or Supabase unavailable: use CSV
    path = _static_path('nhl_seasonstats.csv')
    season_set = set(seasons_norm)
    skip_i = int(skip_season) if skip_season is not None else None
    for r in _iter_csv_dict_rows(path, delimiter=',', encoding='utf-8-sig'):
        if not season_set and skip_i is None:
            yield r
            continue
        try:
            s = int(str(r.get('Season') or '').strip())
        except Exception:
            continue
        if skip_i is not None and s == skip_i:
            continue
        if season_set and s not in season_set:
            continue
        yield r


def _iter_teamseasonstats_static_rows(*, season: Optional[int] = None, seasons: Optional[Sequence[int]] = None, skip_season: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Yield Team SeasonStats rows. Supabase for all queries (small table ~5K rows), CSV fallback."""
    seasons_norm = _normalize_season_id_list(seasons, default=season)

    # Team stats table is small enough for Supabase REST.
    if seasons_norm:
        sb_rows_all: List[Dict[str, Any]] = []
        sb_ok = True
        for season_i in seasons_norm:
            sb_rows = _sb_read("season_stats_teams", filters={"season": f"eq.{int(season_i)}"}, col_map=_COL_MAP_SEASON_STATS_TEAMS)
            if sb_rows is None:
                sb_ok = False
                break
            sb_rows_all.extend(sb_rows)
        if sb_ok:
            yield from sb_rows_all
            return

    sb_filters: Optional[Dict[str, str]] = None
    if skip_season is not None:
        sb_filters = {"season": f"neq.{int(skip_season)}"}
    sb_rows = _sb_read("season_stats_teams", filters=sb_filters, col_map=_COL_MAP_SEASON_STATS_TEAMS)
    if sb_rows is not None:
        yield from sb_rows
        return

    # Fallback to CSV
    path = _static_path('nhl_seasonstats_teams.csv')
    season_set = set(seasons_norm)
    skip_i = int(skip_season) if skip_season is not None else None
    for r in _iter_csv_dict_rows(path, delimiter=',', encoding='utf-8-sig'):
        if not season_set and skip_i is None:
            yield r
            continue
        try:
            s = int(str(r.get('Season') or '').strip())
        except Exception:
            continue
        if skip_i is not None and s == skip_i:
            continue
        if season_set and s not in season_set:
            continue
        yield r


def _build_seasonstats_agg(
    *,
    scope: str,
    season_int: Optional[int] = None,
    season_ids: Optional[Sequence[int]] = None,
    season_state: str,
    strength_state: str,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, str]]:
    """Build (and cache) per-player aggregates for SeasonStats under the requested filters.

    This avoids scanning and materializing the full CSV for every request.
    """
    global _SEASONSTATS_AGG_CACHE
    try:
        ttl_s = max(30, int(os.getenv('SEASONSTATS_AGG_CACHE_TTL_SECONDS', '1800') or '1800'))
    except Exception:
        ttl_s = 1800

    try:
        max_items = max(1, int(os.getenv('SEASONSTATS_AGG_CACHE_MAX_ITEMS', '6') or '6'))
    except Exception:
        max_items = 6

    scope_norm = (scope or 'season').strip().lower()
    if scope_norm not in {'season', 'career'}:
        scope_norm = 'season'
    ss_norm = (season_state or 'regular').strip().lower()
    if ss_norm not in {'regular', 'playoffs', 'all'}:
        ss_norm = 'regular'
    st_norm = (strength_state or '5v5').strip()
    if st_norm not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        st_norm = '5v5'

    season_ids_norm = _normalize_season_id_list(season_ids, default=season_int)
    primary_season_int = _primary_season_id(season_ids_norm, default=season_int) or 0

    key = (
        scope_norm,
        tuple(season_ids_norm) if scope_norm == 'season' else (),
        int(primary_season_int or 0),
        ss_norm,
        st_norm,
    )
    now = time.time()
    _cache_prune_ttl_and_size(_SEASONSTATS_AGG_CACHE, ttl_s=ttl_s, max_items=max_items)
    cached = _SEASONSTATS_AGG_CACHE.get(key)
    if cached and (now - cached[0]) < ttl_s:
        return cached[1], cached[2]

    # Best-effort on-disk cache (helps on Render where workers may restart).
    cache_path = None
    try:
        base = os.getenv('XG_CACHE_DIR')
        if not base:
            base = _disk_cache_base()
        os.makedirs(base, exist_ok=True)
        key_bytes = ('|'.join(map(str, key)) + '|v2').encode('utf-8', errors='ignore')
        h = hashlib.sha1(key_bytes).hexdigest()  # nosec - non-crypto use (filename)
        cache_path = os.path.join(base, f'seasonstats_agg_{h}.pkl.gz')
        if os.path.exists(cache_path):
            mtime = os.path.getmtime(cache_path)
            if (now - float(mtime)) < float(ttl_s):
                with gzip.open(cache_path, 'rb') as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, tuple) and len(loaded) == 2:
                    agg0, pos0 = loaded
                    if isinstance(agg0, dict) and isinstance(pos0, dict):
                        _cache_set_multi_bounded(_SEASONSTATS_AGG_CACHE, key, agg0, pos0, ttl_s=ttl_s, max_items=max_items)
                        return agg0, pos0
    except Exception:
        cache_path = None

    def _iter_rows() -> Iterable[Dict[str, Any]]:
        if scope_norm == 'career':
            return _iter_seasonstats_static_rows()

        # scope == season
        return _iter_seasonstats_static_rows(seasons=season_ids_norm or [int(primary_season_int)])

    def _flt(v: Any) -> float:
        x = _parse_locale_float(v)
        return float(x) if x is not None else 0.0

    def _i(v: Any) -> int:
        return int(_safe_int(v) or 0)

    agg: Dict[int, Dict[str, Any]] = {}
    pos_group_by_pid: Dict[int, str] = {}
    gp_max_by_key: Dict[Tuple[int, int, str], int] = {}

    # Stream rows and aggregate.
    for r in _iter_rows():
        try:
            # Sheets-based SeasonStats tabs often omit some columns (e.g. Season) and may vary casing.
            pos = str(
                (r.get('Position') or _ci_get(r, 'Position') or _ci_get(r, 'position') or _ci_get(r, 'positionCode') or _ci_get(r, 'Pos') or '')
            ).strip().upper()
            if pos.startswith('G'):
                continue

            season_row = None
            try:
                season_row = int(str(r.get('Season') or _ci_get(r, 'Season') or '').strip())
            except Exception:
                season_row = None
            if season_row is None:
                # Sheets6 can be a single-season tab and may omit Season.
                # For career scope, those rows represent 20252026.
                season_row = 20252026 if scope_norm == 'career' else int(primary_season_int)

            ss_raw = str(r.get('SeasonState') or _ci_get(r, 'SeasonState') or _ci_get(r, 'seasonState') or '').strip().lower()
            if ss_raw in {'2', 'reg', 'regular', 'regularseason', 'regular_season'}:
                ss = 'regular'
            elif ss_raw in {'3', 'po', 'playoffs', 'playoff'}:
                ss = 'playoffs'
            else:
                ss = ss_raw or 'regular'

            st = str(r.get('StrengthState') or _ci_get(r, 'StrengthState') or _ci_get(r, 'strengthState') or '').strip() or 'Other'
            if ss_norm != 'all' and ss != ss_norm:
                continue
            if st_norm != 'all' and st != st_norm:
                continue

            pid_i = _i(r.get('PlayerID') or _ci_get(r, 'PlayerID') or _ci_get(r, 'playerId'))
            if pid_i <= 0:
                continue

            gp_row = _i(r.get('GP'))
            k = (pid_i, int(season_row), str(ss))
            prev_gp = gp_max_by_key.get(k)
            if prev_gp is None or gp_row > prev_gp:
                gp_max_by_key[k] = gp_row

            if pid_i not in pos_group_by_pid:
                pos_group_by_pid[pid_i] = 'D' if pos.startswith('D') else 'F'

            d = agg.setdefault(pid_i, {
                'GP': 0,
                'TOI': 0.0,
                'iGoals': 0.0,
                'Assists1': 0.0,
                'Assists2': 0.0,
                'iShots': 0.0,
                'iFenwick': 0.0,
                'ixG_S': 0.0,
                'ixG_F': 0.0,
                'ixG_F2': 0.0,
                # on-ice
                'CA': 0.0,
                'CF': 0.0,
                'FA': 0.0,
                'FF': 0.0,
                'SA': 0.0,
                'SF': 0.0,
                'GA': 0.0,
                'GF': 0.0,
                'xGA_S': 0.0,
                'xGF_S': 0.0,
                'xGA_F': 0.0,
                'xGF_F': 0.0,
                'xGA_F2': 0.0,
                'xGF_F2': 0.0,
                # misc
                'PIM_taken': 0.0,
                'PIM_drawn': 0.0,
                'PIM_for': 0.0,
                'PIM_against': 0.0,
                'Hits': 0.0,
                'Takeaways': 0.0,
                'Giveaways': 0.0,
            })

            d['TOI'] = float(d.get('TOI') or 0.0) + _flt(r.get('TOI') or _ci_get(r, 'TOI'))
            d['iGoals'] = float(d.get('iGoals') or 0.0) + _flt(r.get('iGoals') or _ci_get(r, 'iGoals'))
            d['Assists1'] = float(d.get('Assists1') or 0.0) + _flt(r.get('Assists1') or _ci_get(r, 'Assists1'))
            d['Assists2'] = float(d.get('Assists2') or 0.0) + _flt(r.get('Assists2') or _ci_get(r, 'Assists2'))
            d['iShots'] = float(d.get('iShots') or 0.0) + _flt(r.get('iShots') or _ci_get(r, 'iShots'))
            d['iFenwick'] = float(d.get('iFenwick') or 0.0) + _flt(r.get('iFenwick') or _ci_get(r, 'iFenwick'))
            d['ixG_S'] = float(d.get('ixG_S') or 0.0) + _flt(r.get('ixG_S') or _ci_get(r, 'ixG_S'))
            d['ixG_F'] = float(d.get('ixG_F') or 0.0) + _flt(r.get('ixG_F') or _ci_get(r, 'ixG_F'))
            d['ixG_F2'] = float(d.get('ixG_F2') or 0.0) + _flt(r.get('ixG_F2') or _ci_get(r, 'ixG_F2'))

            d['CA'] = float(d.get('CA') or 0.0) + _flt(r.get('CA') or _ci_get(r, 'CA'))
            d['CF'] = float(d.get('CF') or 0.0) + _flt(r.get('CF') or _ci_get(r, 'CF'))
            d['FA'] = float(d.get('FA') or 0.0) + _flt(r.get('FA') or _ci_get(r, 'FA'))
            d['FF'] = float(d.get('FF') or 0.0) + _flt(r.get('FF') or _ci_get(r, 'FF'))
            d['SA'] = float(d.get('SA') or 0.0) + _flt(r.get('SA') or _ci_get(r, 'SA'))
            d['SF'] = float(d.get('SF') or 0.0) + _flt(r.get('SF') or _ci_get(r, 'SF'))
            d['GA'] = float(d.get('GA') or 0.0) + _flt(r.get('GA') or _ci_get(r, 'GA'))
            d['GF'] = float(d.get('GF') or 0.0) + _flt(r.get('GF') or _ci_get(r, 'GF'))
            d['xGA_S'] = float(d.get('xGA_S') or 0.0) + _flt(r.get('xGA_S') or _ci_get(r, 'xGA_S'))
            d['xGF_S'] = float(d.get('xGF_S') or 0.0) + _flt(r.get('xGF_S') or _ci_get(r, 'xGF_S'))
            d['xGA_F'] = float(d.get('xGA_F') or 0.0) + _flt(r.get('xGA_F') or _ci_get(r, 'xGA_F'))
            d['xGF_F'] = float(d.get('xGF_F') or 0.0) + _flt(r.get('xGF_F') or _ci_get(r, 'xGF_F'))
            d['xGA_F2'] = float(d.get('xGA_F2') or 0.0) + _flt(r.get('xGA_F2') or _ci_get(r, 'xGA_F2'))
            d['xGF_F2'] = float(d.get('xGF_F2') or 0.0) + _flt(r.get('xGF_F2') or _ci_get(r, 'xGF_F2'))

            d['PIM_taken'] = float(d.get('PIM_taken') or 0.0) + _flt(r.get('PIM_taken') or _ci_get(r, 'PIM_taken'))
            d['PIM_drawn'] = float(d.get('PIM_drawn') or 0.0) + _flt(r.get('PIM_drawn') or _ci_get(r, 'PIM_drawn'))
            d['PIM_for'] = float(d.get('PIM_for') or 0.0) + _flt(r.get('PIM_for') or _ci_get(r, 'PIM_for'))
            d['PIM_against'] = float(d.get('PIM_against') or 0.0) + _flt(r.get('PIM_against') or _ci_get(r, 'PIM_against'))
            d['Hits'] = float(d.get('Hits') or 0.0) + _flt(r.get('Hits') or _ci_get(r, 'Hits'))
            d['Takeaways'] = float(d.get('Takeaways') or 0.0) + _flt(r.get('Takeaways') or _ci_get(r, 'Takeaways'))
            d['Giveaways'] = float(d.get('Giveaways') or 0.0) + _flt(r.get('Giveaways') or _ci_get(r, 'Giveaways'))
        except Exception:
            continue

    gp_sum_by_pid: Dict[int, int] = {}
    for (pid_k, _season_k, _ss_k), gp_k in gp_max_by_key.items():
        gp_sum_by_pid[pid_k] = int(gp_sum_by_pid.get(pid_k, 0) + int(gp_k or 0))
    for pid_k, d in agg.items():
        d['GP'] = int(gp_sum_by_pid.get(pid_k, 0))

    _cache_set_multi_bounded(_SEASONSTATS_AGG_CACHE, key, agg, pos_group_by_pid, ttl_s=ttl_s, max_items=max_items)

    if cache_path:
        try:
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump((agg, pos_group_by_pid), f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
    return agg, pos_group_by_pid


def _build_goalies_career_season_matrix(
    *,
    season_state: str,
    strength_state: str,
) -> Tuple[Dict[int, Dict[int, Dict[str, float]]], Dict[int, Tuple[float, float]]]:
    """Build per-goalie per-season aggregates for career calculations.

    Returns:
      - by_pid_season: { playerId: { seasonId: { SA, GA, FA, TOI, xGA_S, xGA_F, xGA_F2 } } }
      - league_sa_ga: { seasonId: (total_sa, total_ga) }
    """
    global _GOALIES_CAREER_MATRIX_CACHE
    try:
        ttl_s = max(30, int(os.getenv('SEASONSTATS_AGG_CACHE_TTL_SECONDS', '1800') or '1800'))
    except Exception:
        ttl_s = 1800

    ss_norm = (season_state or 'regular').strip().lower()
    if ss_norm not in {'regular', 'playoffs', 'all'}:
        ss_norm = 'regular'
    st_norm = (strength_state or '5v5').strip()
    if st_norm not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        st_norm = '5v5'

    key = (
        'goalies_career_matrix',
        ss_norm,
        st_norm,
    )
    now = time.time()
    cached = _GOALIES_CAREER_MATRIX_CACHE.get(key)
    if cached and (now - cached[0]) < ttl_s:
        return cached[1], cached[2]

    cache_path = None
    try:
        base = os.getenv('XG_CACHE_DIR')
        if not base:
            base = _disk_cache_base()
        os.makedirs(base, exist_ok=True)
        key_bytes = ('|'.join(map(str, key)) + '|v2').encode('utf-8', errors='ignore')
        h = hashlib.sha1(key_bytes).hexdigest()  # nosec - non-crypto use (filename)
        cache_path = os.path.join(base, f'goalies_career_matrix_{h}.pkl.gz')
        if os.path.exists(cache_path):
            mtime = os.path.getmtime(cache_path)
            if (now - float(mtime)) < float(ttl_s):
                with gzip.open(cache_path, 'rb') as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, tuple) and len(loaded) == 2:
                    a0, b0 = loaded
                    if isinstance(a0, dict) and isinstance(b0, dict):
                        _GOALIES_CAREER_MATRIX_CACHE[key] = (now, a0, b0)
                        return a0, b0
    except Exception:
        cache_path = None

    def _iter_rows() -> Iterable[Dict[str, Any]]:
        return _iter_seasonstats_static_rows()

    def _flt(v: Any) -> float:
        x = _parse_locale_float(v)
        return float(x) if x is not None else 0.0

    def _i(v: Any) -> int:
        return int(_safe_int(v) or 0)

    by_pid_season: Dict[int, Dict[int, Dict[str, float]]] = {}
    league_acc: Dict[int, Dict[str, float]] = {}

    for r in _iter_rows():
        try:
            pos = str(
                (r.get('Position') or _ci_get(r, 'Position') or _ci_get(r, 'position') or _ci_get(r, 'positionCode') or _ci_get(r, 'Pos') or '')
            ).strip().upper()
            if not pos.startswith('G'):
                continue

            season_row = None
            try:
                season_row = int(str(r.get('Season') or _ci_get(r, 'Season') or '').strip())
            except Exception:
                season_row = None
            if season_row is None:
                season_row = 20252026

            ss_raw = str(r.get('SeasonState') or _ci_get(r, 'SeasonState') or _ci_get(r, 'seasonState') or '').strip().lower()
            if ss_raw in {'2', 'reg', 'regular', 'regularseason', 'regular_season'}:
                ss = 'regular'
            elif ss_raw in {'3', 'po', 'playoffs', 'playoff'}:
                ss = 'playoffs'
            else:
                ss = ss_raw or 'regular'

            st = str(r.get('StrengthState') or _ci_get(r, 'StrengthState') or _ci_get(r, 'strengthState') or '').strip() or 'Other'
            if ss_norm != 'all' and ss != ss_norm:
                continue
            if st_norm != 'all' and st != st_norm:
                continue

            pid_i = _i(r.get('PlayerID') or _ci_get(r, 'PlayerID') or _ci_get(r, 'playerId'))
            if pid_i <= 0:
                continue

            pmap = by_pid_season.setdefault(pid_i, {})
            d = pmap.setdefault(int(season_row), {
                'TOI': 0.0,
                'FA': 0.0,
                'SA': 0.0,
                'GA': 0.0,
                'xGA_S': 0.0,
                'xGA_F': 0.0,
                'xGA_F2': 0.0,
            })
            d['TOI'] += _flt(r.get('TOI') or _ci_get(r, 'TOI'))
            d['FA'] += _flt(r.get('FA') or _ci_get(r, 'FA'))
            d['SA'] += _flt(r.get('SA') or _ci_get(r, 'SA'))
            d['GA'] += _flt(r.get('GA') or _ci_get(r, 'GA'))
            d['xGA_S'] += _flt(r.get('xGA_S') or _ci_get(r, 'xGA_S'))
            d['xGA_F'] += _flt(r.get('xGA_F') or _ci_get(r, 'xGA_F'))
            d['xGA_F2'] += _flt(r.get('xGA_F2') or _ci_get(r, 'xGA_F2'))

            la = league_acc.setdefault(int(season_row), {'SA': 0.0, 'GA': 0.0})
            la['SA'] += _flt(r.get('SA') or _ci_get(r, 'SA'))
            la['GA'] += _flt(r.get('GA') or _ci_get(r, 'GA'))
        except Exception:
            continue

    league_sa_ga: Dict[int, Tuple[float, float]] = {}
    for s, d in league_acc.items():
        try:
            league_sa_ga[int(s)] = (float(d.get('SA') or 0.0), float(d.get('GA') or 0.0))
        except Exception:
            league_sa_ga[int(s)] = (0.0, 0.0)

    _GOALIES_CAREER_MATRIX_CACHE[key] = (now, by_pid_season, league_sa_ga)
    if cache_path:
        try:
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump((by_pid_season, league_sa_ga), f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    return by_pid_season, league_sa_ga


def _load_rapm_static_csv() -> List[Dict[str, Any]]:
    """Load RAPM data (TTL cached). Prefers Supabase; CSV as fallback."""
    global _RAPM_STATIC_CACHE
    try:
        ttl_s = max(30, int(os.getenv('RAPM_STATIC_CACHE_TTL_SECONDS', '600') or '600'))
    except Exception:
        ttl_s = 600
    now = time.time()
    if _RAPM_STATIC_CACHE and (now - _RAPM_STATIC_CACHE[0]) < ttl_s:
        return _RAPM_STATIC_CACHE[1]

    # Try Supabase first (current-season Totals only)
    sb_rows = _sb_read_rapm()
    if sb_rows is not None:
        # Supabase stores only Totals; synthesize Rates from context minutes
        has_rates = any(
            str(r.get('Rates_Totals') or '').strip().lower().startswith('rate')
            for r in sb_rows
        )
        if not has_rates and sb_rows:
            ctx_rows = _load_context_static_csv()
            if ctx_rows:
                sb_rows = sb_rows + _synthesize_rates_rows(sb_rows, ctx_rows)
        _RAPM_STATIC_CACHE = (now, sb_rows)
        return sb_rows

    # Fallback to CSV
    path = _static_path('rapm', 'rapm.csv')
    rows: List[Dict[str, Any]] = []
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8', newline='') as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    if isinstance(row, dict):
                        rows.append(row)
            _RAPM_STATIC_CACHE = (now, rows)
            return rows
    except Exception:
        pass

    _RAPM_STATIC_CACHE = (now, [])
    return []


def _load_context_static_csv() -> List[Dict[str, Any]]:
    """Load context data (TTL cached). Prefers Supabase; CSV as fallback."""
    global _CONTEXT_STATIC_CACHE
    try:
        ttl_s = max(30, int(os.getenv('CONTEXT_STATIC_CACHE_TTL_SECONDS', '600') or '600'))
    except Exception:
        ttl_s = 600
    now = time.time()
    if _CONTEXT_STATIC_CACHE and (now - _CONTEXT_STATIC_CACHE[0]) < ttl_s:
        return _CONTEXT_STATIC_CACHE[1]

    # Try Supabase first
    sb_rows = _sb_read("rapm_context", col_map=_COL_MAP_RAPM_CONTEXT)
    if sb_rows is not None:
        _CONTEXT_STATIC_CACHE = (now, sb_rows)
        return sb_rows

    # Fallback to CSV
    path = _static_path('rapm', 'context.csv')
    rows: List[Dict[str, Any]] = []
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8', newline='') as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    if isinstance(row, dict):
                        rows.append(row)
            _CONTEXT_STATIC_CACHE = (now, rows)
            return rows
    except Exception:
        pass

    _CONTEXT_STATIC_CACHE = (now, [])
    return []


def _load_seasonstats_static_csv() -> List[Dict[str, Any]]:
    """Load season stats (TTL cached). Prefers CSV for full load; Supabase as fallback."""
    global _SEASONSTATS_STATIC_CACHE
    try:
        ttl_s = max(30, int(os.getenv('SEASONSTATS_STATIC_CACHE_TTL_SECONDS', '600') or '600'))
    except Exception:
        ttl_s = 600
    now = time.time()
    if _SEASONSTATS_STATIC_CACHE and (now - _SEASONSTATS_STATIC_CACHE[0]) < ttl_s:
        return _SEASONSTATS_STATIC_CACHE[1]

    # Prefer CSV for full load (136K+ rows is too slow over REST pagination)
    path = _static_path('nhl_seasonstats.csv')
    rows: List[Dict[str, Any]] = []
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8', newline='') as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    if isinstance(row, dict):
                        rows.append(row)
            _SEASONSTATS_STATIC_CACHE = (now, rows)
            return rows
    except Exception:
        pass

    # Fallback to Supabase if CSV not available
    sb_rows = _sb_read("season_stats", col_map=_COL_MAP_SEASON_STATS)
    if sb_rows is not None:
        _SEASONSTATS_STATIC_CACHE = (now, sb_rows)
        return sb_rows

    _SEASONSTATS_STATIC_CACHE = (now, [])
    return []


def _load_card_metrics_defs(card: str = 'skaters') -> Dict[str, Any]:
    """Load app/static/card_metrics.csv as card metric definitions.

    Returns:
      {
        'categories': [<category>...],
        'metrics': [
           {
             'id': 'Category|Metric',
             'category': 'Category',
             'metric': 'Metric',
             'name': 'Name',
             'calculation': '...',
             'default': bool,
             'place': 'L1'|'C1'|'R1'|'L2'|'C2'|'R2'|'L3'|'C3'|'R3'|'0',
           },
        ]
      }
    """
    global _CARD_METRICS_DEF_CACHE
    try:
        ttl_s = max(30, int(os.getenv('CARD_METRICS_DEF_CACHE_TTL_SECONDS', '600') or '600'))
    except Exception:
        ttl_s = 600
    now = time.time()
    card_norm = str(card or 'skaters').strip().lower() or 'skaters'

    cached = _CARD_METRICS_DEF_CACHE.get(card_norm)
    if cached and (now - cached[0]) < ttl_s:
        return cached[1]

    path = _static_path('card_metrics.csv')
    out: Dict[str, Any] = {'categories': [], 'metrics': []}
    try:
        if not os.path.exists(path):
            _CARD_METRICS_DEF_CACHE[card_norm] = (now, out)
            return out

        metrics: List[Dict[str, Any]] = []
        cats: List[str] = []
        seen_cat: set[str] = set()
        # Use utf-8-sig to tolerate UTF-8 BOM in headers (common on Windows).
        with open(path, 'r', encoding='utf-8-sig', newline='') as f:
            # card_metrics.csv is sometimes tab-delimited (Excel/Sheets export).
            # Auto-detect delimiter between ';' and '\t' based on the header line.
            try:
                first_line = f.readline()
                delim = '\t' if first_line.count('\t') > first_line.count(';') else ';'
                f.seek(0)
            except Exception:
                delim = ';'
            rdr = csv.DictReader(f, delimiter=delim)
            for row in rdr:
                if not isinstance(row, dict):
                    continue
                card = str(row.get('Card') or '').strip()
                if card and card.lower() != card_norm:
                    continue
                category = str(row.get('Category') or '').strip()
                metric = str(row.get('Metric') or '').strip()
                name = str(row.get('Name') or '').strip()
                calc = str(row.get('Calculation') or '').strip()
                place = str(row.get('Place') or '').strip() or '0'
                default_raw = str(row.get('Default') or '').strip()
                is_default = default_raw in {'1', 'true', 'True', 'YES', 'Yes', 'yes'}
                if not category or not metric:
                    continue

                metric_id = f"{category}|{metric}"
                metrics.append({
                    'id': metric_id,
                    'category': category,
                    'metric': metric,
                    'name': name or metric,
                    'calculation': calc,
                    'default': bool(is_default),
                    'place': place,
                    'link': str(row.get('Link') or row.get('link') or '').strip(),
                    'strengthCode': str(row.get('StrengthCode') or row.get('strengthCode') or '').strip(),
                    'positionCode': str(row.get('PositionCode') or row.get('positionCode') or row.get('') or '').strip(),
                })
                if category not in seen_cat:
                    seen_cat.add(category)
                    cats.append(category)

        out = {'categories': cats, 'metrics': metrics}
    except Exception:
        out = {'categories': [], 'metrics': []}

    _CARD_METRICS_DEF_CACHE[card_norm] = (now, out)
    return out


def _build_goalies_seasonstats_agg(
    *,
    scope: str,
    season_int: Optional[int] = None,
    season_ids: Optional[Sequence[int]] = None,
    season_state: str,
    strength_state: str,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, str]]:
    """Build (and cache) per-goalie aggregates for SeasonStats under the requested filters."""
    global _SEASONSTATS_AGG_CACHE
    try:
        ttl_s = max(30, int(os.getenv('SEASONSTATS_AGG_CACHE_TTL_SECONDS', '1800') or '1800'))
    except Exception:
        ttl_s = 1800

    scope_norm = (scope or 'season').strip().lower()
    if scope_norm not in {'season', 'career'}:
        scope_norm = 'season'
    ss_norm = (season_state or 'regular').strip().lower()
    if ss_norm not in {'regular', 'playoffs', 'all'}:
        ss_norm = 'regular'
    st_norm = (strength_state or '5v5').strip()
    if st_norm not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        st_norm = '5v5'

    season_ids_norm = _normalize_season_id_list(season_ids, default=season_int)
    primary_season_int = _primary_season_id(season_ids_norm, default=season_int) or 0

    key = (
        'goalies',
        scope_norm,
        tuple(season_ids_norm) if scope_norm == 'season' else (),
        int(primary_season_int or 0),
        ss_norm,
        st_norm,
    )
    now = time.time()
    cached = _SEASONSTATS_AGG_CACHE.get(key)
    if cached and (now - cached[0]) < ttl_s:
        return cached[1], cached[2]

    cache_path = None
    try:
        base = os.getenv('XG_CACHE_DIR')
        if not base:
            base = _disk_cache_base()
        os.makedirs(base, exist_ok=True)
        key_bytes = ('|'.join(map(str, key)) + '|v2').encode('utf-8', errors='ignore')
        h = hashlib.sha1(key_bytes).hexdigest()  # nosec - non-crypto use (filename)
        cache_path = os.path.join(base, f'goalies_seasonstats_agg_{h}.pkl.gz')
        if os.path.exists(cache_path):
            mtime = os.path.getmtime(cache_path)
            if (now - float(mtime)) < float(ttl_s):
                with gzip.open(cache_path, 'rb') as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, tuple) and len(loaded) == 2:
                    agg0, pos0 = loaded
                    if isinstance(agg0, dict) and isinstance(pos0, dict):
                        _SEASONSTATS_AGG_CACHE[key] = (now, agg0, pos0)
                        return agg0, pos0
    except Exception:
        cache_path = None

    def _iter_rows() -> Iterable[Dict[str, Any]]:
        if scope_norm == 'career':
            return _iter_seasonstats_static_rows()

        return _iter_seasonstats_static_rows(seasons=season_ids_norm or [int(primary_season_int)])

    def _flt(v: Any) -> float:
        x = _parse_locale_float(v)
        return float(x) if x is not None else 0.0

    def _i(v: Any) -> int:
        return int(_safe_int(v) or 0)

    agg: Dict[int, Dict[str, Any]] = {}
    pos_group_by_pid: Dict[int, str] = {}
    gp_max_by_key: Dict[Tuple[int, int, str], int] = {}

    for r in _iter_rows():
        try:
            pos = str(
                (r.get('Position') or _ci_get(r, 'Position') or _ci_get(r, 'position') or _ci_get(r, 'positionCode') or _ci_get(r, 'Pos') or '')
            ).strip().upper()
            if not pos.startswith('G'):
                continue

            season_row = None
            try:
                season_row = int(str(r.get('Season') or _ci_get(r, 'Season') or '').strip())
            except Exception:
                season_row = None
            if season_row is None:
                season_row = 20252026 if scope_norm == 'career' else int(primary_season_int)

            ss_raw = str(r.get('SeasonState') or _ci_get(r, 'SeasonState') or _ci_get(r, 'seasonState') or '').strip().lower()
            if ss_raw in {'2', 'reg', 'regular', 'regularseason', 'regular_season'}:
                ss = 'regular'
            elif ss_raw in {'3', 'po', 'playoffs', 'playoff'}:
                ss = 'playoffs'
            else:
                ss = ss_raw or 'regular'

            st = str(r.get('StrengthState') or _ci_get(r, 'StrengthState') or _ci_get(r, 'strengthState') or '').strip() or 'Other'
            if ss_norm != 'all' and ss != ss_norm:
                continue
            if st_norm != 'all' and st != st_norm:
                continue

            pid_i = _i(r.get('PlayerID') or _ci_get(r, 'PlayerID') or _ci_get(r, 'playerId'))
            if pid_i <= 0:
                continue

            gp_row = _i(r.get('GP'))
            k = (pid_i, int(season_row), str(ss))
            prev_gp = gp_max_by_key.get(k)
            if prev_gp is None or gp_row > prev_gp:
                gp_max_by_key[k] = gp_row

            if pid_i not in pos_group_by_pid:
                pos_group_by_pid[pid_i] = 'G'

            d = agg.setdefault(pid_i, {
                'GP': 0,
                'TOI': 0.0,
                'FA': 0.0,
                'SA': 0.0,
                'GA': 0.0,
                'xGA_S': 0.0,
                'xGA_F': 0.0,
                'xGA_F2': 0.0,
            })

            d['TOI'] = float(d.get('TOI') or 0.0) + _flt(r.get('TOI') or _ci_get(r, 'TOI'))
            d['FA'] = float(d.get('FA') or 0.0) + _flt(r.get('FA') or _ci_get(r, 'FA'))
            d['SA'] = float(d.get('SA') or 0.0) + _flt(r.get('SA') or _ci_get(r, 'SA'))
            d['GA'] = float(d.get('GA') or 0.0) + _flt(r.get('GA') or _ci_get(r, 'GA'))
            d['xGA_S'] = float(d.get('xGA_S') or 0.0) + _flt(r.get('xGA_S') or _ci_get(r, 'xGA_S'))
            d['xGA_F'] = float(d.get('xGA_F') or 0.0) + _flt(r.get('xGA_F') or _ci_get(r, 'xGA_F'))
            d['xGA_F2'] = float(d.get('xGA_F2') or 0.0) + _flt(r.get('xGA_F2') or _ci_get(r, 'xGA_F2'))
        except Exception:
            continue

    gp_sum_by_pid: Dict[int, int] = {}
    for (pid_k, _season_k, _ss_k), gp_k in gp_max_by_key.items():
        gp_sum_by_pid[pid_k] = int(gp_sum_by_pid.get(pid_k, 0) + int(gp_k or 0))
    for pid_k, d in agg.items():
        d['GP'] = int(gp_sum_by_pid.get(pid_k, 0))

    _SEASONSTATS_AGG_CACHE[key] = (now, agg, pos_group_by_pid)
    if cache_path:
        try:
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump((agg, pos_group_by_pid), f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
    return agg, pos_group_by_pid


def _edge_game_type(season_state: str) -> int:
    # NHL Edge endpoints use 2=regular, 3=playoffs.
    return 3 if str(season_state).strip().lower() == 'playoffs' else 2


def _edge_strength_code(strength_state: str) -> Optional[str]:
    s = str(strength_state or '').strip()
    if s == '5v5':
        return 'es'
    if s == 'PP':
        return 'pp'
    if s == 'SH':
        return 'pk'
    if s == 'all':
        return 'all'
    return None


def _edge_format_url(example_link: str, player_id: int, season_int: int, game_type: int) -> Optional[str]:
    link = str(example_link or '').strip()
    if not link:
        return None
    if link.startswith('api-web.nhle.com/'):
        link = 'https://' + link
    if link.startswith('http://'):
        link = 'https://' + link[len('http://'):]
    if not link.startswith('https://'):
        return None

    # Replace trailing /<player>/<season>/<gameType> if present.
    try:
        parts = link.rstrip('/').split('/')
        if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit() and parts[-3].isdigit():
            parts[-3] = str(int(player_id))
            parts[-2] = str(int(season_int))
            parts[-1] = str(int(game_type))
            return '/'.join(parts)
    except Exception:
        pass
    return link


def _edge_get_cached_json(url: str) -> Optional[Dict[str, Any]]:
    try:
        ttl_s = max(30, int(os.getenv('EDGE_API_CACHE_TTL_SECONDS', '3600') or '3600'))
    except Exception:
        ttl_s = 3600
    try:
        max_items = max(1, int(os.getenv('EDGE_API_CACHE_MAX_ITEMS', '256') or '256'))
    except Exception:
        max_items = 256
    now = time.time()
    _cache_prune_ttl_and_size(_EDGE_API_CACHE, ttl_s=ttl_s, max_items=max_items)
    cached = _EDGE_API_CACHE.get(url)
    if cached and (now - cached[0]) < ttl_s:
        try:
            return cached[1]
        except Exception:
            return None

    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return None
        j = r.json()
        if not isinstance(j, dict):
            return None
        _cache_set_multi_bounded(_EDGE_API_CACHE, url, j, ttl_s=ttl_s, max_items=max_items)
        return j
    except Exception:
        return None


def _ci_get(d: Dict[str, Any], key: str) -> Any:
    if key in d:
        return d.get(key)
    lk = str(key).lower()
    for k, v in d.items():
        if str(k).lower() == lk:
            return v
    return None


def _edge_pct_to_100(p: Any) -> Optional[float]:
    try:
        if p is None:
            return None
        f = float(p)
        if not math.isfinite(f):
            return None
        # NHL Edge uses 0..1 percentiles
        if 0.0 <= f <= 1.0:
            return 100.0 * f
        # Already 0..100
        if 0.0 <= f <= 100.0:
            return f
        return None
    except Exception:
        return None


def _edge_extract_value_and_pct(payload: Dict[str, Any], metric_key: str, strength_code: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """Extract a single metric value + percentile from an NHL Edge JSON payload.

    Supports:
      - dict-of-metrics (e.g. shotSpeedDetails.topShotSpeed)
      - list-of-strength-rows (e.g. zoneTimeDetails with strengthCode)
      - nested dict values with {value|imperial|metric, percentile}
      - scalar values with sibling <metric>Percentile
    """
    # 1) Direct hit in a nested dict of details.
    for v in payload.values():
        if isinstance(v, dict):
            node = _ci_get(v, metric_key)
            if isinstance(node, dict):
                val = _ci_get(node, 'imperial')
                if val is None:
                    val = _ci_get(node, 'value')
                if val is None:
                    val = _ci_get(node, 'metric')
                pct = _edge_pct_to_100(_ci_get(node, 'percentile'))
                try:
                    out_val = float(val) if val is not None else None
                except Exception:
                    out_val = None
                if out_val is not None and str(metric_key).endswith('Pctg') and 0.0 <= out_val <= 1.0:
                    out_val = 100.0 * out_val
                return (out_val, pct)
            if node is not None and not isinstance(node, (dict, list)):
                try:
                    out_val = float(node)
                    if str(metric_key).endswith('Pctg') and 0.0 <= out_val <= 1.0:
                        out_val = 100.0 * out_val
                    return (out_val, _edge_pct_to_100(_ci_get(v, f'{metric_key}Percentile')))
                except Exception:
                    return (None, _edge_pct_to_100(_ci_get(v, f'{metric_key}Percentile')))

    # 2) Strength-split list.
    rows: Optional[List[Dict[str, Any]]] = None
    for v in payload.values():
        if isinstance(v, list) and v and isinstance(v[0], dict) and any(str(k).lower() == 'strengthcode' for k in v[0].keys()):
            rows = v  # type: ignore[assignment]
            break
    if rows:
        wanted = strength_code
        row = None
        if wanted:
            for rr in rows:
                if str(rr.get('strengthCode') or '').lower() == str(wanted).lower():
                    row = rr
                    break
        if row is None:
            for rr in rows:
                if str(rr.get('strengthCode') or '').lower() == 'all':
                    row = rr
                    break
        if row is None:
            row = rows[0]

        # Scalar metric with separate percentile key
        val0 = _ci_get(row, metric_key)
        pct_raw = _ci_get(row, f'{metric_key}Percentile')
        if pct_raw is None and str(metric_key).endswith('Pctg'):
            pct_raw = _ci_get(row, f"{str(metric_key)[:-4]}Percentile")
        pct0 = _edge_pct_to_100(pct_raw)
        if isinstance(val0, dict):
            val = _ci_get(val0, 'imperial')
            if val is None:
                val = _ci_get(val0, 'value')
            if val is None:
                val = _ci_get(val0, 'metric')
            pct = _edge_pct_to_100(_ci_get(val0, 'percentile'))
            try:
                out_val = float(val) if val is not None else None
            except Exception:
                out_val = None
            if out_val is not None and str(metric_key).endswith('Pctg') and 0.0 <= out_val <= 1.0:
                out_val = 100.0 * out_val
            return (out_val, pct)
        try:
            out_val = float(val0) if val0 is not None else None
            if out_val is not None and str(metric_key).endswith('Pctg') and 0.0 <= out_val <= 1.0:
                out_val = 100.0 * out_val
            return (out_val, pct0)
        except Exception:
            return (None, pct0)

    return (None, None)


def _edge_extract_value_pct_avg(payload: Dict[str, Any], metric_key: str, strength_code: Optional[str]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract value + percentile + NHL/league average from an NHL Edge JSON payload.

    Many NHL Edge endpoints include league averages in fields like `<metricBase>LeagueAvg`.
    Example (zone time): `offensiveZonePctg` + `offensiveZoneLeagueAvg`.
    """
    val, pct = _edge_extract_value_and_pct(payload, metric_key, strength_code)

    def _coerce_avg(x: Any) -> Optional[float]:
        try:
            if x is None:
                return None
            f = float(x)
            if not math.isfinite(f):
                return None
            if str(metric_key).endswith('Pctg') and 0.0 <= f <= 1.0:
                return 100.0 * f
            return f
        except Exception:
            return None

    # Strength-split list rows (e.g., zoneTimeDetails).
    rows: Optional[List[Dict[str, Any]]] = None
    for v in payload.values():
        if isinstance(v, list) and v and isinstance(v[0], dict) and any(str(k).lower() == 'strengthcode' for k in v[0].keys()):
            rows = v  # type: ignore[assignment]
            break
    if rows:
        wanted = strength_code
        row = None
        if wanted:
            for rr in rows:
                if str(rr.get('strengthCode') or '').lower() == str(wanted).lower():
                    row = rr
                    break
        if row is None:
            for rr in rows:
                if str(rr.get('strengthCode') or '').lower() == 'all':
                    row = rr
                    break
        if row is None:
            row = rows[0]

        mk = str(metric_key)
        base = mk[:-4] if mk.endswith('Pctg') else mk
        avg_raw = (
            _ci_get(row, f'{mk}LeagueAvg')
            or _ci_get(row, f'{base}LeagueAvg')
            or _ci_get(row, f'{mk}LeagueAverage')
            or _ci_get(row, f'{base}LeagueAverage')
        )
        return (val, pct, _coerce_avg(avg_raw))

    # Nested dict-of-metrics nodes sometimes contain avg-like fields.
    for v in payload.values():
        if isinstance(v, dict):
            node = _ci_get(v, metric_key)
            if isinstance(node, dict):
                avg_raw = (
                    _ci_get(node, 'leagueAvg')
                    or _ci_get(node, 'leagueAverage')
                    or _ci_get(node, 'nhlAvg')
                    or _ci_get(node, 'nhlAverage')
                )
                return (val, pct, _coerce_avg(avg_raw))

    return (val, pct, None)


@main_bp.route('/api/skaters/card/defs')
def api_skaters_card_defs():
    """Return the available Card categories/metrics (from app/static/card_metrics.csv)."""
    defs = _load_card_metrics_defs('skaters')
    j = jsonify(defs)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/goalies/card/defs')
def api_goalies_card_defs():
    """Return the available Goalie Card categories/metrics (from app/static/card_metrics.csv)."""
    defs = _load_card_metrics_defs('goalies')
    j = jsonify(defs)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/teams/card/defs')
def api_teams_card_defs():
    """Return the available Team Card categories/metrics (from app/static/card_metrics.csv)."""
    defs = _load_card_metrics_defs('teams')
    j = jsonify(defs)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


def _parse_locale_float(v: Any) -> Optional[float]:
    """Parse numbers that may use either decimal comma or decimal dot.

    Handles e.g. '1.234,56' (DK) and '1,234.56' (US).
    Returns None if not parseable.
    """
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if not s:
            return None
        s = s.replace('\u00a0', ' ').replace(' ', '')
        last_dot = s.rfind('.')
        last_comma = s.rfind(',')
        if last_dot != -1 and last_comma != -1:
            if last_comma > last_dot:
                # DK style: 1.234,56
                s = s.replace('.', '').replace(',', '.')
            else:
                # US style: 1,234.56
                s = s.replace(',', '')
        elif last_comma != -1 and last_dot == -1:
            s = s.replace(',', '.')
        return float(s)
    except Exception:
        return None


def _safe_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return v
        s = str(v).strip()
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None


def _extract_name(obj: Any) -> str:
    try:
        if not isinstance(obj, dict):
            return ''
        fn = obj.get('firstName')
        ln = obj.get('lastName')
        if fn is not None and ln is not None:
            if isinstance(fn, dict):
                fn = fn.get('default') or (next(iter(fn.values())) if fn else '')
            if isinstance(ln, dict):
                ln = ln.get('default') or (next(iter(ln.values())) if ln else '')
            name = f"{fn or ''} {ln or ''}".strip()
            if name:
                return name
        for k in ('fullName', 'name'):
            val = obj.get(k)
            if isinstance(val, str) and val.strip():
                return val.strip()
        person = obj.get('person')
        if isinstance(person, dict):
            val = person.get('fullName')
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ''
    except Exception:
        return ''


def _extract_pos(obj: Any) -> str:
    try:
        if not isinstance(obj, dict):
            return ''
        for k in ('positionCode', 'pos', 'position', 'primaryPosition'):
            val = obj.get(k)
            if isinstance(val, dict):
                val = val.get('abbrev') or val.get('type')
            if isinstance(val, str) and val.strip():
                s = val.strip().upper()
                if s.startswith('G'):
                    return 'G'
                if s.startswith('D'):
                    return 'D'
                return 'F'
        return ''
    except Exception:
        return ''


def _load_skater_bios_season_cached(season_id: int) -> Dict[int, Dict[str, str]]:
    """Build a playerId->info map from NHL stats skater bios for a season.

    Uses: https://api.nhle.com/stats/rest/en/skater/bios
    This provides currentTeamAbbrev and positionCode for skaters.
    """
    global _SKATER_BIOS_CACHE
    try:
        ttl_s = max(60, int(os.getenv('SKATER_BIOS_CACHE_TTL_SECONDS', '21600') or '21600'))
    except Exception:
        ttl_s = 21600

    try:
        max_items = max(1, int(os.getenv('SKATER_BIOS_CACHE_MAX_ITEMS', '4') or '4'))
    except Exception:
        max_items = 4

    try:
        season_i = int(season_id)
    except Exception:
        season_i = 0
    if season_i <= 0:
        try:
            season_i = int(current_season_id())
        except Exception:
            season_i = 0

    now = time.time()
    _cache_prune_ttl_and_size(_SKATER_BIOS_CACHE, ttl_s=ttl_s, max_items=max_items)
    cached = _SKATER_BIOS_CACHE.get(season_i)
    if cached and (now - cached[0]) < ttl_s:
        return cached[1]

    out: Dict[int, Dict[str, str]] = {}
    # NOTE: The endpoint returns 500 if cayenneExp is omitted.
    url = f'https://api.nhle.com/stats/rest/en/skater/bios?limit=-1&start=0&cayenneExp=seasonId={season_i}'
    try:
        r = requests.get(url, timeout=20, allow_redirects=True)
        if r.status_code == 200:
            data = r.json() if r.content else {}
            rows = data.get('data') if isinstance(data, dict) else None
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    pid = _safe_int(row.get('playerId'))
                    if not pid or pid <= 0:
                        continue
                    team = str(row.get('currentTeamAbbrev') or '').strip().upper()
                    name = str(row.get('skaterFullName') or '').strip()
                    pos_raw = str(row.get('positionCode') or '').strip().upper()
                    pos = 'D' if pos_raw.startswith('D') else 'F'
                    if pid not in out or (name and not (out.get(pid) or {}).get('name')):
                        out[pid] = {
                            'playerId': str(pid),
                            'name': name,
                            'team': team,
                            'position': pos,
                            'positionCode': pos_raw,
                            'birthDate': str(row.get('birthDate') or '').strip(),
                        }
    except Exception:
        out = {}

    _cache_set_multi_bounded(_SKATER_BIOS_CACHE, season_i, out, ttl_s=ttl_s, max_items=max_items)
    return out


def _load_goalie_bios_season_cached(season_id: int) -> Dict[int, Dict[str, str]]:
    """Build a playerId->info map from NHL stats goalie bios for a season.

    Uses: https://api.nhle.com/stats/rest/en/goalie/bios
    This provides currentTeamAbbrev and positionCode for goalies.
    """
    global _SKATER_BIOS_CACHE
    try:
        ttl_s = max(60, int(os.getenv('SKATER_BIOS_CACHE_TTL_SECONDS', '21600') or '21600'))
    except Exception:
        ttl_s = 21600

    try:
        max_items = max(1, int(os.getenv('SKATER_BIOS_CACHE_MAX_ITEMS', '4') or '4'))
    except Exception:
        max_items = 4

    try:
        season_i = int(season_id)
    except Exception:
        season_i = 0
    if season_i <= 0:
        try:
            season_i = int(current_season_id())
        except Exception:
            season_i = 0

    cache_key = -int(season_i or 0)
    now = time.time()
    _cache_prune_ttl_and_size(_SKATER_BIOS_CACHE, ttl_s=ttl_s, max_items=max_items)
    cached = _SKATER_BIOS_CACHE.get(cache_key)
    if cached and (now - cached[0]) < ttl_s:
        return cached[1]

    out: Dict[int, Dict[str, str]] = {}
    url = f'https://api.nhle.com/stats/rest/en/goalie/bios?limit=-1&start=0&cayenneExp=seasonId={season_i}'
    try:
        r = requests.get(url, timeout=20, allow_redirects=True)
        if r.status_code == 200:
            data = r.json() if r.content else {}
            rows = data.get('data') if isinstance(data, dict) else None
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    pid = _safe_int(row.get('playerId'))
                    if not pid or pid <= 0:
                        continue
                    team = str(row.get('currentTeamAbbrev') or '').strip().upper()
                    name = str(row.get('goalieFullName') or row.get('playerFullName') or row.get('skaterFullName') or '').strip()
                    pos_raw = str(row.get('positionCode') or 'G').strip().upper()
                    out[pid] = {
                        'playerId': str(pid),
                        'name': name,
                        'team': team,
                        'position': 'G',
                        'positionCode': pos_raw,
                        'birthDate': str(row.get('birthDate') or '').strip(),
                    }
    except Exception:
        out = {}

    _cache_set_multi_bounded(_SKATER_BIOS_CACHE, cache_key, out, ttl_s=ttl_s, max_items=max_items)
    return out


def _load_all_rosters_cached() -> Dict[int, Dict[str, str]]:
    """Build a playerId->info map by fetching current rosters for all teams.

    Cached with TTL to avoid hammering upstream.
    """
    global _ALL_ROSTERS_CACHE
    try:
        ttl_s = max(60, int(os.getenv('ALL_ROSTERS_CACHE_TTL_SECONDS', '21600') or '21600'))
    except Exception:
        ttl_s = 21600
    now = time.time()
    if _ALL_ROSTERS_CACHE and (now - _ALL_ROSTERS_CACHE[0]) < ttl_s:
        return _ALL_ROSTERS_CACHE[1]

    # Primary source: NHL stats skater bios for the current season.
    # (We intentionally do NOT use app/static/lineups_all.json here; it may be stale.)
    try:
        season_i = int(current_season_id())
    except Exception:
        season_i = 0
    out = _load_skater_bios_season_cached(season_i)

    # Merge goalie bios (current season)
    try:
        g = _load_goalie_bios_season_cached(season_i) or {}
        if g:
            out = {**out, **g}
    except Exception:
        pass

    _ALL_ROSTERS_CACHE = (now, out)
    return out


# Historical roster cache: {seasonId -> (timestamp, playerId->info)}
_ALL_ROSTERS_BY_SEASON_CACHE: Dict[int, Tuple[float, Dict[int, Dict[str, str]]]] = {}


def _load_all_rosters_for_season_cached(season_id: int) -> Dict[int, Dict[str, str]]:
    """Build a playerId->info map for a specific season.

    For the current season, prefer NHL stats skater bios.
    For other seasons, best-effort fetch team rosters from api-web.nhle.com to recover
    a season-appropriate team abbreviation.
    """
    global _ALL_ROSTERS_BY_SEASON_CACHE
    try:
        ttl_s = max(60, int(os.getenv('ALL_ROSTERS_BY_SEASON_CACHE_TTL_SECONDS', '21600') or '21600'))
    except Exception:
        ttl_s = 21600

    try:
        max_items = max(1, int(os.getenv('ALL_ROSTERS_BY_SEASON_CACHE_MAX_ITEMS', '6') or '6'))
    except Exception:
        max_items = 6

    try:
        season_i = int(season_id)
    except Exception:
        season_i = 0
    if season_i <= 0:
        try:
            season_i = int(current_season_id())
        except Exception:
            season_i = 0

    try:
        current_i = int(current_season_id())
    except Exception:
        current_i = 0

    now = time.time()
    _cache_prune_ttl_and_size(_ALL_ROSTERS_BY_SEASON_CACHE, ttl_s=ttl_s, max_items=max_items)
    cached = _ALL_ROSTERS_BY_SEASON_CACHE.get(season_i)
    if cached and (now - cached[0]) < ttl_s:
        return cached[1]

    # Current season: bios has fast, complete coverage.
    if current_i and season_i == current_i:
        out = _load_skater_bios_season_cached(season_i)
        try:
            g = _load_goalie_bios_season_cached(season_i) or {}
            if g:
                out = {**out, **g}
        except Exception:
            pass
        _cache_set_multi_bounded(_ALL_ROSTERS_BY_SEASON_CACHE, season_i, out, ttl_s=ttl_s, max_items=max_items)
        return out

    out: Dict[int, Dict[str, str]] = {}
    # Historical season: build mapping from per-team roster endpoints.
    teams = [str(r.get('Team') or '').strip().upper() for r in (TEAM_ROWS or []) if isinstance(r, dict)]
    teams = [t for t in teams if t]
    for team in teams:
        url = f'https://api-web.nhle.com/v1/roster/{team}/{season_i}'
        try:
            r = requests.get(url, timeout=20, allow_redirects=True)
            if r.status_code != 200:
                continue
            data = r.json() if r.content else {}
            if not isinstance(data, dict):
                continue
            forwards = data.get('forwards') or []
            defensemen = data.get('defensemen') or []
            goalies = data.get('goalies') or []
            for p in list(forwards) + list(defensemen) + list(goalies):
                if not isinstance(p, dict):
                    continue
                pid = _safe_int(p.get('id') or p.get('playerId'))
                if not pid or pid <= 0:
                    continue
                fn = (p.get('firstName') or {}).get('default') if isinstance(p.get('firstName'), dict) else (p.get('firstName') or '')
                ln = (p.get('lastName') or {}).get('default') if isinstance(p.get('lastName'), dict) else (p.get('lastName') or '')
                name = (str(fn).strip() + ' ' + str(ln).strip()).strip() or str(pid)
                pos_raw = str(p.get('positionCode') or p.get('position') or '').strip().upper()
                pos = 'G' if pos_raw.startswith('G') else ('D' if pos_raw.startswith('D') else 'F')
                out[int(pid)] = {
                    'playerId': str(int(pid)),
                    'name': name,
                    'team': team,
                    'position': pos,
                    'positionCode': pos_raw,
                }
        except Exception:
            continue

    # Fill gaps with bios names (team may be currentTeamAbbrev; keep historical roster team when present).
    try:
        bios = _load_skater_bios_season_cached(season_i) or {}
        for pid_s, info in bios.items():
            try:
                pid_i = int(pid_s)
            except Exception:
                continue
            if pid_i in out:
                if (not out[pid_i].get('name')) and info.get('name'):
                    out[pid_i]['name'] = str(info.get('name') or '')
                continue
            try:
                info_d = info if isinstance(info, dict) else {}
                name_s = str(info_d.get('name') or '').strip() or str(pid_i)
                team_s = str(info_d.get('team') or '').strip().upper()
                pos_raw = str(info_d.get('positionCode') or info_d.get('position') or '').strip().upper()
                pos = 'G' if pos_raw.startswith('G') else ('D' if pos_raw.startswith('D') else ('F' if pos_raw else ''))
                out[pid_i] = {
                    'playerId': str(pid_i),
                    'name': name_s,
                    'team': team_s,
                    'position': pos,
                    'positionCode': pos_raw,
                }
            except Exception:
                out[pid_i] = {
                    'playerId': str(pid_i),
                    'name': str(pid_i),
                    'team': '',
                    'position': '',
                    'positionCode': '',
                }
    except Exception:
        pass

    _cache_set_multi_bounded(_ALL_ROSTERS_BY_SEASON_CACHE, season_i, out, ttl_s=ttl_s, max_items=max_items)
    return out


def _parse_proj_row(row: Dict[str, Any]) -> Dict[str, Any]:
    # player_projections.csv columns: PlayerID,Position,Game_No,Age,Rookie,EVO,EVD,PP,SH,GSAx
    pid = _safe_int(row.get('PlayerID') or row.get('playerId') or row.get('player_id') or row.get('id'))
    pos = str(row.get('Position') or row.get('position') or '').strip().upper()[:1]
    gp = _safe_int(row.get('Game_No') or row.get('GP') or row.get('games') or row.get('gamesPlayed'))
    age = _parse_locale_float(row.get('Age'))
    rookie = _parse_locale_float(row.get('Rookie'))
    evo = _parse_locale_float(row.get('EVO'))
    evd = _parse_locale_float(row.get('EVD'))
    pp = _parse_locale_float(row.get('PP'))
    sh = _parse_locale_float(row.get('SH'))
    gsax = _parse_locale_float(row.get('GSAx') or row.get('gsax') or row.get('Gsax') or row.get('GsaX'))
    # Total excludes GSAx per spec
    total = sum([(age or 0.0), (rookie or 0.0), (evo or 0.0), (evd or 0.0), (pp or 0.0), (sh or 0.0)])
    return {
        'playerId': pid,
        'position': pos,
        'gp': gp,
        'Age': age,
        'Rookie': rookie,
        'EVO': evo,
        'EVD': evd,
        'PP': pp,
        'SH': sh,
        'GSAx': gsax,
        'total': total,
    }


@main_bp.route('/api/player-projections/league')
def api_player_projections_league():
    """League-wide player projections.

    Query params:
      team=EDM (optional)
      include_goalies=1 (optional)
    """
    team = str(request.args.get('team') or '').strip().upper()
    include_goalies = str(request.args.get('include_goalies') or '').strip().lower() in ('1', 'true', 'yes', 'y')

    proj_map = _load_player_projections_cached()
    roster_map = _load_all_rosters_cached()

    out: List[Dict[str, Any]] = []
    for pid, raw in (proj_map or {}).items():
        try:
            row = _parse_proj_row(raw)
            if not row.get('playerId'):
                continue
            pos = str(row.get('position') or '').upper()
            if (not include_goalies) and pos.startswith('G'):
                continue  # skaters only by default
            info = roster_map.get(int(row['playerId'])) or {}
            t = (info.get('team') or '').upper()
            if team and t and t != team:
                continue
            if team and not t:
                # If filtering by team and we don't know the team, skip.
                continue
            out.append({
                **row,
                'name': info.get('name') or '',
                'team': t,
                # prefer CSV position for skaters, but ensure a fallback
                'position': pos if pos in ('F', 'D', 'G') else (info.get('position') or 'F'),
            })
        except Exception:
            continue

    out.sort(key=lambda r: float(r.get('total') or 0.0), reverse=True)
    j = jsonify({'players': out})
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/skaters/current-projections')
def api_skaters_current_projections():
    """Current centered skater projections for the Skaters projections tab.

    Returns current-season skaters only, enriched with team identity and 5v5 context.
    """
    team = str(request.args.get('team') or '').strip().upper()
    try:
        season_i = int(str(request.args.get('season') or '').strip() or current_season_id())
    except Exception:
        try:
            season_i = int(current_season_id())
        except Exception:
            season_i = 0

    proj_map = _load_current_player_projections_cached() or {}
    roster_map = _load_all_rosters_for_season_cached(season_i) or _load_all_rosters_cached() or {}
    ctx_rows = _load_context_static_csv() or []

    ctx_by_pid: Dict[int, Dict[str, Optional[float]]] = {}
    for row in ctx_rows:
        try:
            row_season = _safe_int(row.get('Season'))
            if season_i > 0 and row_season and int(row_season) != int(season_i):
                continue
            if str(row.get('StrengthState') or '').strip() != '5v5':
                continue
            pid_i = _safe_int(row.get('PlayerID') or row.get('player_id') or row.get('playerId'))
            if not pid_i or pid_i <= 0:
                continue
            ctx_by_pid[int(pid_i)] = {
                'QoT': _parse_locale_float(row.get('QoT_blend_xG67_G33')),
                'QoC': _parse_locale_float(row.get('QoC_blend_xG67_G33')),
                'ZS': _parse_locale_float(row.get('ZS_Difficulty')),
            }
        except Exception:
            continue

    players: List[Dict[str, Any]] = []
    for pid, raw in proj_map.items():
        try:
            pid_i = _safe_int(raw.get('player_id') or raw.get('playerId') or pid)
            if not pid_i or pid_i <= 0:
                continue

            raw_pos = str(raw.get('position') or '').strip().upper()
            pos = raw_pos
            roster_info = roster_map.get(int(pid_i)) or {}
            if roster_info.get('position'):
                pos = str(roster_info.get('position') or '').strip().upper() or pos
            if pos.startswith('G'):
                continue

            team_abbrev = str(roster_info.get('team') or '').strip().upper()
            if team and team_abbrev != team:
                continue

            projected_value = _parse_locale_float(raw.get('projected_value'))
            gp = _safe_int(raw.get('games_in_window') or raw.get('window_games') or raw.get('gp')) or 0

            players.append({
                'playerId': int(pid_i),
                'name': str(raw.get('player') or roster_info.get('name') or '').strip(),
                'team': team_abbrev,
                'position': pos,
                'gp': int(gp),
                'projectedValue': float(projected_value) if projected_value is not None else None,
                'projection': float(projected_value) if projected_value is not None else None,
                'contextData': ctx_by_pid.get(int(pid_i), {'QoT': None, 'QoC': None, 'ZS': None}),
            })
        except Exception:
            continue

    players.sort(
        key=lambda row: (
            float(row.get('projectedValue')) if row.get('projectedValue') is not None else float('-inf'),
            str(row.get('name') or ''),
        ),
        reverse=True,
    )

    j = jsonify({'players': players, 'season': season_i})
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/skaters/player-projection-trend/<int:player_id>')
def api_skaters_player_projection_trend(player_id: int):
    """Return a player's season projection trend for the Skaters projections tab."""
    pid = int(player_id)
    if pid <= 0:
        return jsonify({'error': 'invalid_player_id'}), 400

    try:
        season_i = int(str(request.args.get('season') or '').strip() or current_season_id())
    except Exception:
        try:
            season_i = int(current_season_id())
        except Exception:
            season_i = 0

    if season_i <= 0:
        return jsonify({'error': 'invalid_season'}), 400

    sb_rows = _sb_read(
        'player_game_projections',
        columns='season,game_id,source_game_id,game_date,player_id,source_player_id,team,opponent,position,projected_value,poss_value,off_the_puck,gax,games_in_window,model_key',
        filters={
            'season': f'eq.{season_i}',
            'player_id': f'eq.{pid}',
            'model_key': 'eq.preseason_updating',
        },
    ) or []

    if not sb_rows:
        sb_rows = _load_player_game_projection_export_rows_cached(season_i, pid)

    if not sb_rows:
        return jsonify({'error': 'not_found'}), 404

    source_player_id = _safe_int((sb_rows[0] or {}).get('source_player_id')) if sb_rows else None
    performance_by_game = _load_player_game_performance_rows_cached(season_i, source_player_id) if source_player_id else {}

    def _sort_key(row: Dict[str, Any]) -> Tuple[str, int, int]:
        game_date = str(row.get('game_date') or '').strip()
        source_game_id = _safe_int(row.get('source_game_id')) or 0
        game_id = _safe_int(row.get('game_id')) or 0
        return (game_date, source_game_id, game_id)

    rows_sorted = sorted(sb_rows, key=_sort_key)
    points: List[Dict[str, Any]] = []
    player_name = ''
    position = ''
    team_abbrev = ''
    for idx, row in enumerate(rows_sorted, start=1):
        projection = _parse_locale_float(row.get('projected_value'))
        poss_value = _parse_locale_float(row.get('poss_value'))
        off_the_puck = _parse_locale_float(row.get('off_the_puck'))
        gax = _parse_locale_float(row.get('gax'))
        source_game_id = _safe_int(row.get('source_game_id'))
        performance = performance_by_game.get(int(source_game_id)) if source_game_id else None

        if not player_name:
            player_name = str(row.get('player') or '').strip()
        if not position:
            position = str(row.get('position') or '').strip().upper()
        if not team_abbrev:
            team_abbrev = str(row.get('team') or '').strip().upper()

        points.append({
            'gameNumber': idx,
            'gameDate': str(row.get('game_date') or '').strip(),
            'gameId': _safe_int(row.get('game_id')),
            'sourceGameId': source_game_id,
            'opponent': str(row.get('opponent') or '').strip().upper(),
            'projection': float(projection) if projection is not None else None,
            'performance': float(performance) if performance is not None else None,
            'metrics': {
                'possValue': float(poss_value) if poss_value is not None else None,
                'offThePuck': float(off_the_puck) if off_the_puck is not None else None,
                'gax': float(gax) if gax is not None else None,
            },
            'gamesInWindow': _safe_int(row.get('games_in_window')),
        })

    j = jsonify({
        'playerId': pid,
        'season': season_i,
        'name': player_name,
        'position': position,
        'team': team_abbrev,
        'points': points,
    })
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


def _age_from_birthdate(birth_date: Any, today: Optional[datetime] = None) -> Optional[float]:
    try:
        raw = str(birth_date or '').strip()
        if not raw:
            return None
        born = datetime.strptime(raw[:10], '%Y-%m-%d').date()
        ref = (today or datetime.utcnow()).date()
        days = (ref - born).days
        if days <= 0:
            return None
        return float(days) / 365.2425
    except Exception:
        return None


@main_bp.route('/api/teams/current-projections')
def api_teams_current_projections():
    """Current centered team/player projections for the Teams projections tab."""
    try:
        season_i = int(str(request.args.get('season') or '').strip() or current_season_id())
    except Exception:
        try:
            season_i = int(current_season_id())
        except Exception:
            season_i = 0

    proj_map = _load_current_player_projections_cached() or {}
    skater_bios = _load_skater_bios_season_cached(season_i) or {}
    goalie_bios = _load_goalie_bios_season_cached(season_i) or {}
    roster_map = {**skater_bios, **goalie_bios}
    lineups_all = _load_lineups_all() or {}

    lineup_pids_by_team: Dict[str, set[int]] = {}
    for team_abbrev, node in lineups_all.items():
        try:
            team_key = str(team_abbrev or '').strip().upper()
            if not team_key:
                continue
            ids: set[int] = set()
            for bucket in ('forwards', 'defense', 'goalies'):
                for player in (node.get(bucket) or []):
                    pid_i = _safe_int((player or {}).get('playerId'))
                    if pid_i and pid_i > 0:
                        ids.add(int(pid_i))
            lineup_pids_by_team[team_key] = ids
        except Exception:
            continue

    players: List[Dict[str, Any]] = []
    for pid, raw in proj_map.items():
        try:
            pid_i = _safe_int(raw.get('player_id') or raw.get('playerId') or pid)
            if not pid_i or pid_i <= 0:
                continue
            info = roster_map.get(int(pid_i)) or {}
            team_abbrev = str(info.get('team') or '').strip().upper()
            if not team_abbrev:
                continue
            pos = str(raw.get('position') or info.get('position') or '').strip().upper()
            if pos.startswith(('L', 'R', 'C')):
                pos = 'F'
            elif pos.startswith('D'):
                pos = 'D'
            elif pos.startswith('G'):
                pos = 'G'
            else:
                pos = str(info.get('position') or 'F').strip().upper() or 'F'
            projection = _parse_locale_float(raw.get('projected_value'))
            age_years = _age_from_birthdate(info.get('birthDate'))
            players.append({
                'playerId': int(pid_i),
                'name': str(info.get('name') or raw.get('player') or '').strip(),
                'team': team_abbrev,
                'position': pos,
                'projection': float(projection) if projection is not None else None,
                'age': float(age_years) if age_years is not None else None,
                'inCurrentLineup': int(pid_i) in lineup_pids_by_team.get(team_abbrev, set()),
            })
        except Exception:
            continue

    players.sort(key=lambda row: (str(row.get('team') or ''), str(row.get('position') or ''), -(float(row.get('projection')) if row.get('projection') is not None else -9999.0), str(row.get('name') or '')))

    j = jsonify({'players': players, 'season': season_i})
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


def _load_player_projections_cached() -> Dict[int, Dict[str, Any]]:
    """Load player projections from Google Sheets (with CSV fallback) into memory (TTL cached)."""
    return _load_player_projections_from_sheets()

_PLAYER_PROJECTIONS_SHEETS_CACHE: Optional[Tuple[float, Dict[int, Dict[str, Any]]]] = None

def _load_current_player_projections_cached() -> Dict[int, Dict[str, Any]]:
    """Load current-season game projection inputs from player_current_projections.

    Falls back to the legacy player projections source if the current table is unavailable.
    """
    global _CURRENT_PLAYER_PROJECTIONS_CACHE
    try:
        ttl_s = max(30, int(os.getenv('PLAYER_PROJECTIONS_CACHE_TTL_SECONDS', '300') or '300'))
    except Exception:
        ttl_s = 300
    now = time.time()
    if _CURRENT_PLAYER_PROJECTIONS_CACHE and (now - _CURRENT_PLAYER_PROJECTIONS_CACHE[0]) < ttl_s:
        return _CURRENT_PLAYER_PROJECTIONS_CACHE[1]

    try:
        season_i = int(current_season_id())
    except Exception:
        season_i = 0

    sb_rows = _sb_read(
        'player_current_projections',
        columns='season,player_id,position,raw_projected_value,projected_value,games_in_window,rookie_factor,source_player_id,source_game_id,model_key',
        filters={
            'season': f'eq.{season_i}',
            'model_key': 'eq.preseason_updating',
        },
    ) if season_i > 0 else None

    if sb_rows is not None and len(sb_rows) > 0:
        out: Dict[int, Dict[str, Any]] = {}
        for r in sb_rows:
            try:
                pid_raw = r.get('player_id') or r.get('playerId')
                pid = _safe_int(pid_raw)
                if not pid or pid <= 0:
                    continue
                out[pid] = r
            except Exception:
                continue
        _CURRENT_PLAYER_PROJECTIONS_CACHE = (now, out)
        return out

    data = _load_player_projections_cached()
    _CURRENT_PLAYER_PROJECTIONS_CACHE = (now, data)
    return data


def _load_player_game_projection_export_rows_cached(season_i: int, player_id: int) -> List[Dict[str, Any]]:
    """Load per-player season trend rows from the local export-validation cache."""
    try:
        ttl_s = max(30, int(os.getenv('PLAYER_PROJECTIONS_CACHE_TTL_SECONDS', '300') or '300'))
    except Exception:
        ttl_s = 300

    now = time.time()
    cached = _PLAYER_GAME_PROJECTION_EXPORT_CACHE.get(int(season_i))
    if cached and (now - cached[0]) < ttl_s:
        return list(cached[1].get(int(player_id), []))

    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        export_path = os.path.join(project_root, 'data', 'game_projection', 'export_validation', f'{int(season_i)}_export.pkl')
        if not os.path.exists(export_path):
            return []
        export_df = pd.read_pickle(export_path)
    except Exception:
        return []

    try:
        if 'model_key' in export_df.columns:
            export_df = export_df[export_df['model_key'].astype(str) == 'preseason_updating'].copy()
        if 'player_id' not in export_df.columns:
            return []
        export_df['player_id'] = pd.to_numeric(export_df['player_id'], errors='coerce').astype('Int64')
        export_df = export_df[export_df['player_id'].notna()].copy()
        if export_df.empty:
            return []
        export_df['player_id'] = export_df['player_id'].astype(int)
    except Exception:
        return []

    player_map: Dict[int, List[Dict[str, Any]]] = {}
    for row in export_df.to_dict(orient='records'):
        try:
            pid_i = _safe_int(row.get('player_id'))
            if not pid_i or pid_i <= 0:
                continue
            player_map.setdefault(int(pid_i), []).append(row)
        except Exception:
            continue

    _PLAYER_GAME_PROJECTION_EXPORT_CACHE[int(season_i)] = (now, player_map)
    return list(player_map.get(int(player_id), []))


def _projection_position_group(raw_position: Any) -> str:
    pos = str(raw_position or '').strip().upper()
    if pos.startswith(('L', 'R', 'C', 'F')):
        return 'F'
    if pos.startswith('D'):
        return 'D'
    if pos.startswith('G'):
        return 'G'
    return pos or 'F'


def _load_player_game_performance_rows_cached(season_i: int, player_id: int) -> Dict[int, float]:
    """Load cumulative in-season performance values from cached game-projection CSVs."""
    try:
        ttl_s = max(30, int(os.getenv('PLAYER_PROJECTIONS_CACHE_TTL_SECONDS', '300') or '300'))
    except Exception:
        ttl_s = 300

    now = time.time()
    cached = _PLAYER_GAME_PERFORMANCE_CACHE.get(int(season_i))
    if cached and (now - cached[0]) < ttl_s:
        return dict(cached[1].get(int(player_id), {}))

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data', 'game_projection')
    games_path = os.path.join(data_dir, 'games.csv')
    pvm_path = os.path.join(data_dir, 'pvm.csv')
    skaters_path = os.path.join(data_dir, 'skaters.csv')
    goalies_path = os.path.join(data_dir, 'goalies.csv')
    if not all(os.path.exists(path) for path in (games_path, pvm_path, skaters_path, goalies_path)):
        return {}

    try:
        season_key = str(int(season_i))
        games = pd.read_csv(games_path, parse_dates=['date'], dtype={'season': str})
        pvm = pd.read_csv(pvm_path, dtype={'season': str})
        skaters = pd.read_csv(skaters_path, dtype={'season': str})
        goalies = pd.read_csv(goalies_path, dtype={'season': str})
    except Exception:
        return {}

    games = games[games['season'].astype(str) == season_key].copy()
    pvm = pvm[pvm['season'].astype(str) == season_key].copy()
    skaters = skaters[skaters['season'].astype(str) == season_key].copy()
    goalies = goalies[goalies['season'].astype(str) == season_key].copy()
    if games.empty or pvm.empty:
        return {}

    pvm['poss_value'] = (
        pvm['faceoffs'].fillna(0.0)
        + pvm['defensive'].fillna(0.0)
        + pvm['passes'].fillna(0.0)
        + pvm['carries'].fillna(0.0)
        + pvm['dump_ins_outs'].fillna(0.0)
    )
    pvm['position_group'] = pvm['position'].apply(_projection_position_group)
    pvm_pg = (
        pvm.groupby(['season', 'playerid', 'gameid', 'position_group'], as_index=False)
        .agg(
            poss_value=('poss_value', 'sum'),
            off_the_puck=('off_the_puck', 'sum'),
        )
        .rename(columns={'position_group': 'position'})
    )

    skater_pg = (
        skaters.groupby(['season', 'playerid', 'gameid'], as_index=False)
        .agg(ig=('ig', 'sum'), ixg=('ixg', 'sum'))
    )
    skater_pg['gax'] = skater_pg['ig'].fillna(0.0) - skater_pg['ixg'].fillna(0.0)

    goalie_pg = (
        goalies.groupby(['season', 'playerid', 'gameid'], as_index=False)
        .agg(xg_on_a=('xg_on_a', 'sum'), ga=('ga', 'sum'))
    )
    goalie_pg['goalie_gsax'] = goalie_pg['xg_on_a'].fillna(0.0) - goalie_pg['ga'].fillna(0.0)

    game_dates = games[['game_id', 'date']].rename(columns={'game_id': 'gameid'})
    pg = (
        pvm_pg
        .merge(skater_pg[['season', 'playerid', 'gameid', 'gax']], on=['season', 'playerid', 'gameid'], how='left')
        .merge(goalie_pg[['season', 'playerid', 'gameid', 'goalie_gsax']], on=['season', 'playerid', 'gameid'], how='left')
        .merge(game_dates, on='gameid', how='inner')
    )
    if pg.empty:
        return {}

    pg['gax'] = pg['gax'].fillna(0.0)
    for metric_name in _SKATER_PROJECTION_COEFS:
        pg[metric_name] = pd.to_numeric(pg[metric_name], errors='coerce').fillna(0.0)
        group_mean = pg.groupby(['season', 'position'])[metric_name].transform('mean')
        pg[metric_name] = pg[metric_name] - group_mean.fillna(0.0)

    pg['weighted_value'] = 0.0
    for metric_name, coef in _SKATER_PROJECTION_COEFS.items():
        pg['weighted_value'] += pg[metric_name] * float(coef)

    pg = pg.sort_values(['playerid', 'date', 'gameid']).copy()
    pg['game_number'] = pg.groupby(['season', 'playerid']).cumcount() + 1
    pg['performance'] = pg.groupby(['season', 'playerid'])['weighted_value'].cumsum() / pg['game_number']

    player_map: Dict[int, Dict[int, float]] = {}
    for row in pg[['playerid', 'gameid', 'performance']].itertuples(index=False):
        pid_i = _safe_int(getattr(row, 'playerid', None))
        game_id = _safe_int(getattr(row, 'gameid', None))
        perf = _parse_locale_float(getattr(row, 'performance', None))
        if not pid_i or not game_id or perf is None:
            continue
        player_map.setdefault(int(pid_i), {})[int(game_id)] = float(perf)

    _PLAYER_GAME_PERFORMANCE_CACHE[int(season_i)] = (now, player_map)
    return dict(player_map.get(int(player_id), {}))

def _load_player_projections_from_sheets() -> Dict[int, Dict[str, Any]]:
    """Load player projections. Tries Supabase, falls back to CSV."""
    global _PLAYER_PROJECTIONS_SHEETS_CACHE
    try:
        ttl_s = max(30, int(os.getenv('PLAYER_PROJECTIONS_CACHE_TTL_SECONDS', '300') or '300'))
    except Exception:
        ttl_s = 300
    now = time.time()
    if _PLAYER_PROJECTIONS_SHEETS_CACHE and (now - _PLAYER_PROJECTIONS_SHEETS_CACHE[0]) < ttl_s:
        return _PLAYER_PROJECTIONS_SHEETS_CACHE[1]

    # Try Supabase first
    sb_rows = _sb_read("player_projections", col_map=_COL_MAP_PLAYER_PROJECTIONS)
    if sb_rows is not None:
        out: Dict[int, Dict[str, Any]] = {}
        for r in sb_rows:
            try:
                pid_raw = r.get('PlayerID') or r.get('playerId') or r.get('player_id')
                if pid_raw is None:
                    continue
                pid = _safe_int(pid_raw)
                if not pid or pid <= 0:
                    continue
                out[pid] = r
            except Exception:
                continue
        _PLAYER_PROJECTIONS_SHEETS_CACHE = (now, out)
        return out

    # Fallback to CSV
    data = _load_player_projections_csv()
    _PLAYER_PROJECTIONS_SHEETS_CACHE = (now, data)
    return data

_LINEUPS_ALL_CACHE: Optional[Tuple[float, Dict[str, Any]]] = None

def _load_lineups_all() -> Dict[str, Any]:
    global _LINEUPS_ALL_CACHE
    ttl_s = int(os.getenv('LINEUPS_SHEET_CACHE_TTL_SECONDS', '300') or '300')
    now = time.time()
    if _LINEUPS_ALL_CACHE and (now - _LINEUPS_ALL_CACHE[0]) < max(1, ttl_s):
        return _LINEUPS_ALL_CACHE[1]

    sb_raw = _sb_read("dailyfaceoff_lineups")
    if sb_raw:
        # Map Supabase snake_case → original column names used below
        rows = []
        for r in sb_raw:
            rows.append({
                'Team': r.get('team') or r.get('Team') or '',
                'Unit': r.get('unit') or r.get('line') or r.get('Unit') or '',
                'Pos': r.get('pos') or r.get('position') or r.get('Pos') or '',
                'PlayerName': r.get('player_name') or r.get('player') or r.get('name') or r.get('PlayerName') or '',
                'playerId': r.get('player_id') or r.get('playerId') or r.get('PlayerID') or '',
                'Timestamp': str(r.get('timestamp') or r.get('updated_at') or r.get('created_at') or r.get('Timestamp') or ''),
            })
    else:
        _LINEUPS_ALL_CACHE = (now, {})
        return {}

    out: Dict[str, Any] = {}
    latest_ts_by_team: Dict[str, str] = {}

    def _ensure_team(t: str) -> Dict[str, Any]:
        if t not in out:
            out[t] = {'team': t, 'forwards': [], 'defense': [], 'goalies': [], 'generated_at': None}
        return out[t]

    for r in rows:
        try:
            team = str(r.get('Team') or '').strip().upper()
            if not team:
                continue
            unit = str(r.get('Unit') or '').strip().upper()
            pos = str(r.get('Pos') or '').strip().upper()[:1]
            name = str(r.get('PlayerName') or r.get('Name') or '').strip()
            pid_raw = r.get('playerId') if 'playerId' in r else r.get('PlayerId')
            pid = int(str(pid_raw).strip())
            ts = str(r.get('Timestamp') or '').strip()
        except Exception:
            continue

        rec = {'name': name, 'playerId': pid, 'unit': unit, 'pos': ('G' if unit.startswith('G') else pos)}

        bucket = 'forwards'
        if rec['pos'] == 'G' or unit.startswith('G'):
            bucket = 'goalies'
            rec['pos'] = 'G'
        elif rec['pos'] == 'D' or unit.startswith('LD') or unit.startswith('RD'):
            bucket = 'defense'
            rec['pos'] = 'D'
        else:
            rec['pos'] = 'F'

        tnode = _ensure_team(team)
        tnode[bucket].append(rec)
        if ts:
            latest_ts_by_team[team] = max(latest_ts_by_team.get(team, ''), ts)

    # Set generated_at per team
    for t, node in out.items():
        node['generated_at'] = latest_ts_by_team.get(t)

    _LINEUPS_ALL_CACHE = (now, out)
    return out

_ODDS_SNAPSHOT_ROWS_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}

def _load_game_team_abbrevs(game_id: int) -> Tuple[str, str]:
    try:
        ttl = int(os.getenv('BOX_CACHE_TTL_SECONDS', '600') or '600')
    except Exception:
        ttl = 600

    data = _cache_get(_BOX_CACHE, int(game_id), ttl) or {}
    if not data:
        try:
            resp = requests.get(f'https://api-web.nhle.com/v1/gamecenter/{int(game_id)}/boxscore', timeout=20)
            if resp.status_code == 200:
                data = resp.json() or {}
                try:
                    max_items = max(1, int(os.getenv('BOX_CACHE_MAX_ITEMS', '64') or '64'))
                except Exception:
                    max_items = 64
                try:
                    _cache_set_multi_bounded(_BOX_CACHE, int(game_id), data, ttl_s=ttl, max_items=max_items)
                except Exception:
                    pass
        except Exception:
            data = {}

    away = str(((data.get('awayTeam') or {}).get('abbrev') or (data.get('awayTeam') or {}).get('abbreviation') or '')).strip().upper()
    home = str(((data.get('homeTeam') or {}).get('abbrev') or (data.get('homeTeam') or {}).get('abbreviation') or '')).strip().upper()
    return away, home

def _odds_snapshot_cache_key(game_ids: Optional[Iterable[int]] = None, game_id: Optional[int] = None) -> str:
    if game_id is not None:
        return f'game:{int(game_id)}'
    if game_ids is None:
        return 'all'
    vals = sorted({int(gid) for gid in game_ids if gid})
    return 'games:' + ','.join(str(v) for v in vals)

def _snapshot_pick(row: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row.get(key) not in (None, ''):
            return row.get(key)
    return None

def _snapshot_percent(raw: Any) -> Optional[float]:
    val = _parse_locale_float(raw)
    if val is None:
        return None
    pct = float(val)
    if 0.0 <= pct <= 1.0:
        pct *= 100.0
    return pct

def _snapshot_ml(raw: Any) -> Optional[float]:
    val = _parse_locale_float(raw)
    if val is None:
        return None
    return float(val)

def _snapshot_timestamp(row: Dict[str, Any]) -> str:
    raw = _snapshot_pick(row, 'timestamp', 'timestamp_utc', 'fetched_at_utc', 'snapshot_at', 'created_at', 'updated_at', 'TimestampUTC')
    return str(raw or '').strip()

def _snapshot_game_id(row: Dict[str, Any]) -> Optional[int]:
    return _safe_int(_snapshot_pick(row, 'game_id', 'gameId', 'GameID', 'gameid'))

def _snapshot_team(row: Dict[str, Any], side: Optional[str] = None) -> str:
    if side == 'away':
        raw = _snapshot_pick(row, 'away_team', 'away', 'away_abbrev', 'awayTeam', 'Away')
    elif side == 'home':
        raw = _snapshot_pick(row, 'home_team', 'home', 'home_abbrev', 'homeTeam', 'Home')
    else:
        raw = _snapshot_pick(row, 'team', 'team_abbrev', 'teamAbbrev', 'abbrev', 'Team')
    return str(raw or '').strip().upper()

def _snapshot_side_ml(row: Dict[str, Any], side: Optional[str] = None) -> Optional[float]:
    if side == 'away':
        raw = _snapshot_pick(row, 'odds_away', 'away_odds', 'away_ml', 'ml_away', 'oddsAway', 'OddsAway', 'awayPrice')
    elif side == 'home':
        raw = _snapshot_pick(row, 'odds_home', 'home_odds', 'home_ml', 'ml_home', 'oddsHome', 'OddsHome', 'homePrice')
    else:
        raw = _snapshot_pick(row, 'money_line_2_way', 'ml', 'odds', 'american_odds', 'price', 'value')
    return _snapshot_ml(raw)

def _snapshot_side_pct(row: Dict[str, Any], side: str, kind: str) -> Optional[float]:
    if kind == 'win':
        raw = _snapshot_pick(
            row,
            f'{side}_win_pct', f'win_{side}_pct', f'{side}WinPct',
            f'{side}_win_prob', f'win_{side}_prob',
            'winAwayPct' if side == 'away' else 'winHomePct',
            'WinAway' if side == 'away' else 'WinHome'
        )
    else:
        raw = _snapshot_pick(
            row,
            f'{side}_bet_pct', f'bet_{side}_pct', f'{side}BetPct',
            'betAwayPct' if side == 'away' else 'betHomePct',
            'BetAway' if side == 'away' else 'BetHome'
        )
    return _snapshot_percent(raw)

def _load_odds_snapshot_rows(*, game_ids: Optional[Iterable[int]] = None, game_id: Optional[int] = None) -> List[Dict[str, Any]]:
    global _ODDS_SNAPSHOT_ROWS_CACHE
    try:
        ttl_s = max(10, int(os.getenv('ODDS_SNAPSHOTS_CACHE_TTL_SECONDS', '60') or '60'))
    except Exception:
        ttl_s = 60
    cache_key = _odds_snapshot_cache_key(game_ids, game_id)
    now = time.time()
    cached = _ODDS_SNAPSHOT_ROWS_CACHE.get(cache_key)
    if cached and (now - cached[0]) < ttl_s:
        return cached[1]

    filters: Optional[Dict[str, str]] = None
    if game_id is not None:
        filters = {'game_id': f'eq.{int(game_id)}'}
    elif game_ids is not None:
        vals = sorted({int(gid) for gid in game_ids if gid})
        if vals:
            filters = {'game_id': 'in.(' + ','.join(str(v) for v in vals) + ')'}

    rows = _sb_read('odds_snapshots', filters=filters)
    if filters and not rows:
        alt_filters = {('gameid' if k == 'game_id' else k): v for k, v in filters.items()}
        rows = _sb_read('odds_snapshots', filters=alt_filters)
    out = rows or []
    _ODDS_SNAPSHOT_ROWS_CACHE[cache_key] = (now, out)
    return out

def _build_odds_snapshot_payloads(rows: List[Dict[str, Any]], away_abbrev: str = '', home_abbrev: str = '') -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for idx, row in enumerate(rows or []):
        ts = _snapshot_timestamp(row) or f'row-{idx:08d}'
        groups.setdefault(ts, []).append(row)

    latest_payload: Dict[str, Any] = {}
    points_by_team: Dict[str, List[Dict[str, Any]]] = {}

    for ts in sorted(groups.keys()):
        group = groups[ts]
        by_team: Dict[str, Dict[str, Any]] = {}
        payload: Dict[str, Any] = {
            'timestamp': ts,
            'awayTeam': away_abbrev or '',
            'homeTeam': home_abbrev or '',
            'oddsAway': None,
            'oddsHome': None,
            'winAwayPct': None,
            'winHomePct': None,
            'betAwayPct': None,
            'betHomePct': None,
        }
        for row in group:
            away_team = _snapshot_team(row, 'away')
            home_team = _snapshot_team(row, 'home')
            if away_team:
                payload['awayTeam'] = away_team
            if home_team:
                payload['homeTeam'] = home_team

            away_ml = _snapshot_side_ml(row, 'away')
            home_ml = _snapshot_side_ml(row, 'home')
            if away_ml is not None:
                payload['oddsAway'] = away_ml
            if home_ml is not None:
                payload['oddsHome'] = home_ml

            away_win = _snapshot_side_pct(row, 'away', 'win')
            home_win = _snapshot_side_pct(row, 'home', 'win')
            away_bet = _snapshot_side_pct(row, 'away', 'bet')
            home_bet = _snapshot_side_pct(row, 'home', 'bet')
            if away_win is not None:
                payload['winAwayPct'] = away_win
            if home_win is not None:
                payload['winHomePct'] = home_win
            if away_bet is not None:
                payload['betAwayPct'] = away_bet
            if home_bet is not None:
                payload['betHomePct'] = home_bet

            team_row = _snapshot_team(row)
            if team_row:
                team_rec = by_team.setdefault(team_row, {'ml': None, 'winPct': None, 'betPct': None})
                team_ml = _snapshot_side_ml(row)
                if team_ml is not None:
                    team_rec['ml'] = team_ml
                team_win = _snapshot_percent(_snapshot_pick(row, 'win_pct', 'win_prop', 'win_probability', 'implied_win_pct'))
                team_bet = _snapshot_percent(_snapshot_pick(row, 'bet_pct', 'bet_prop', 'kelly_pct'))
                if team_win is not None:
                    team_rec['winPct'] = team_win
                if team_bet is not None:
                    team_rec['betPct'] = team_bet

        payload['awayTeam'] = str(payload.get('awayTeam') or away_abbrev or '').upper()
        payload['homeTeam'] = str(payload.get('homeTeam') or home_abbrev or '').upper()

        away_team = str(payload.get('awayTeam') or '').upper()
        home_team = str(payload.get('homeTeam') or '').upper()
        if payload.get('oddsAway') is None and away_team and isinstance(by_team.get(away_team), dict):
            payload['oddsAway'] = by_team[away_team].get('ml')
            payload['winAwayPct'] = payload.get('winAwayPct') if payload.get('winAwayPct') is not None else by_team[away_team].get('winPct')
            payload['betAwayPct'] = payload.get('betAwayPct') if payload.get('betAwayPct') is not None else by_team[away_team].get('betPct')
        if payload.get('oddsHome') is None and home_team and isinstance(by_team.get(home_team), dict):
            payload['oddsHome'] = by_team[home_team].get('ml')
            payload['winHomePct'] = payload.get('winHomePct') if payload.get('winHomePct') is not None else by_team[home_team].get('winPct')
            payload['betHomePct'] = payload.get('betHomePct') if payload.get('betHomePct') is not None else by_team[home_team].get('betPct')

        added: set[str] = set()
        if away_team and payload.get('oddsAway') is not None:
            away_win_prop = _parse_locale_float(payload.get('winAwayPct'))
            points_by_team.setdefault(away_team, []).append({
                't': ts,
                'ml': payload.get('oddsAway'),
                'winProp': (away_win_prop / 100.0) if away_win_prop is not None else None,
            })
            added.add(away_team)
        if home_team and payload.get('oddsHome') is not None:
            home_win_prop = _parse_locale_float(payload.get('winHomePct'))
            points_by_team.setdefault(home_team, []).append({
                't': ts,
                'ml': payload.get('oddsHome'),
                'winProp': (home_win_prop / 100.0) if home_win_prop is not None else None,
            })
            added.add(home_team)
        for team_key, rec in by_team.items():
            if team_key in added or rec.get('ml') is None:
                continue
            rec_win_pct = _parse_locale_float(rec.get('winPct'))
            points_by_team.setdefault(team_key, []).append({
                't': ts,
                'ml': rec.get('ml'),
                'winProp': (rec_win_pct / 100.0) if rec_win_pct is not None else None,
            })

        payload['_by_team'] = by_team
        latest_payload = payload

    return latest_payload, points_by_team

def _load_latest_odds_snapshot_map(game_team_map: Dict[int, Tuple[str, str]]) -> Dict[int, Dict[str, Any]]:
    if not game_team_map:
        return {}
    rows = _load_odds_snapshot_rows(game_ids=game_team_map.keys())
    by_game: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        gid = _snapshot_game_id(row)
        if gid:
            by_game.setdefault(int(gid), []).append(row)
    out: Dict[int, Dict[str, Any]] = {}
    for gid, teams in game_team_map.items():
        away_abbrev, home_abbrev = teams
        payload, _ = _build_odds_snapshot_payloads(by_game.get(int(gid), []), away_abbrev, home_abbrev)
        if payload:
            out[int(gid)] = payload
    return out

def _load_player_projections_csv() -> Dict[int, Dict[str, Any]]:
    """Load app/static/player_projections.csv into a dict keyed by playerId (int).
    Accepts flexible column casing/names for 'playerId'.
    """
    path = _static_path('player_projections.csv')
    out: Dict[int, Dict[str, Any]] = {}
    try:
        if not os.path.exists(path):
            return {}
        with open(path, 'r', encoding='utf-8', newline='') as f:
            rdr = csv.DictReader(f)
            # Find id column flexibly
            id_col = None
            cols = [c for c in (rdr.fieldnames or [])]
            lowers = { (c or '').lower(): (c or '') for c in cols }
            for cand in ('playerid', 'player_id', 'playerid', 'id'):
                if cand in lowers:
                    id_col = lowers[cand]
                    break
            # If not found, try exact 'playerId'
            if id_col is None and 'playerId' in cols:
                id_col = 'playerId'
            for row in rdr:
                try:
                    pid_raw = row.get(id_col) if id_col else None
                    if pid_raw is None:
                        continue
                    pid = int(str(pid_raw).strip())
                except Exception:
                    continue
                out[pid] = row
    except Exception:
        return {}
    return out

def _load_prestart_snapshots_map() -> Dict[int, Dict[str, Any]]:
    """Load prestart_snapshots.csv into a map keyed by GameID (int) keeping the latest row per game.
    Expected columns: TimestampUTC,DateET,GameID,StartTimeET,Away,Home,WinAway,WinHome,OddsAway,OddsHome,BetAway,BetHome
    """
    path = _prestart_csv_path()
    latest: Dict[int, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return latest
    try:
        with open(path, 'r', encoding='utf-8', newline='') as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                try:
                    gid_raw = row.get('GameID')
                    if gid_raw is None:
                        continue
                    gid = int(str(gid_raw).strip())
                except Exception:
                    continue
                ts = row.get('TimestampUTC') or ''
                # Keep the last seen row per GameID (file is append-only)
                latest[gid] = {
                    'TimestampUTC': ts,
                    'DateET': row.get('DateET'),
                    'StartTimeET': row.get('StartTimeET'),
                    'Away': row.get('Away'),
                    'Home': row.get('Home'),
                    # Store numeric percents as floats
                    'winAwayPct': _safe_float(row.get('WinAway')),
                    'winHomePct': _safe_float(row.get('WinHome')),
                    'oddsAway': row.get('OddsAway'),
                    'oddsHome': row.get('OddsHome'),
                    'betAwayPct': _safe_float(row.get('BetAway')),
                    'betHomePct': _safe_float(row.get('BetHome')),
                }
    except Exception:
        return latest
    return latest


_STARTED_GAME_SHEET_CACHE: Optional[Tuple[float, Dict[int, Dict[str, Any]]]] = None

def _load_started_game_overrides_from_sheet(sheet_id: str, worksheet: str) -> Dict[int, Dict[str, Any]]:
    """Load latest ML overrides for started games from Supabase."""
    global _STARTED_GAME_SHEET_CACHE
    try:
        ttl_s = max(1, int(os.getenv('STARTED_OVERRIDES_SHEET_CACHE_TTL_SECONDS', '30') or '30'))
    except Exception:
        ttl_s = 30
    now = time.time()
    if _STARTED_GAME_SHEET_CACHE and (now - _STARTED_GAME_SHEET_CACHE[0]) < ttl_s:
        return _STARTED_GAME_SHEET_CACHE[1]

    # Supabase
    sb_rows = _sb_read("started_overrides")
    if sb_rows is not None and len(sb_rows) > 0:
        by_game_team: Dict[int, Dict[str, Dict[str, Any]]] = {}
        for r in sb_rows:
            try:
                gid = int(r.get('game_id') or 0)
                if gid <= 0:
                    continue
                team_ab = str(r.get('team') or '').strip().upper()
                ml = r.get('ml')
                if not team_ab and ml is None:
                    continue
                team_map = by_game_team.setdefault(gid, {})
                team_map[team_ab] = {'ml': ml, 'winPct': None}
            except Exception:
                continue
        out: Dict[int, Dict[str, Any]] = {}
        for gid, team_map in by_game_team.items():
            out[gid] = {'_by_team': team_map}
        _STARTED_GAME_SHEET_CACHE = (now, out)
        return out

    _STARTED_GAME_SHEET_CACHE = (now, {})
    return {}

def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == '':
            return None
        return float(v)
    except Exception:
        return None

def _proj_value_for_player(row: Optional[Dict[str, Any]]) -> float:
    """Return the player contribution used in team game projection aggregation.

    Prefer the uncentered current-projection value when available. Otherwise fall back
    to the legacy (Age + Rookie + EVO + EVD + PP + SH + GSAx) row shape.
    Non-numeric values are treated as 0. Handles comma decimal separators.
    """
    if not row:
        return 0.0
    raw_projected_value = _parse_locale_float(
        row.get('raw_projected_value')
        or row.get('rawProjectedValue')
        or row.get('RawProjectedValue')
    )
    if raw_projected_value is not None:
        return float(raw_projected_value)
    def f(k: str) -> float:
        try:
            v = row.get(k)
            if v is None:
                return 0.0
            # Use _parse_locale_float to handle comma decimal separators
            parsed = _parse_locale_float(v)
            return parsed if parsed is not None else 0.0
        except Exception:
            # try case-insensitive
            try:
                for key in row.keys():
                    if str(key).lower() == k.lower():
                        vv = row.get(key)
                        if vv is None:
                            return 0.0
                        parsed = _parse_locale_float(vv)
                        return parsed if parsed is not None else 0.0
            except Exception:
                pass
            return 0.0
    return (
        f('Age') + f('Rookie') + f('EVO') + f('EVD') + f('PP') + f('SH') + f('GSAx')
    )

_ROOKIE_FALLBACK = { 'D': -0.031768511, 'F': -0.024601581, 'G': -0.12 }

def _team_proj_from_lineup(team_abbrev: str, lineups_all: Dict[str, Any], proj_map: Dict[int, Dict[str, Any]]) -> float:
    t = (team_abbrev or '').upper()
    li = lineups_all.get(t) or {}
    total = 0.0
    for sec in ('forwards', 'defense', 'goalies'):
        arr = li.get(sec) or []
        if not isinstance(arr, list):
            continue
        for it in arr:
            try:
                # Exclude all EXT players from projections
                unit_val = str(it.get('unit') or '').upper()
                if unit_val == 'EXT':
                    continue
                pid = it.get('playerId')
                pos = (it.get('pos') or '').upper()[:1]
                # For goalies, include only G1 explicitly (others should be EXT already)
                if pos == 'G':
                    if unit_val and unit_val != 'G1':
                        continue
                if isinstance(pid, int) and pid in proj_map:
                    total += _proj_value_for_player(proj_map.get(pid))
                else:
                    total += _ROOKIE_FALLBACK.get(pos or 'F', _ROOKIE_FALLBACK['F'])
            except Exception:
                continue
    return float(total)

def _fetch_partner_odds_map(date_hint: Optional[str] = None) -> Dict[int, Dict[str, Any]]:
    """Fetch odds from NHL partner endpoint and map by game id to {'away': <odds>, 'home': <odds>}.
    Parsing is best-effort to tolerate upstream shape changes.
    """
    out: Dict[int, Dict[str, Any]] = {}
    # Try date-specific endpoint first if provided, then fallback to 'now'
    urls: List[str] = []
    if date_hint:
        urls.append(f'https://api-web.nhle.com/v1/partner-game/US/{date_hint}')
    urls.append('https://api-web.nhle.com/v1/partner-game/US/now')
    js = None
    for u in urls:
        try:
            r = requests.get(u, timeout=15)
            if r.status_code == 200:
                js = r.json()
                break
        except Exception:
            continue
    if js is None:
        return {}

    def extract_odds_from_node(node: Any) -> Tuple[Optional[Any], Optional[Any]]:
        """Return (away, home) odds from a node if present, else (None, None)."""
        away = None
        home = None
        if isinstance(node, dict):
            # Direct keys
            away = node.get('away') or node.get('awayPrice') or node.get('oddsAway') or node.get('priceAway') or node.get('A')
            home = node.get('home') or node.get('homePrice') or node.get('oddsHome') or node.get('priceHome') or node.get('H')
            # outcomes variants
            if away is None and home is None:
                outcomes = node.get('outcomes') or node.get('selections') or node.get('lines') or []
                if isinstance(outcomes, list):
                    for oc in outcomes:
                        if not isinstance(oc, dict):
                            continue
                        lbl = str(oc.get('label') or oc.get('name') or oc.get('type') or oc.get('outcome') or '').lower()
                        val = oc.get('americanOdds') or oc.get('oddsAmerican') or oc.get('price') or oc.get('american') or oc.get('odds')
                        # Some shapes might use side indicators
                        side = (oc.get('side') or oc.get('team') or oc.get('participant') or '').lower()
                        if 'away' in lbl or side == 'away' or side == 'visitor':
                            away = away if away is not None else val
                        elif 'home' in lbl or side == 'home':
                            home = home if home is not None else val
        return away, home

    def extract_ml_from_team_odds(team_obj: Any) -> Optional[Any]:
        """Given a team object that may contain an 'odds' list as in partner API, pick MONEY_LINE_2_WAY value.
        Fallback to MONEY_LINE_2_WAY_TNB if needed.
        """
        if not isinstance(team_obj, dict):
            return None
        lst = team_obj.get('odds')
        if not isinstance(lst, list):
            return None
        val_ml = None
        try:
            # Prefer exact MONEY_LINE_2_WAY
            for it in lst:
                if not isinstance(it, dict):
                    continue
                desc = str(it.get('description') or '').upper().strip()
                if desc == 'MONEY_LINE_2_WAY':
                    val_ml = it.get('value')
                    break
            # Fallback to MONEY_LINE_2_WAY_TNB
            if val_ml is None:
                for it in lst:
                    if not isinstance(it, dict):
                        continue
                    desc = str(it.get('description') or '').upper().strip()
                    if desc == 'MONEY_LINE_2_WAY_TNB':
                        val_ml = it.get('value')
                        break
        except Exception:
            return None
        return val_ml

    def try_add(gid_val, ml_node):
        if not ml_node:
            return
        try:
            gid = int(gid_val)
        except Exception:
            return
        away, home = extract_odds_from_node(ml_node)
        if away is None and home is None:
            return
        out[gid] = {'away': away, 'home': home}

    # Case 1: top-level list of games
    if isinstance(js, dict) and isinstance(js.get('games'), list):
        for g in (js.get('games') or []):
            gid = g.get('id') or g.get('gameId') or g.get('eventId')
            # First, support partner format where odds are inside team objects
            h = g.get('homeTeam') or {}
            a = g.get('awayTeam') or {}
            h_ml = extract_ml_from_team_odds(h)
            a_ml = extract_ml_from_team_odds(a)
            if h_ml is not None or a_ml is not None:
                try:
                    gid2 = g.get('gameId') or gid
                    try_add(gid2, {'home': h_ml, 'away': a_ml})
                except Exception:
                    pass
            # Also fall back to legacy bets/markets shapes if present
            bets = g.get('bets') or g.get('markets') or g.get('sportsbook') or g.get('sportsbookLines') or {}
            if isinstance(bets, dict):
                ml = bets.get('MONEY_LINE_2_WAY')
                if not ml:
                    for v in bets.values():
                        if isinstance(v, list):
                            for it in v:
                                if isinstance(it, dict) and str(it.get('market') or it.get('type')).upper() == 'MONEY_LINE_2_WAY':
                                    try_add(gid, it)
                        elif isinstance(v, dict) and str(v.get('market') or v.get('type')).upper() == 'MONEY_LINE_2_WAY':
                            try_add(gid, v)
                else:
                    try_add(gid, ml)
            elif isinstance(bets, list):
                for it in bets:
                    if isinstance(it, dict) and str(it.get('market') or it.get('type')).upper() == 'MONEY_LINE_2_WAY':
                        try_add(gid, it)
    # Fallback: recursive search for MONEY_LINE_2_WAY under any node with an id context
    if not out:
        def walk(node, ctx_id=None):
            if isinstance(node, dict):
                gid = node.get('id') or node.get('gameId') or node.get('eventId') or ctx_id
                for k, v in node.items():
                    if k == 'MONEY_LINE_2_WAY':
                        try_add(gid, v if isinstance(v, dict) else None)
                    else:
                        walk(v, gid)
            elif isinstance(node, list):
                for it in node:
                    walk(it, ctx_id)
        walk(js)
    return out


# --- Model utilities ---
def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def _model_dir() -> str:
    return os.path.join(_project_root(), 'Model')

def load_model_file(fname: str) -> Optional[Any]:
    """Module-level model loader with cache."""
    if fname in _MODEL_CACHE:
        return _MODEL_CACHE[fname]
    path = os.path.join(_model_dir(), fname)
    if not os.path.exists(path):
        return None
    try:
        m = joblib.load(path)
        try:
            max_items = max(1, int(os.getenv('MODEL_CACHE_MAX_ITEMS', '24') or '24'))
        except Exception:
            max_items = 24
        _dict_set_bounded(_MODEL_CACHE, fname, m, max_items=max_items)
        return m
    except Exception:
        return None

def current_season_id(now: Optional[datetime] = None) -> int:
    d = now or datetime.utcnow()
    y = d.year
    if d.month >= 9:
        start_y = y
        end_y = y + 1
    else:
        start_y = y - 1
        end_y = y
    return start_y * 10000 + end_y

def preload_common_models() -> None:
    """Eager-load central window models for the current season to reduce cold-start latency."""
    try:
        s = current_season_id()
        a = int(str(s)[:4]); b = int(str(s)[4:])
        s_prev = (a-1)*10000 + (b-1)
        s_next = (a+1)*10000 + (b+1)
        middle = f"{s_prev}_{s_next}.pkl"
        for prefix in ('xgbs', 'xgb', 'xgb2'):
            load_model_file(f"{prefix}_{middle}")
    except Exception:
        pass


@main_bp.route('/api/team/<team_code>/<int:season>/schedule')
def api_team_schedule(team_code: str, season: int):
    """Team schedule for a given season using NHL club-schedule-season endpoint."""
    url = f"https://api-web.nhle.com/v1/club-schedule-season/{team_code.upper()}/{season}"
    try:
        r = requests.get(url, timeout=20)
    except Exception:
        return jsonify({'error': 'Failed to fetch schedule'}), 502
    if r.status_code != 200:
        return jsonify({'error': 'Failed to fetch schedule'}), 502
    data = r.json()
    games_out = []
    for g in data.get('games', []) or []:
        game_date = g.get('gameDate') or g.get('startTimeUTC')
        try:
            dt = datetime.fromisoformat(game_date.replace('Z', '+00:00')) if game_date else None
        except Exception:
            dt = None
        home = (g.get('homeTeam') or {}).get('abbrev')
        away = (g.get('awayTeam') or {}).get('abbrev')
        home_score = (g.get('homeTeam') or {}).get('score')
        away_score = (g.get('awayTeam') or {}).get('score')
        opp = away if (home and home == team_code.upper()) else home
        is_home = bool(home and home == team_code.upper())
        status = g.get('gameState') or g.get('gameStatus')
        last_period_type = (g.get('gameOutcome') or {}).get('lastPeriodType') or (g.get('periodDescriptor') or {}).get('periodType')
        games_out.append({
            'date': dt.isoformat() if dt else game_date,
            'home': home,
            'away': away,
            'opponent': opp,
            'is_home': is_home,
            'status': status,
            'gameType': g.get('gameType') or g.get('gameTypeId'),
            'home_score': home_score,
            'away_score': away_score,
            'lastPeriodType': last_period_type,
            'id': g.get('id') or g.get('gamePk'),
        })
    return jsonify(games_out)

# Alias to match frontend fetch path `/api/schedule/{team}/{season}`
@main_bp.route('/api/schedule/<team_code>/<int:season>')
def api_schedule_alias(team_code: str, season: int):
    return api_team_schedule(team_code, season)


@main_bp.route('/game/<int:game_id>')
def game_page(game_id: int):
    """Render a game detail page."""
    teams = TEAM_ROWS
    return render_template('game.html', game_id=game_id, teams=teams, active_tab='Schedule', show_season_state=False)


@main_bp.route('/api/game/<int:game_id>/boxscore')
def api_game_boxscore(game_id: int):
    # Allow bypassing cache for live refreshes
    try:
        force = str(request.args.get('force', '')).lower() in ('1', 'true', 'yes', 'y', 'force')
    except Exception:
        force = False
    # Serve from cache if available and not forced
    if not force:
        try:
            ttl = int(os.getenv('BOX_CACHE_TTL_SECONDS', '600'))
            try:
                max_items = max(1, int(os.getenv('BOX_CACHE_MAX_ITEMS', '64') or '64'))
            except Exception:
                max_items = 64
            _cache_prune_ttl_and_size(_BOX_CACHE, ttl_s=ttl, max_items=max_items)
            cached = _cache_get(_BOX_CACHE, int(game_id), ttl)
            if cached:
                return jsonify(cached)
        except Exception:
            pass
    url = f'https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore'
    try:
        resp = requests.get(url, timeout=20)
    except Exception:
        return jsonify({'error': 'Fetch failed'}), 502
    if resp.status_code != 200:
        return jsonify({'error': 'Upstream error', 'status': resp.status_code}), 502
    data = resp.json()
    # Pass through mostly untouched; rename id to gameId for consistency
    if 'id' in data and 'gameId' not in data:
        data['gameId'] = data['id']
    try:
        ttl = int(os.getenv('BOX_CACHE_TTL_SECONDS', '600'))
    except Exception:
        ttl = 600
    try:
        max_items = max(1, int(os.getenv('BOX_CACHE_MAX_ITEMS', '64') or '64'))
    except Exception:
        max_items = 64
    try:
        _cache_set_multi_bounded(_BOX_CACHE, int(game_id), data, ttl_s=ttl, max_items=max_items)
    except Exception:
        pass
    resp_json = jsonify(data)
    # Add no-store when forced to help downstream avoid caching
    if force:
        try:
            resp_json.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
    return resp_json


@main_bp.route('/api/game/<int:game_id>/right-rail')
def api_game_right_rail(game_id: int):
    """Proxy NHL right-rail endpoint for a game to avoid browser CORS."""
    try:
        force = str(request.args.get('force', '')).lower() in ('1', 'true', 'yes', 'y', 'force')
    except Exception:
        force = False
    url = f'https://api-web.nhle.com/v1/gamecenter/{game_id}/right-rail'
    try:
        resp = requests.get(url, timeout=20)
    except Exception:
        return jsonify({'error': 'Fetch failed'}), 502
    if resp.status_code != 200:
        return jsonify({'error': 'Upstream error', 'status': resp.status_code}), 502
    try:
        data = resp.json()
    except Exception:
        # Fall back to raw text if upstream is not JSON
        return jsonify({'error': 'Invalid upstream format'}), 502
    j = jsonify(data)
    if force:
        try:
            j.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
    return j


@main_bp.route('/api/game/<int:game_id>/play-by-play')
def api_game_pbp(game_id: int):
    """Fetch NHL play-by-play and map to requested wide schema."""
    # Serve from disk/memory cache when available; for live games use short TTL
    try:
        force = str(request.args.get('force', '')).lower() in ('1', 'true', 'yes', 'y', 'force')
    except Exception:
        force = False
    live_ttl = 5  # seconds for live
    std_ttl = int(os.getenv('PBP_CACHE_TTL_SECONDS', '600'))
    try:
        max_items = max(1, int(os.getenv('PBP_CACHE_MAX_ITEMS', '24') or '24'))
    except Exception:
        max_items = 24
    disk_path = _disk_cache_path_pbp(int(game_id))
    if not force:
        try:
            # Try disk cache first (has metadata such as gameState)
            if os.path.exists(disk_path):
                import json, time
                with open(disk_path, 'r', encoding='utf-8') as f:
                    js = json.load(f)
                ts = float(js.get('_cachedAt', 0.0))
                gstate = str(js.get('gameState') or '').upper()
                ttl = live_ttl if gstate in ('LIVE', 'SCHEDULED', 'PREVIEW', 'INPROGRESS') else std_ttl
                if ts and (time.time() - ts) < ttl:
                    return jsonify({k: v for k, v in js.items() if not k.startswith('_')})
            # Try in-memory cache if disk miss
            _cache_prune_ttl_and_size(_PBP_CACHE, ttl_s=std_ttl, max_items=max_items)
            cached = _cache_get(_PBP_CACHE, int(game_id), std_ttl)
            if cached:
                return jsonify(cached)
        except Exception:
            pass
    url = f'https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play'
    try:
        resp = requests.get(url, timeout=25)
    except Exception:
        return jsonify({'error': 'Fetch failed'}), 502
    if resp.status_code != 200:
        return jsonify({'error': 'Upstream error', 'status': resp.status_code}), 502
    data = resp.json()
    # Capture upstream game state
    game_state = str(data.get('gameState') or data.get('gameStatus') or '').upper()

    # Fetch skater bios for this game to retrieve shoots/catches for players (optional for performance)
    shoots_map: Dict[int, str] = {}
    try:
        if os.getenv('FETCH_BIOS', '0') == '1':
            gid_for_bios = data.get('id') or game_id
            bios_url = f"https://api.nhle.com/stats/rest/en/skater/bios?limit=-1&start=0&cayenneExp=gameId={gid_for_bios}"
            r_bios = requests.get(bios_url, timeout=15)
            if r_bios.status_code == 200:
                bios_json = r_bios.json()
                rows = bios_json.get('data') if isinstance(bios_json, dict) else []
                if isinstance(rows, list):
                    for row in rows:
                        try:
                            pid = row.get('playerId')
                            sc = row.get('shootsCatches') or row.get('shoots') or row.get('ShootsCatches')
                            if isinstance(pid, int) and sc:
                                shoots_map[pid] = str(sc).strip().upper()[:1]
                        except Exception:
                            continue
    except Exception:
        shoots_map = {}

    # Fetch landing endpoint for goal highlight video URLs
    goal_highlights: Dict[int, str] = {}
    try:
        landing_url = f'https://api-web.nhle.com/v1/gamecenter/{game_id}/landing'
        r_land = requests.get(landing_url, timeout=15)
        if r_land.status_code == 200:
            land_data = r_land.json()
            for per in (land_data.get('summary', {}).get('scoring') or []):
                for gl in (per.get('goals') or []):
                    eid = gl.get('eventId')
                    clip = gl.get('highlightClipSharingUrl')
                    if eid is not None and clip:
                        goal_highlights[int(eid)] = clip
    except Exception:
        goal_highlights = {}

    plays_raw = data.get('plays', []) if isinstance(data, dict) else []
    away_team = (data.get('awayTeam') or {})
    home_team = (data.get('homeTeam') or {})
    away_id = away_team.get('id')
    home_id = home_team.get('id')
    away_abbrev = away_team.get('abbrev')
    home_abbrev = home_team.get('abbrev')
    # RinkVenue per request: Season-HomeTeam
    rink_venue_value = None
    try:
        sv = data.get('season')
        if sv is not None and home_abbrev:
            rink_venue_value = f"{sv}-{home_abbrev}"
    except Exception:
        rink_venue_value = None
    roster = {r.get('playerId'): r for r in data.get('rosterSpots', [])}

    def player_name(pid: Optional[int]) -> Optional[str]:
        r = roster.get(pid)
        if not r:
            return None
        fn = (r.get('firstName') or {}).get('default') if isinstance(r.get('firstName'), dict) else r.get('firstName')
        ln = (r.get('lastName') or {}).get('default') if isinstance(r.get('lastName'), dict) else r.get('lastName')
        if fn and ln:
            return f"{fn} {ln}".strip()
        return fn or ln

    def parse_time_to_seconds(t: str) -> Optional[int]:
        try:
            mm, ss = t.split(':')
            return int(mm) * 60 + int(ss)
        except Exception:
            return None

    def strength_from_situation(code: str, event_owner_team_id: Optional[int]) -> str:
        s = code or ''
        if not s:
            return ''
        # Empty-net handling: if situation code starts with '0' => Away ENF/Home ENA; ends with '0' => Home ENF/Away ENA
        away_empty = len(s) >= 1 and s[0] == '0'
        home_empty = len(s) >= 1 and s[-1] == '0'
        if away_empty or home_empty:
            if event_owner_team_id == away_id:
                return 'ENF' if away_empty else 'ENA'
            if event_owner_team_id == home_id:
                return 'ENF' if home_empty else 'ENA'
            # Unknown owner: fall back to neutral numeric state when possible
            # If both empty or unknown, return empty string to avoid misleading tag
            return ''
        # Numeric skater counts: typical 4-digit code, use middle digits for A/H skaters
        if len(s) == 4 and s.isdigit():
            away_skaters = int(s[1])
            home_skaters = int(s[2])
            if event_owner_team_id is None:
                return f"{away_skaters}v{home_skaters}"
            if event_owner_team_id == home_id:
                return f"{home_skaters}v{away_skaters}"
            if event_owner_team_id == away_id:
                return f"{away_skaters}v{home_skaters}"
            return f"{away_skaters}v{home_skaters}"
        # Fallback to raw code if format unrecognized
        return s

    # Running score tracking (pre-event state). We'll update AFTER mapping current event.
    running_away = 0
    running_home = 0

    # Determine orientation using SUM(x) of shot attempts grouped by (period, event-owner team)
    period_team_sum_x: Dict[Tuple[int, int], float] = {}
    period_sum_all: Dict[int, float] = {}
    for pl in plays_raw:
        pd = ((pl.get('periodDescriptor') or {}).get('number'))
        try:
            pd_key = int(pd) if pd is not None else None
        except Exception:
            pd_key = None
        if not pd_key:
            continue
        tc = pl.get('typeCode')
        if tc not in (505, 506, 507, 508):  # shot attempts
            continue
        d0 = pl.get('details') or {}
        x0 = d0.get('xCoord')
        if x0 is None:
            continue
        try:
            xx = float(x0)
        except Exception:
            continue
        period_sum_all[pd_key] = period_sum_all.get(pd_key, 0.0) + xx
        owner0 = d0.get('eventOwnerTeamId')
        if isinstance(owner0, int) and owner0 in (home_id, away_id):
            key_t = (pd_key, owner0)
            period_team_sum_x[key_t] = period_team_sum_x.get(key_t, 0.0) + xx

    # Precompute shift slices and on-ice players from our own shifts endpoint
    slices: List[Tuple[int, int, int]] = []  # (start, end, ShiftIndex)
    starts: List[int] = []
    slice_players: Dict[int, List[Dict]] = {}
    try:
        resp_shifts = api_game_shifts(game_id)
        js = None
        status_code = 200
        if isinstance(resp_shifts, tuple):
            resp_obj, status_code = resp_shifts
            js = resp_obj.get_json(silent=True) if hasattr(resp_obj, 'get_json') else None
        else:
            js = resp_shifts.get_json(silent=True) if hasattr(resp_shifts, 'get_json') else None
        rows = (js.get('shifts') or []) if isinstance(js, dict) and status_code == 200 else []
        by_idx: Dict[int, Tuple[int, int]] = {}
        for r in rows:
            si = r.get('ShiftIndex'); st = r.get('Start'); en = r.get('End')
            if si is None or st is None or en is None:
                continue
            try:
                sii = int(si); sti = int(st); eni = int(en)
            except Exception:
                continue
            if sii not in by_idx:
                by_idx[sii] = (sti, eni)
            else:
                a, b = by_idx[sii]
                by_idx[sii] = (min(a, sti), max(b, eni))
            # collect players for this slice
            pl_id = r.get('PlayerID'); pl_nm = r.get('Name'); pl_pos = (r.get('Position') or '').upper(); pl_tm = r.get('Team')
            if pl_id is not None and pl_tm is not None:
                slice_players.setdefault(sii, []).append({'PlayerID': pl_id, 'Name': pl_nm, 'Position': pl_pos, 'Team': pl_tm})
        slices = sorted([(v[0], v[1], k) for k, v in by_idx.items()], key=lambda x: x[0])
        starts = [s for s, _, _ in slices]
    except Exception:
        slices = []
        starts = []
        slice_players = {}

    # Precompute on-ice string fields per ShiftIndex
    onice_cache: Dict[int, Dict[str, Optional[str]]] = {}
    for si, plist in slice_players.items():
        # Partition by team and position
        def filter_and_sort(team_abbr: str, pos_code: str):
            flt = [p for p in plist if (p.get('Team') == team_abbr and (p.get('Position') or '').upper() == pos_code)]
            # Ensure numeric PlayerID sort
            def to_int(x):
                try:
                    return int(x)
                except Exception:
                    return 0
            flt_sorted = sorted(flt, key=lambda p: to_int(p.get('PlayerID')))
            ids = ' '.join(str(p.get('PlayerID')) for p in flt_sorted if p.get('PlayerID') is not None)
            names = ' - '.join(str(p.get('Name')) for p in flt_sorted if p.get('Name'))
            return ids or None, names or None

        hf_id, hf_nm = filter_and_sort(str(home_abbrev or ''), 'F')
        hd_id, hd_nm = filter_and_sort(str(home_abbrev or ''), 'D')
        hg_id, hg_nm = filter_and_sort(str(home_abbrev or ''), 'G')
        af_id, af_nm = filter_and_sort(str(away_abbrev or ''), 'F')
        ad_id, ad_nm = filter_and_sort(str(away_abbrev or ''), 'D')
        ag_id, ag_nm = filter_and_sort(str(away_abbrev or ''), 'G')

        onice_cache[si] = {
            'Home_Forwards_ID': hf_id, 'Home_Forwards': hf_nm,
            'Home_Defenders_ID': hd_id, 'Home_Defenders': hd_nm,
            'Home_Goalie_ID': hg_id, 'Home_Goalie': hg_nm,
            'Away_Forwards_ID': af_id, 'Away_Forwards': af_nm,
            'Away_Defenders_ID': ad_id, 'Away_Defenders': ad_nm,
            'Away_Goalie_ID': ag_id, 'Away_Goalie': ag_nm,
        }

    def find_shift_index_for_event(gt: int, event_key: Optional[str]) -> Optional[int]:
        if not slices:
            return None
        ev = (event_key or '').lower()
        ev_norm = ev.replace('-', '_')
        # Check if on a slice start boundary
        k = bisect.bisect_left(starts, gt)
        if k < len(starts) and starts[k] == gt:
            # Boundary rule: faceoff/period_start take later (max) slice; others previous (min) slice
            if ev_norm in ('faceoff', 'period_start'):
                idx = k
            else:
                idx = k - 1 if k > 0 else k
            return slices[idx][2] if 0 <= idx < len(slices) else None
        # Otherwise, select slice with start <= gt < end
        i = bisect.bisect_right(starts, gt) - 1
        if i < 0 or i >= len(slices):
            return None
        s0, e0, si0 = slices[i]
        if gt < e0:
            return si0
        if gt == e0:
            # End boundary equals next start
            if (i + 1) < len(slices) and starts[i + 1] == gt:
                if ev_norm in ('faceoff', 'period_start'):
                    return slices[i + 1][2]
                return si0
        return None

    # StrengthState2 mapping per spec
    def remap_strength_state(s: Optional[str]) -> Optional[str]:
        if not s:
            return s
        s2 = str(s).upper()
        if s2 in ("5V4", "ENF"):
            return "PP1"
        if s2 in ("5V3", "4V3"):
            return "PP2"
        if s2 in ("4V5", "3V5", "3V4"):
            return "SH"
        return s

    # BoxID2 mapping per spec (combines left/right and splits by handedness)
    def compute_boxid2(boxid: Optional[str], shoots: Optional[str]) -> str:
        b = (boxid or '').upper().strip()
        h = (shoots or '').upper().strip()[:1] if shoots else ''
        # Helper shortcuts
        is_r_or_null = (h == 'R' or h == '')
        is_l_or_null = (h == 'L' or h == '')
        if b in ('O01', 'O03'):
            return 'O01'
        if b == 'O02':
            return 'O02'
        if b == 'O04':
            return 'O04-W' if is_r_or_null else 'O04-S'
        if b == 'O05':
            return 'O05-W' if is_r_or_null else 'O05-S'
        if b == 'O06':
            return 'O06-W' if is_r_or_null else 'O06-S'
        if b == 'O07':
            return 'O07'
        if b == 'O08':
            return 'O06-W' if is_l_or_null else 'O06-S'
        if b == 'O09':
            return 'O05-W' if is_l_or_null else 'O05-S'
        if b == 'O10':
            return 'O04-W' if is_l_or_null else 'O04-S'
        if b == 'O11':
            return 'O11'
        if b == 'O12':
            return 'O12-W' if is_r_or_null else 'O12-S'
        if b == 'O13':
            return 'O13-W' if is_r_or_null else 'O13-S'
        if b == 'O14':
            return 'O14-W' if is_r_or_null else 'O14-S'
        if b == 'O15':
            return 'O15'
        if b == 'O16':
            return 'O14-W' if is_l_or_null else 'O14-S'
        if b == 'O17':
            return 'O13-W' if is_l_or_null else 'O13-S'
        if b == 'O18':
            return 'O12-W' if is_l_or_null else 'O12-S'
        if b == 'O19':
            return 'O19-W' if is_r_or_null else 'O19-S'
        if b == 'O20':
            return 'O20-W' if is_r_or_null else 'O20-S'
        if b == 'O21':
            return 'O21'
        if b == 'O22':
            return 'O20-W' if is_l_or_null else 'O20-S'
        if b == 'O23':
            return 'O19-W' if is_l_or_null else 'O19-S'
        if b == 'O24':
            return 'O24-W' if is_r_or_null else 'O24-S'
        if b == 'O25':
            return 'O25'
        if b == 'O26':
            return 'O24-W' if is_l_or_null else 'O24-S'
        return 'D_or_N'

    mapped: List[Dict] = []
    for idx_pl, pl in enumerate(plays_raw):
        period = ((pl.get('periodDescriptor') or {}).get('number'))
        time_in_period = pl.get('timeInPeriod') or ''
        type_code = pl.get('typeCode')
        event_key = pl.get('typeDescKey')
        details = pl.get('details') or {}
        situation = pl.get('situationCode') or ''
        strength = strength_from_situation(situation, details.get('eventOwnerTeamId'))
        x = details.get('xCoord')
        y = details.get('yCoord')
        zone = details.get('zoneCode')
        reason = details.get('reason')
        secondary_reason = details.get('secondaryReason')
        type_code2 = details.get('typeCode') if isinstance(details.get('typeCode'), str) else None
        pen_dur = details.get('duration')
        event_owner = details.get('eventOwnerTeamId')
        event_team_abbrev = away_abbrev if event_owner == away_id else home_abbrev if event_owner == home_id else None
        opponent_abbrev = home_abbrev if event_team_abbrev == away_abbrev else away_abbrev if event_team_abbrev == home_abbrev else None
        goalie_id = details.get('goalieInNetId')
        goalie_name = player_name(goalie_id) if goalie_id else None

        # Collect involved player ids in priority order
        candidate_ids: List[int] = []
        for key in [
            'scoringPlayerId', 'shootingPlayerId', 'playerId', 'hittingPlayerId', 'hitteePlayerId',
            'assist1PlayerId', 'assist2PlayerId', 'blockingPlayerId', 'losingPlayerId', 'winningPlayerId',
            'committedByPlayerId', 'drawnByPlayerId'
        ]:
            pid = details.get(key)
            if pid and pid not in candidate_ids:
                candidate_ids.append(pid)
        p1_id = candidate_ids[0] if len(candidate_ids) > 0 else None

        p2_id = candidate_ids[1] if len(candidate_ids) > 1 else None
        p3_id = candidate_ids[2] if len(candidate_ids) > 2 else None
        p1_name = player_name(p1_id) if p1_id else None
        p2_name = player_name(p2_id) if p2_id else None
        p3_name = player_name(p3_id) if p3_id else None

        # Shot / goal classification
        is_goal = (type_code == 505)
        is_sog = (type_code == 506) or is_goal
        is_miss = (type_code == 507)
        is_block = (type_code == 508)
        # Blocked shots: swap zone O <-> D for display only (coords are shooter-perspective upstream)
        if is_block and zone in ('O', 'D'):
            zone = 'O' if zone == 'D' else 'D'

        # Normalize coordinates so offensive zone is to the right (positive x) for the period
        nx: Optional[float] = None
        ny: Optional[float] = None
        try:
            pd_key2 = int(period) if period is not None else None
        except Exception:
            pd_key2 = None
        sign = 1
        if pd_key2 is not None:
            if isinstance(event_owner, int) and event_owner in (home_id, away_id):
                key = (pd_key2, event_owner)
                if key in period_team_sum_x:
                    sign = 1 if period_team_sum_x[key] >= 0 else -1
                else:
                    opp = home_id if event_owner == away_id else away_id if event_owner == home_id else None
                    if isinstance(opp, int) and (pd_key2, opp) in period_team_sum_x:
                        sign = -1 if period_team_sum_x[(pd_key2, opp)] >= 0 else 1
                    else:
                        sign = 1 if period_sum_all.get(pd_key2, 0.0) >= 0 else -1
            else:
                sign = 1 if period_sum_all.get(pd_key2, 0.0) >= 0 else -1
        try:
            nx = (float(x) * sign) if x is not None else None
        except Exception:
            nx = None
        try:
            ny = (float(y) * sign) if y is not None else None
        except Exception:
            ny = None

        # ScoreState: goal differential from perspective of event team BEFORE applying current event.
        if event_owner == away_id:
            score_state_val = running_away - running_home
        elif event_owner == home_id:
            score_state_val = running_home - running_away
        else:
            score_state_val = running_away - running_home
        # Bounded ScoreState2 per spec
        score_state2_val = -3 if score_state_val < -2 else (3 if score_state_val > 2 else score_state_val)

        # Possession attempts (Corsi/Fenwick)
        corsi = 1 if (is_goal or is_sog or is_miss or is_block) and event_team_abbrev else 0
        fenwick = 1 if (is_goal or is_sog or is_miss) and event_team_abbrev else 0
        shot = 1 if is_sog else 0

        # Position & shoots from primary player if available
        position = None
        shoots = None
        if p1_id and p1_id in roster:
            pos_code = roster[p1_id].get('positionCode')
            if pos_code:
                c = str(pos_code).strip().upper()[:1]
                position = 'F' if c in ('C', 'L', 'R') else c
        # Shoots from bios map (fallback only if present)
        if p1_id and p1_id in shoots_map:
            shoots = shoots_map.get(p1_id)

        # gameTime calculation in seconds
        secs_elapsed = parse_time_to_seconds(time_in_period) or 0
        try:
            game_time = ((period - 1) * 20 * 60 + secs_elapsed) if period else secs_elapsed
        except Exception:
            game_time = secs_elapsed

        # Venue from event team perspective: Home/Away
        venue_ha = 'Home' if event_owner == home_id else ('Away' if event_owner == away_id else '')

        # Shot geometry (feet/degrees) relative to net at (89,0), using normalized coords
        shot_distance = None
        shot_angle = None
        if nx is not None and ny is not None and (is_goal or is_sog or is_miss or is_block):
            try:
                dx = 89.0 - float(nx)
                dy = 0.0 - float(ny)
                dist = (dx * dx + dy * dy) ** 0.5
                ang = math.degrees(math.atan2(abs(dy), dx if dx != 0 else 1e-6))
                shot_distance = round(dist, 2)
                shot_angle = round(ang, 2)
            except Exception:
                pass

        # SeasonState: map gameType to 'regular' or 'playoffs'
        # gameType: 1=preseason, 2=regular, 3=playoffs, 4=allstar
        gt = data.get('gameType')
        _GT_MAP = {'2': 'regular', '3': 'playoffs'}
        season_state = _GT_MAP.get(str(gt), 'other')

        # EventIndex: GameID*10000 + Index (1-based index in plays list)
        try:
            gid_for_idx = int(data.get('id') or game_id)
        except Exception:
            gid_for_idx = int(game_id)
        event_index_val = gid_for_idx * 10000 + (idx_pl + 1)

        # ShiftIndex from slices with boundary rules
        shift_index_val = find_shift_index_for_event(int(game_time), event_key)

        # Prepare on-ice fields from cache by ShiftIndex (if available)
        oi = onice_cache.get(shift_index_val or -1, {})

        # Box geometry left-join on integer grid: round normalized coords and match exactly
        box_id = None
        box_rev = None
        box_size = None
        xi = None
        yi = None
        if nx is not None and ny is not None:
            try:
                xf = float(nx); yf = float(ny)
                # Clamp to rink bounds first
                xf = max(-100.0, min(100.0, xf))
                yf = max(-42.0, min(42.0, yf))
                xi = int(round(xf))
                yi = int(round(yf))
                # Many BoxID grids are defined for x >= 0 and use BoxID_rev for the mirrored side.
                # Use abs(x) to find the grid cell, and keep both BoxID and BoxID_rev from the CSV.
                rec = _get_boxid_map().get((xi, yi))
                if rec:
                    box_id, box_rev, box_size = rec[0], rec[1], rec[2]
            except Exception:
                pass

        # shotType2 normalization per spec
        raw_shot_type = details.get('shotType')
        try:
            shot_type_norm = str(raw_shot_type or '').strip()
        except Exception:
            shot_type_norm = ''
        allowed_types = {"wrist", "tip-in", "snap", "slap", "backhand", "deflected", "wrap-around", ""}
        st_lower = shot_type_norm.lower()
        shot_type2 = shot_type_norm if st_lower in allowed_types else 'other'

        # StrengthState2 remap
        strength2 = remap_strength_state(strength)

        mapped.append({
            'GameID': data.get('id'),
            'Season': data.get('season'),
            'SeasonState': season_state,
            'Venue': venue_ha,
            'Period': period,
            'gameTime': int(game_time),
            'StrengthState': strength,
            'StrengthState2': strength2,
            'typeCode': type_code,
            'Event': event_key,
            'x': nx,
            'y': ny,
            'X': xi,
            'Y': yi,
            'Zone': zone,
            'reason': reason,
            'shotType': details.get('shotType'),
            'shotType2': shot_type2,
            'secondaryReason': secondary_reason,
            'typeCode2': type_code2,
            'PEN_duration': pen_dur,
            'EventTeam': event_team_abbrev,
            'Opponent': opponent_abbrev,
            'Goalie_ID': goalie_id,
            'Goalie': goalie_name,
            'Player1_ID': p1_id,
            'Player1': p1_name,
            'Player2_ID': p2_id,
            'Player2': p2_name,
            'Player3_ID': p3_id,
            'Player3': p3_name,
            'Corsi': corsi,
            'Fenwick': fenwick,
            'Shot': shot,
            'Goal': 1 if is_goal else 0,
            'EventIndex': event_index_val,
            'ShiftIndex': shift_index_val,
            'ScoreState': score_state_val,
            'ScoreState2': score_state2_val,
            'Home_Forwards_ID': oi.get('Home_Forwards_ID'),
            'Home_Forwards': oi.get('Home_Forwards'),
            'Home_Defenders_ID': oi.get('Home_Defenders_ID'),
            'Home_Defenders': oi.get('Home_Defenders'),
            'Home_Goalie_ID': oi.get('Home_Goalie_ID'),
            'Home_Goalie': oi.get('Home_Goalie'),
            'Away_Forwards_ID': oi.get('Away_Forwards_ID'),
            'Away_Forwards': oi.get('Away_Forwards'),
            'Away_Defenders_ID': oi.get('Away_Defenders_ID'),
            'Away_Defenders': oi.get('Away_Defenders'),
            'Away_Goalie_ID': oi.get('Away_Goalie_ID'),
            'Away_Goalie': oi.get('Away_Goalie'),
            'BoxID': box_id,
            'BoxID_rev': box_rev,
            'BoxSize': box_size,
            'BoxID2': None,  # fill after row assembled using Shoots
            # snake_case aliases for downstream consumers expecting these names
            'box_id': box_id,
            'box_rev': box_rev,
            'box_size': box_size,
            'ShotDistance': shot_distance,
            'ShotAngle': shot_angle,
            'Position': position,
            'Shoots': shoots,
            'RinkVenue': rink_venue_value,
            'HighlightUrl': goal_highlights.get(pl.get('eventId')) if is_goal else None,
            'LastEvent': None,  # to be computed post-pass
            'xG_F': None,
            'xG_S': None,
            'xG_F2': None,
        })

        # AFTER mapping current play, update running score when this is a goal
        if is_goal:
            if 'awayScore' in details and 'homeScore' in details and details.get('awayScore') is not None and details.get('homeScore') is not None:
                try:
                    ra = details.get('awayScore')
                    rh = details.get('homeScore')
                    running_away = int(ra) if ra is not None else running_away
                    running_home = int(rh) if rh is not None else running_home
                except Exception:
                    if event_owner == away_id:
                        running_away += 1
                    elif event_owner == home_id:
                        running_home += 1
            else:
                if event_owner == away_id:
                    running_away += 1
                elif event_owner == home_id:
                    running_home += 1

    # Post-process BoxID2 and LastEvent
    last_event_name: Optional[str] = None
    last_game_time: Optional[int] = None
    for row in mapped:
        # BoxID2 depends on BoxID and Shoots
        row['BoxID2'] = compute_boxid2(row.get('BoxID'), row.get('Shoots'))
        # LastEvent labeling per spec
        if row.get('Fenwick') == 1:
            prev_ev = last_event_name or ''
            gt = row.get('gameTime')
            tsle = (gt - last_game_time) if (gt is not None and last_game_time is not None) else None
            if tsle is not None and tsle < 4 and prev_ev in ('blocked-shot', 'shot-on-goal', 'takeaway', 'giveaway'):
                row['LastEvent'] = 'Rebound'
            elif tsle is not None and tsle < 4:
                row['LastEvent'] = 'Quick'
            else:
                row['LastEvent'] = 'None'
        else:
            row['LastEvent'] = ''
        # Update lag trackers for all events
        last_event_name = row.get('Event') or last_event_name
        last_game_time = row.get('gameTime') if row.get('gameTime') is not None else last_game_time

    # xG computations using pickled models, skipping ENA strength
    compute_xg = (request.args.get('xg', '1') != '0') and (os.getenv('XG_DISABLED', '0') != '1')
    try:
        if not compute_xg:
            raise Exception('xg_disabled')

        # Helper: map season integer like 20142015 to previous, current, next for 3 sliding windows
        def season_prev(s: int) -> int:
            a = int(str(s)[:4]); b = int(str(s)[4:])
            return (a-1)*10000 + (b-1)
        def season_next(s: int) -> int:
            a = int(str(s)[:4]); b = int(str(s)[4:])
            return (a+1)*10000 + (b+1)

        # Low-memory per-row one-hot encoding without building a full dummy matrix
        base_feature_cols = [
            "Venue", "shotType2", "ScoreState2", "RinkVenue",
            "StrengthState2", "BoxID2", "LastEvent"
        ]

        def _required_columns_for_model(m: Any) -> Optional[List[str]]:
            try:
                key = f"cols_id_{id(m)}"
                if key in _FEATURE_COLS_CACHE:
                    return _FEATURE_COLS_CACHE[key]
                cols = None
                if hasattr(m, 'feature_names_in_'):
                    cols = list(getattr(m, 'feature_names_in_'))
                elif hasattr(m, 'get_booster'):
                    booster = m.get_booster()
                    cols = getattr(booster, 'feature_names', None)
                if cols:
                    try:
                        max_items = max(1, int(os.getenv('FEATURE_COLS_CACHE_MAX_ITEMS', '512') or '512'))
                    except Exception:
                        max_items = 512
                    _dict_set_bounded(_FEATURE_COLS_CACHE, key, cols, max_items=max_items)
                return cols
            except Exception:
                return None

        def _vectorize_row_for_model(row_obj: Dict[str, Any], m: Any):
            cols = _required_columns_for_model(m)
            if not cols:
                return None, None  # can't align reliably
            # Build one-hot vector aligned to cols
            vec = [0.0] * len(cols)
            # Precompute string values for the row features
            vals = {}
            for c in base_feature_cols:
                v = row_obj.get(c)
                vals[c] = 'missing' if v is None else str(v)
            # For each required column, parse as prefix_value and set 1.0 if match
            for i, cname in enumerate(cols):
                if '_' not in cname:
                    # Unexpected; leave as 0.0
                    continue
                base, suffix = cname.split('_', 1)
                rv = vals.get(base)
                if rv is None:
                    continue
                if rv == suffix:
                    vec[i] = 1.0
            return vec, cols

        def predict_avg_for_row(row_obj: Dict[str, Any], season_val: Optional[int], model_prefix: str) -> Optional[float]:
            if season_val is None:
                return None
            s_cur = int(season_val)
            # Special-case for 20252026 games: use ..._20222023_20242025.pkl
            if s_cur == 20252026:
                names = [f"{model_prefix}_20222023_20242025.pkl"]
            else:
                s_prev = season_prev(s_cur)
                s_next = season_next(s_cur)
                s_prev2 = season_prev(s_prev)   # s-2
                s_next2 = season_next(s_next)   # s+2
                # Derive filenames; number of windows configurable via env XG_WINDOWS (default 1 for perf)
                num_windows = 1
                try:
                    num_windows = max(1, min(3, int(os.getenv('XG_WINDOWS', '1'))))
                except Exception:
                    num_windows = 1
                all_names = [
                    f"{model_prefix}_{s_prev2}_{s_cur}.pkl",    # window 1: s-2..s
                    f"{model_prefix}_{s_prev}_{s_next}.pkl",    # window 2: s-1..s+1
                    f"{model_prefix}_{s_cur}_{s_next2}.pkl",    # window 3: s..s+2
                ]
                # Choose middle window first as most centered; try all, use first N that load
                order = [1, 0, 2]
                names = [all_names[i] for i in order]
            # Try all candidates, keep the first num_windows that actually load
            models = []
            for n in names:
                m = load_model_file(n)
                if m is not None:
                    models.append(m)
                    if len(models) >= num_windows:
                        break
            if not models:
                return None
            preds = []
            for m in models:
                try:
                    vec, cols = _vectorize_row_for_model(row_obj, m)
                    if vec is None:
                        # Fallback: try tiny pandas DF for this single row (still low-memory)
                        try:
                            import pandas as _pd2  # local import fallback
                            _d = {c: [str(row_obj.get(c) if row_obj.get(c) is not None else 'missing')] for c in base_feature_cols}
                            df1 = _pd2.DataFrame(_d)
                            df1 = _pd2.get_dummies(df1).astype(float)
                            if hasattr(m, 'feature_names_in_'):
                                cols_needed = list(getattr(m, 'feature_names_in_'))
                                df1 = df1.reindex(columns=cols_needed, fill_value=0.0)
                            p = m.predict_proba(df1)[:, 1]
                            preds.append(float(p[0]))
                            continue
                        except Exception:
                            continue
                    else:
                        import numpy as _np2  # local import
                        x_arr = _np2.asarray([vec], dtype=float)
                        p = m.predict_proba(x_arr)[:, 1]
                    preds.append(float(p[0]))
                except Exception:
                    continue
            if not preds:
                return None
            return float(sum(preds) / len(preds))

        # Helper for ENA fenwick attempts
        def compute_empty_net_fenwick(sd: Optional[float], sa: Optional[float]) -> Optional[float]:
            if sd is None or sa is None:
                return None
            try:
                val = 1.0 / (1.0 + math.exp(0.013609495*float(sd) + 0.023174225*abs(float(sa)) - 1.97392131))
                return float(val)
            except Exception:
                return None

        # Compute xG with batched predictions by (family, window) to reduce Python overhead
        # 1) Handle ENA upfront
        for row in mapped:
            if row.get('StrengthState') == 'ENA':
                if row.get('Shot') == 1:
                    row['xG_S'] = 1.0
                if row.get('Fenwick') == 1:
                    val_en = compute_empty_net_fenwick(row.get('ShotDistance'), row.get('ShotAngle'))
                    if val_en is not None:
                        row['xG_F'] = round(val_en, 6)
                        row['xG_F2'] = round(val_en, 6)
        # 2) Group remaining rows by family
        families = {
            'xgbs': [i for i, r in enumerate(mapped) if r.get('Shot') == 1 and r.get('StrengthState') != 'ENA'],
            'xgb':  [i for i, r in enumerate(mapped) if r.get('Fenwick') == 1 and r.get('StrengthState') != 'ENA'],
            'xgb2': [i for i, r in enumerate(mapped) if r.get('Fenwick') == 1 and r.get('StrengthState') != 'ENA'],
        }

        def window_filenames_for_season(s_cur: int, prefix: str) -> List[str]:
            s_prev = season_prev(s_cur)
            s_next = season_next(s_cur)
            s_prev2 = season_prev(s_prev)
            s_next2 = season_next(s_next)
            num_windows = 1
            try:
                num_windows = max(1, min(3, int(os.getenv('XG_WINDOWS', '1'))))
            except Exception:
                num_windows = 1
            all_names = [
                f"{prefix}_{s_prev2}_{s_cur}.pkl",
                f"{prefix}_{s_prev}_{s_next}.pkl",
                f"{prefix}_{s_cur}_{s_next2}.pkl",
            ]
            # Return all candidates in preferred order; caller loads first N that exist
            order = [1, 0, 2]
            return [all_names[i] for i in order]

        for family, idxs in families.items():
            if not idxs:
                continue
            # Group by season to resolve window filenames once per season
            by_season: Dict[int, List[int]] = {}
            for i in idxs:
                s = mapped[i].get('Season')
                if s is None:
                    continue
                by_season.setdefault(int(s), []).append(i)
            for s_cur, row_idx in by_season.items():
                if s_cur == 20252026:
                    names = [f"{family}_20222023_20242025.pkl"]
                else:
                    names = window_filenames_for_season(s_cur, family)
                # Try all candidates, keep first num_windows that actually load
                num_windows_batch = 1
                try:
                    num_windows_batch = max(1, min(3, int(os.getenv('XG_WINDOWS', '1'))))
                except Exception:
                    num_windows_batch = 1
                models = []
                for n in names:
                    m = load_model_file(n)
                    if m is not None:
                        models.append(m)
                        if len(models) >= num_windows_batch:
                            break
                if not models:
                    continue
                # Precompute required columns per model
                model_cols: List[Tuple[Any, Optional[List[str]]]] = []
                for m in models:
                    cols = _required_columns_for_model(m)
                    model_cols.append((m, cols))
                # Vectorize all rows once per model and predict
                preds_accum = [[] for _ in row_idx]
                for (m, cols) in model_cols:
                    if cols is None:
                        # fallback to per-row tiny pandas
                        try:
                            import pandas as _pd2
                            # Build rows into DF strings
                            data_rows = []
                            for i in row_idx:
                                r = mapped[i]
                                data_rows.append({c: str(r.get(c) if r.get(c) is not None else 'missing') for c in base_feature_cols})
                            df = _pd2.DataFrame(data_rows)
                            df = _pd2.get_dummies(df).astype(float)
                            if hasattr(m, 'feature_names_in_'):
                                cols_needed = list(getattr(m, 'feature_names_in_'))
                                df = df.reindex(columns=cols_needed, fill_value=0.0)
                            p = m.predict_proba(df)[:, 1]
                            for j, val in enumerate(p):
                                preds_accum[j].append(float(val))
                            continue
                        except Exception:
                            continue
                    # Fast vectorization using known columns
                    import numpy as _np2
                    mat = _np2.zeros((len(row_idx), len(cols)), dtype=float)
                    for rpos, i in enumerate(row_idx):
                        r = mapped[i]
                        # Precompute string values
                        vals = {c: ('missing' if r.get(c) is None else str(r.get(c))) for c in base_feature_cols}
                        for cix, cname in enumerate(cols):
                            if '_' not in cname:
                                continue
                            base, suffix = cname.split('_', 1)
                            if vals.get(base) == suffix:
                                mat[rpos, cix] = 1.0
                    p = m.predict_proba(mat)[:, 1]
                    for j, val in enumerate(p):
                        preds_accum[j].append(float(val))
                # Average predictions across windows and write back
                for j, i in enumerate(row_idx):
                    if not preds_accum[j]:
                        continue
                    avgp = float(sum(preds_accum[j]) / len(preds_accum[j]))
                    if family == 'xgbs':
                        mapped[i]['xG_S'] = round(avgp, 6)
                    elif family == 'xgb':
                        mapped[i]['xG_F'] = round(avgp, 6)
                    elif family == 'xgb2':
                        mapped[i]['xG_F2'] = round(avgp, 6)
    except Exception:
        # Fail-safe: don't block PBP if models or pandas are unavailable
        pass

    # Sanitize JSON output to avoid NaN/Inf and numpy types
    def _safe_val(v: Any):
        try:
            import numpy as _np  # type: ignore
            if isinstance(v, (_np.generic,)):
                v = v.item()
        except Exception:
            pass
        if isinstance(v, float):
            try:
                if not math.isfinite(v):
                    return None
            except Exception:
                return None
        return v

    def _sanitize_row(r: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in r.items():
            out[k] = _safe_val(v)
        return out

    mapped_sanitized = [_sanitize_row(r) for r in mapped]

    out_obj = {
        'gameId': data.get('id'),
        'plays': mapped_sanitized,
        'gameState': game_state,
    }
    try:
        _cache_set_multi_bounded(_PBP_CACHE, int(game_id), out_obj, ttl_s=std_ttl, max_items=max_items)
    except Exception:
        pass
    # Write to disk cache with metadata
    try:
        import json, time
        js = dict(out_obj)
        js['_cachedAt'] = time.time()
        with open(disk_path, 'w', encoding='utf-8') as f:
            json.dump(js, f)
    except Exception:
        pass
    resp = jsonify(out_obj)
    if force:
        try:
            resp.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
    return resp


@main_bp.route('/api/game/<int:game_id>/shifts')
def api_game_shifts(game_id: int):
    """Scrape HTML TV/TH reports for shifts and map players to playerIds via boxscore.

    Output rows: PlayerID, Name, Team, Period, Start (sec), End (sec), Duration (End-Start)
    """
    try:
        force = str(request.args.get('force', '')).lower() in ('1', 'true', 'yes', 'y', 'force')
    except Exception:
        force = False

    # Optional controls for scripts/backfills:
    # - cache=0 / nocache=1: bypass in-memory cache and do not populate it
    # - disk=0 / nodisk=1: bypass on-disk cache and do not write it
    try:
        no_cache = str(request.args.get('nocache', '')).lower() in ('1', 'true', 'yes', 'y') or str(request.args.get('cache', '')).strip() in ('0', 'false', 'False')
    except Exception:
        no_cache = False
    try:
        no_disk = str(request.args.get('nodisk', '')).lower() in ('1', 'true', 'yes', 'y') or str(request.args.get('disk', '')).strip() in ('0', 'false', 'False')
    except Exception:
        no_disk = False
    gid = str(game_id)
    if len(gid) < 10:
        return jsonify({'error': 'Invalid gameId'}), 400
    try:
        start_year = int(gid[:4])
    except Exception:
        return jsonify({'error': 'Invalid gameId'}), 400
    season_dir = f"{start_year}{start_year+1}"
    suffix = gid[4:]

    urls = {
        'away': f"https://www.nhl.com/scores/htmlreports/{season_dir}/TV{suffix}.HTM",
        'home': f"https://www.nhl.com/scores/htmlreports/{season_dir}/TH{suffix}.HTM",
    }

    # Cache TTLs
    live_ttl = 5
    std_ttl = int(os.getenv('SHIFTS_CACHE_TTL_SECONDS', '600'))
    try:
        max_items = max(1, int(os.getenv('SHIFTS_CACHE_MAX_ITEMS', '24') or '24'))
    except Exception:
        max_items = 24
    disk_path = _disk_cache_path_shifts(int(game_id))

    # Disk cache first (contains gameState so we can pick live vs std TTL without fetching boxscore)
    if (not force) and (not no_disk):
        try:
            if os.path.exists(disk_path):
                import json
                js = None
                with open(disk_path, 'r', encoding='utf-8') as f:
                    js = json.load(f)
                ts = float((js or {}).get('_cachedAt', 0.0) or 0.0)
                gstate = str((js or {}).get('gameState') or '').upper()
                ttl = live_ttl if gstate in ('LIVE', 'SCHEDULED', 'PREVIEW', 'INPROGRESS') else std_ttl
                if ts and (time.time() - ts) < ttl:
                    return jsonify({k: v for k, v in (js or {}).items() if not str(k).startswith('_')})
        except Exception:
            pass

    # In-memory cache next (also includes gameState)
    if (not force) and (not no_cache):
        try:
            _cache_prune_ttl_and_size(_SHIFTS_CACHE, ttl_s=std_ttl, max_items=max_items)
            cached = _SHIFTS_CACHE.get(int(game_id))
            if cached:
                ts = float((cached[0] or 0.0))
                payload = cached[1]
                gstate = str((payload or {}).get('gameState') or '').upper()
                ttl = live_ttl if gstate in ('LIVE', 'SCHEDULED', 'PREVIEW', 'INPROGRESS') else std_ttl
                if ts and (time.time() - ts) < ttl:
                    return jsonify(payload)
        except Exception:
            pass

    def fetch_html(url: str) -> Optional[str]:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            }
            r = requests.get(url, timeout=25, headers=headers)
            if r.status_code == 200:
                text = r.text
                if text and len(text) > 500:
                    return text
        except Exception:
            return None
        return None

    # Fetch boxscore to map to player IDs
    try:
        r = requests.get(f'https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore', timeout=20)
        if r.status_code != 200:
            return jsonify({'error': 'Failed to fetch boxscore'}), 502
        box = r.json()
    except Exception:
        return jsonify({'error': 'Failed to fetch boxscore'}), 502

    pages = {side: fetch_html(u) for side, u in urls.items()}

    def unify_roster(team_stats: Dict) -> List[Dict]:
        res: List[Dict] = []
        for grp in ('forwards', 'defense', 'goalies'):
            for p in (team_stats.get(grp) or []):
                nm = p.get('name')
                if isinstance(nm, dict):
                    nm = nm.get('default')
                # Position: prefer explicit position/positionCode, else infer from group.
                raw_pos = (p.get('position') or p.get('positionCode') or '').strip().upper()
                pos = None
                if raw_pos:
                    code = raw_pos[0]
                    pos = 'F' if code in ('C', 'L', 'R') else code
                else:
                    pos = 'F' if grp == 'forwards' else ('D' if grp == 'defense' else 'G')
                res.append({
                    'playerId': p.get('playerId'),
                    'name': nm,
                    'sweaterNumber': str(p.get('sweaterNumber') or p.get('sweater') or p.get('jersey') or '').strip(),
                    'pos': pos,
                })
        return res

    pbg = box.get('playerByGameStats') or {}
    roster_home = unify_roster(pbg.get('homeTeam') or {})
    roster_away = unify_roster(pbg.get('awayTeam') or {})
    # Canonical display name by playerId from lineups/boxscore
    name_by_id: Dict[int, str] = {}
    try:
        for p in roster_home + roster_away:
            pid = p.get('playerId')
            nm = p.get('name')
            if isinstance(pid, int) and nm:
                name_by_id[pid] = str(nm)
    except Exception:
        name_by_id = {}

    def canonical_name_for(pid: Optional[int], fallback: Optional[str]) -> Optional[str]:
        try:
            if isinstance(pid, int) and pid in name_by_id:
                return name_by_id[pid]
        except Exception:
            pass
        return fallback

    def _strip_diacritics(text: str) -> str:
        try:
            import unicodedata as _ud
            nfkd = _ud.normalize('NFKD', text)
            return ''.join([c for c in nfkd if not _ud.combining(c)])
        except Exception:
            return text

    def _normalize_jersey(s: Optional[str]) -> str:
        if not s:
            return ''
        # Keep only digits; drop leading zeros for stable compare
        digits = ''.join(ch for ch in str(s) if ch.isdigit())
        return str(int(digits)) if digits.isdigit() else ''

    def _strip_parentheticals_local(s: Optional[str]) -> str:
        if not s:
            return ''
        try:
            return re.sub(r"\s*\([^)]*\)", '', s).strip()
        except Exception:
            return s or ''

    def norm_name(s: Optional[str]) -> str:
        if not s:
            return ''
        t = s.replace('\xa0', ' ').replace('\u00a0', ' ').strip()
        t = _strip_parentheticals_local(t)
        if ',' in t:
            parts = [x.strip() for x in t.split(',', 1)]
            if len(parts) == 2:
                t = parts[1] + ' ' + parts[0]
        t = t.replace('.', ' ').replace("'", '').replace('-', ' ')
        t = ' '.join(t.split())
        t = _strip_diacritics(t)
        return t.lower()

    def last_token_norm(name: Optional[str]) -> str:
        """Return a normalized last-name token: diacritics removed, suffixes stripped.
        E.g., "McDavid Jr." -> "mcdavid"; "Smith III" -> "smith".
        """
        base = norm_name(name)
        if not base:
            return ''
        toks = base.split(' ')
        if not toks:
            return ''
        # Strip common suffixes from the tail
        suffixes = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}
        while toks and toks[-1].strip('.').lower() in suffixes:
            toks.pop()
        return toks[-1] if toks else ''

    def build_indices(roster: List[Dict]):
        by_num: Dict[str, Dict] = {}
        by_name: Dict[str, Dict] = {}
        by_last: Dict[str, List[Dict]] = {}
        for p in roster:
            num = _normalize_jersey(p.get('sweaterNumber') or '')
            if num:
                by_num[num] = p
            nm = norm_name(p.get('name'))
            if nm:
                by_name[nm] = p
                last = last_token_norm(nm)
                by_last.setdefault(last, []).append(p)
        return by_num, by_name, by_last

    idx_home = build_indices(roster_home)
    idx_away = build_indices(roster_away)

    def to_seconds(ts: Optional[str]) -> Optional[int]:
        if not ts:
            return None
        ts = ts.strip()
        if '/' in ts:
            ts = ts.split('/', 1)[0].strip()
        m = re.match(r'^(\d{1,2}):(\d{2})$', ts)
        if not m:
            return None
        return int(m.group(1)) * 60 + int(m.group(2))

    def parse_period_value(p: Optional[str]) -> Optional[int]:
        """Map period cell text to integer period.
        - Numeric strings -> int
        - 'OT' -> 4 (overtime starts at 3600s)
        - 'SO' or other text -> None (ignored for shifts)
        """
        if p is None:
            return None
        s = p.strip().upper()
        if not s:
            return None
        if s == 'OT':
            return 4
        if s == 'SO':
            return None
        if s.isdigit():
            try:
                return int(s)
            except Exception:
                return None
        return None

    def proper_name(last_upper: str, first_upper: str) -> str:
        # Use Unicode-aware title-casing while preserving hyphens and apostrophes
        def fix(part: str) -> str:
            part = (part or '').strip().replace('\xa0', ' ').replace('\u00a0', ' ')
            # Split on spaces, then title-case each token; keep hyphens/apostrophes intact
            tokens = []
            for tok in part.split():
                subtoks = re.split(r'([-\'])', tok)
                subtoks = [st.title() if st.isalpha() else st for st in subtoks]
                tokens.append(''.join(subtoks))
            return ' '.join(tokens)
        return f"{fix(first_upper)} {fix(last_upper)}".strip()

    def parse_shifts_from_html(html: str, side: str, idx, team_abbrev: str) -> List[Dict]:
        by_num, by_name, by_last = idx
        out: List[Dict] = []
        if not html:
            return out
        soup_results: List[Dict] = []
        if BeautifulSoup is not None:
            soup = BeautifulSoup(html, 'html.parser')

            # Power Query–style: target content table and iterate rows
            def find_content_table(sp):
                try:
                    trs = sp.find_all('tr')
                    if len(trs) >= 4:
                        td = trs[3].find_all('td')
                        if td:
                            tbl = td[0].find('table')
                            if tbl:
                                return tbl
                except Exception:
                    pass
                # Fallback: heuristic scanning
                for tbl in sp.find_all('table'):
                    rows = tbl.find_all('tr')[:15]
                    for tr in rows:
                        texts = [c.get_text(' ', strip=True).lower() for c in tr.find_all(['th', 'td'])]
                        if not texts:
                            continue
                        if any('shift' in t for t in texts) and (any(t == 'per' or 'period' in t for t in texts) or any(t.startswith('per') for t in texts)):
                            return tbl
                    if tbl.find('td', attrs={'colspan': True}):
                        dense = False
                        for tr in rows:
                            tds = [td for td in tr.find_all('td') if not td.has_attr('colspan') and not td.has_attr('rowspan')]
                            if len(tds) >= 6:
                                dense = True
                                break
                        if dense:
                            return tbl
                return None

            content_tbl = find_content_table(soup)
            if content_tbl is not None:
                current_name = None
                current_jersey = None
                current_pid = None
                current_pos = None
                for tr in content_tbl.find_all('tr'):
                    tds_all = tr.find_all('td')
                    if not tds_all:
                        continue
                    if len(tds_all) == 1 and tds_all[0].has_attr('colspan'):
                        txt = tds_all[0].get_text(' ', strip=True)
                        # Allow parenthetical nicknames (e.g., JOHN (JACK)) by stripping them first
                        txt2 = _strip_parentheticals_local(txt)
                        # Support accented Latin letters (e.g., É, è) in names
                        m1 = re.match(r'^(\d{1,2})\s+([A-ZÀ-ÖØ-Þ .\'-]+),\s*([A-ZÀ-ÖØ-Þ .\'-]+)$', txt2)
                        m2 = re.match(r'^(\d{1,2})\s+([A-Za-zÀ-ÖØ-öø-ÿ .\'-]+)$', txt2)
                        if m1:
                            current_jersey = m1.group(1)
                            last_u = m1.group(2)
                            first_u = _strip_parentheticals_local(m1.group(3))
                            current_name = proper_name(last_u, first_u)
                        elif m2:
                            current_jersey = m2.group(1)
                            name_plain = _strip_parentheticals_local(m2.group(2))
                            parts = name_plain.strip().split()
                            current_name = ' '.join(p.capitalize() for p in parts)
                        else:
                            current_name = None
                            current_jersey = None
                        # Resolve PID on header change
                        current_pid = None
                        current_pos = None
                        if current_jersey:
                            p = by_num.get(current_jersey)
                            if p:
                                current_pid = p.get('playerId')
                                current_pos = p.get('pos')
                        if not current_pid and current_name:
                            p = by_name.get(norm_name(current_name))
                            if p:
                                current_pid = p.get('playerId')
                                current_pos = p.get('pos')
                        if not current_pid and current_name:
                            last_tok = last_token_norm(current_name)
                            cands = by_last.get(last_tok, [])
                            if cands:
                                if len(cands) == 1:
                                    current_pid = cands[0].get('playerId')
                                    current_pos = cands[0].get('pos')
                                else:
                                    for cand in cands:
                                        if _normalize_jersey(cand.get('sweaterNumber')) == _normalize_jersey(current_jersey):
                                            current_pid = cand.get('playerId')
                                            current_pos = cand.get('pos')
                                            break
                        continue

                    # Data rows: ignore colspan/rowspan cells
                    tds = [td for td in tds_all if not td.has_attr('colspan') and not td.has_attr('rowspan')]
                    if len(tds) < 4:
                        continue
                    ctext = [td.get_text(' ', strip=True) for td in tds[:6]]
                    shift_no = ctext[0].strip()
                    per_txt = ctext[1].strip()
                    start_txt = ctext[2].strip()
                    end_txt = ctext[3].strip()
                    per_val = parse_period_value(per_txt)
                    if not (shift_no.isdigit() and per_val is not None):
                        continue
                    start_sec = to_seconds(start_txt)
                    end_sec = to_seconds(end_txt)
                    if start_sec is None or end_sec is None:
                        continue
                    if current_pid is None and not current_name:
                        # If we couldn't resolve the current player header, don't emit anonymous shifts.
                        continue
                    # Prefer canonical name from lineups if we have a playerId
                    name_out = canonical_name_for(current_pid, current_name)
                    out.append({
                        'PlayerID': current_pid,
                        'Name': name_out,
                        'Position': current_pos,
                        'Team': team_abbrev or ('Away' if side == 'away' else 'Home'),
                        'Period': int(per_val),
                        'Start': start_sec,
                        'End': end_sec,
                        'Duration': end_sec - start_sec,
                    })
                if out:
                    return out

            # Player-first scan: find header texts and parse the next table
            # Include Latin-1 accented ranges in character classes
            pat_comma = re.compile(r'^(\s*)(\d{1,2})\s+([A-Za-zÀ-ÖØ-öø-ÿ .\'-]+),\s*([A-Za-zÀ-ÖØ-öø-ÿ .\'-]+)(\s*)$')
            pat_plain = re.compile(r'^(\s*)(\d{1,2})\s+([A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ .\'-]+)(\s*)$')
            header_nodes = []
            for node in soup.find_all(string=True):
                txt = (node or '').replace('\xa0', ' ').strip()
                if not txt:
                    continue
                txt2 = _strip_parentheticals_local(txt)
                if pat_comma.match(txt2) or pat_plain.match(txt2):
                    header_nodes.append(node)

            for node in header_nodes:
                raw = (node or '').replace('\xa0', ' ').strip()
                raw2 = _strip_parentheticals_local(raw)
                m1 = pat_comma.match(raw2)
                m2 = pat_plain.match(raw2) if not m1 else None
                if m1:
                    jersey = m1.group(2)
                    last_u = m1.group(3)
                    first_u = _strip_parentheticals_local(m1.group(4))
                    disp_name = proper_name(last_u, first_u)
                    last_for_idx = last_token_norm(last_u)
                elif m2:
                    jersey = m2.group(2)
                    name_plain = _strip_parentheticals_local(m2.group(3))
                    parts = name_plain.strip().split()
                    disp_name = ' '.join(p.capitalize() for p in parts)
                    last_for_idx = last_token_norm(parts[-1]) if parts else ''
                else:
                    continue

                tbl = node.find_parent().find_next('table') if node else None
                if not tbl:
                    continue
                trs = tbl.find_all('tr')
                if not trs:
                    continue
                header_row_idx = None
                i_shift = i_per = i_start = i_end = -1

                def compute_indexes(cells_text: List[str]):
                    nonlocal i_shift, i_per, i_start, i_end
                    hlow = [h.lower() for h in cells_text]
                    def idx_of(parts: List[str]) -> int:
                        for i, h in enumerate(hlow):
                            if all(p in h for p in parts):
                                return i
                        return -1
                    i_shift = idx_of(['shift'])
                    i_per = idx_of(['per']) if idx_of(['per']) >= 0 else idx_of(['period'])
                    i_start = idx_of(['start'])
                    i_end = idx_of(['end'])

                for ridx, tr in enumerate(trs[:6]):
                    cells = [c.get_text(' ', strip=True) for c in tr.find_all(['th', 'td'])]
                    if not cells:
                        continue
                    compute_indexes(cells)
                    if min(i_shift, i_per, i_start, i_end) >= 0:
                        header_row_idx = ridx
                        break
                if header_row_idx is None:
                    continue

                # Resolve PlayerID
                pid = None
                pos_val = None
                p = by_num.get(jersey)
                if p:
                    pid = p.get('playerId')
                    pos_val = p.get('pos')
                if not pid:
                    p = by_name.get(norm_name(disp_name))
                    if p:
                        pid = p.get('playerId')
                        pos_val = p.get('pos')
                if not pid and last_for_idx:
                    cands = by_last.get(last_for_idx, [])
                    if cands:
                        if len(cands) == 1:
                            pid = cands[0].get('playerId')
                            pos_val = cands[0].get('pos')
                        else:
                            for cand in cands:
                                if _normalize_jersey(cand.get('sweaterNumber')) == _normalize_jersey(jersey):
                                    pid = cand.get('playerId')
                                    pos_val = cand.get('pos')
                                    break

                for tr in trs[header_row_idx + 1:]:
                    tds = [td.get_text(' ', strip=True) for td in tr.find_all('td')]
                    if len(tds) <= max(i_shift, i_per, i_start, i_end):
                        continue
                    per_val = parse_period_value(tds[i_per].strip())
                    if not (tds[i_shift].strip().isdigit() and per_val is not None):
                        continue
                    per = int(per_val)
                    start_sec = to_seconds(tds[i_start])
                    end_sec = to_seconds(tds[i_end])
                    if start_sec is None or end_sec is None:
                        continue
                    name_out2 = canonical_name_for(pid, disp_name)
                    if pid is None and not name_out2:
                        continue
                    soup_results.append({
                        'PlayerID': pid,
                        'Name': name_out2,
                        'Position': pos_val,
                        'Team': team_abbrev or ('Away' if side == 'away' else 'Home'),
                        'Period': per,
                        'Start': start_sec,
                        'End': end_sec,
                        'Duration': end_sec - start_sec,
                    })
        if soup_results:
            return soup_results

        # Regex fallback (last resort)
        def strip_tags(s: str) -> str:
            s = re.sub(r'<[^>]+>', ' ', s)
            s = re.sub(r'\s+', ' ', s).strip()
            return s

        section_iter = re.finditer(r'<td[^>]*colspan=\"?\d+\"?[^>]*>\s*(.*?)\s*</td>', html, re.I | re.S)
        positions = []
        for m in section_iter:
            positions.append((m.start(), m.end(), m.group(1)))
        positions.append((len(html), len(html), ''))  # sentinel
        for i in range(len(positions) - 1):
            start, end, header_html = positions[i]
            next_start = positions[i + 1][0]
            header_text = strip_tags(header_html)
            header_text2 = _strip_parentheticals_local(header_text)
            jersey = None
            disp_name = None
            last_for_idx = None
            m1 = re.match(r'^(\d{1,2})\s+([A-ZÀ-ÖØ-Þ .\'-]+),\s*([A-ZÀ-ÖØ-Þ .\'-]+)$', header_text2)
            m2 = re.match(r'^(\d{1,2})\s+([A-Za-zÀ-ÖØ-öø-ÿ .\'-]+)$', header_text2)
            if m1:
                jersey = m1.group(1)
                last_u = m1.group(2)
                first_u = _strip_parentheticals_local(m1.group(3))
                disp_name = proper_name(last_u, first_u)
                last_for_idx = last_token_norm(last_u)
            elif m2:
                jersey = m2.group(1)
                name_plain = _strip_parentheticals_local(m2.group(2))
                parts = name_plain.strip().split()
                disp_name = ' '.join(p.capitalize() for p in parts)
                last_for_idx = last_token_norm(parts[-1]) if parts else ''
            else:
                continue

            # Resolve PlayerID
            pid = None
            pos_val = None
            if jersey:
                p = by_num.get(_normalize_jersey(jersey))
                if p:
                    pid = p.get('playerId')
                    pos_val = p.get('pos')
            if not pid and disp_name:
                p = by_name.get(norm_name(disp_name))
                if p:
                    pid = p.get('playerId')
                    pos_val = p.get('pos')
            if not pid and last_for_idx:
                cands = by_last.get(last_for_idx, [])
                if cands:
                    if len(cands) == 1:
                        pid = cands[0].get('playerId')
                        pos_val = cands[0].get('pos')
                    else:
                        for cand in cands:
                            if _normalize_jersey(cand.get('sweaterNumber')) == _normalize_jersey(jersey):
                                pid = cand.get('playerId')
                                pos_val = cand.get('pos')
                                break

            section_html = html[end:next_start]
            row_re = re.compile(r'<tr[^>]*>\s*(.*?)\s*</tr>', re.I | re.S)
            cell_re = re.compile(r'<t[dh][^>]*>\s*(.*?)\s*</t[dh]>', re.I | re.S)
            rows = row_re.findall(section_html)
            for row_html in rows:
                if re.search(r'<td[^>]*colspan=', row_html, re.I):
                    continue
                cells_html = cell_re.findall(row_html)
                cells = [strip_tags(c) for c in cells_html]
                if len(cells) < 4:
                    continue
                shift_no = cells[0].strip()
                per_txt = cells[1].strip()
                start_txt = cells[2].strip()
                end_txt = cells[3].strip()
                per_val = parse_period_value(per_txt)
                if not (shift_no.isdigit() and per_val is not None):
                    continue
                start_sec = to_seconds(start_txt)
                end_sec = to_seconds(end_txt)
                if start_sec is None or end_sec is None:
                    continue
                name_out3 = canonical_name_for(pid, disp_name)
                if pid is None and not name_out3:
                    continue
                out.append({
                    'PlayerID': pid,
                    'Name': name_out3,
                    'Position': pos_val,
                    'Team': team_abbrev or ('Away' if side == 'away' else 'Home'),
                    'Period': int(per_val),
                    'Start': start_sec,
                    'End': end_sec,
                    'Duration': end_sec - start_sec,
                })
        return out

    away_abbrev = (box.get('awayTeam') or {}).get('abbrev') or 'AWY'
    home_abbrev = (box.get('homeTeam') or {}).get('abbrev') or 'HME'
    shifts_out: List[Dict] = []
    shifts_out += parse_shifts_from_html(pages.get('away') or '', 'away', idx_away, away_abbrev)
    shifts_out += parse_shifts_from_html(pages.get('home') or '', 'home', idx_home, home_abbrev)

    # Transform Start/End to game time (seconds since game start) and build global shift slices
    # Period offset = (Period-1) * 1200 seconds (20-minute periods)
    entries: List[Dict] = []
    boundaries: set[int] = set()
    max_end = 0
    for row in shifts_out:
        try:
            per = int(row.get('Period') or 1)
        except Exception:
            per = 1
        try:
            st = int(row.get('Start') or 0)
            et = int(row.get('End') or 0)
        except Exception:
            continue
        base = (per - 1) * 1200
        gs = base + max(0, st)
        ge = base + max(0, et)
        if ge <= gs:
            continue
        boundaries.add(gs)
        boundaries.add(ge)
        if ge > max_end:
            max_end = ge
        entries.append({
            'gs': gs,
            'ge': ge,
            'PlayerID': row.get('PlayerID'),
            'Name': row.get('Name'),
            'Position': row.get('Position'),
            'Team': row.get('Team'),
        })

    if not entries:
        return jsonify({
            'gameId': game_id,
            'seasonDir': season_dir,
            'suffix': suffix,
            'source': urls,
            'shifts': [],
        })

    # Create sorted unique start times; ensure max_end is included as the last boundary
    times = sorted(t for t in boundaries)
    if not times or times[-1] != max_end:
        times.append(max_end)

    # Build global shift slices and split players into these slices
    split_rows: List[Dict] = []
    for i in range(len(times) - 1):
        s = times[i]
        e = times[i + 1]
        if e <= s:
            continue
        shift_index = int(game_id) * 10000 + (i + 1)

        # Determine active players in [s, e)
        active: List[Dict] = [rec for rec in entries if rec['gs'] <= s < rec['ge']]
        if not active:
            continue

        # Compute unique on-ice skater/goalie counts per team for this slice.
        # The HTML shift reports can contain duplicate or overlapping rows; counting rows inflates
        # skater counts (e.g., 10v8). Use unique PlayerID (fallback to Name) per side.
        team_players: Dict[str, Dict[str, set]] = {}
        for rec in active:
            team = str(rec.get('Team') or '')
            pos = str(rec.get('Position') or '').upper()
            pid = rec.get('PlayerID')
            name = str(rec.get('Name') or '').strip()
            key = pid if isinstance(pid, int) else (name if name else None)
            if key is None:
                continue
            tp = team_players.setdefault(team, {'G': set(), 'S': set()})
            if pos == 'G':
                tp['G'].add(key)
            else:
                tp['S'].add(key)

        team_counts_raw: Dict[str, Dict[str, int]] = {}
        team_counts_clamped: Dict[str, Dict[str, int]] = {}
        for t, tp in team_players.items():
            try:
                g_raw = int(len(tp.get('G') or set()))
                s_raw = int(len(tp.get('S') or set()))
            except Exception:
                g_raw = 0
                s_raw = 0

            # Clamp to realistic maxima to avoid noise in downstream calculations.
            g = min(max(g_raw, 0), 1)
            sk = min(max(s_raw, 0), 6)

            team_counts_raw[t] = {'G': g_raw, 'S': s_raw}
            team_counts_clamped[t] = {'G': g, 'S': sk}

        def _normalize_strength_state(*, my_s: int, their_s: int, my_g: int, their_g: int) -> str:
            observed_goalies = (my_g + their_g) > 0
            if observed_goalies and my_g == 0 and their_g >= 1:
                return 'ENF'
            if observed_goalies and their_g == 0 and my_g >= 1:
                return 'ENA'

            ms = int(my_s or 0)
            ts = int(their_s or 0)
            ms = max(0, min(ms, 6))
            ts = max(0, min(ts, 6))

            # Preserve empty-net states (6vX / Xv6) explicitly. The expected set includes
            # 6v5/6v4/6v3/6v2 and their inverses.
            if ms == 6 and ts == 6:
                # Extremely rare (both teams with extra attacker) — snap to even strength.
                return '5v5'
            # Requested bucketing for future shifts calcs:
            # - treat 6v5/5v6 as 5v5 (empty net shouldn't split the state)
            # - treat 6v4 as PP and 4v6 as SH
            if (ms, ts) in {(6, 5), (5, 6)}:
                return '5v5'
            if (ms, ts) == (6, 4):
                return 'PP'
            if (ms, ts) == (4, 6):
                return 'SH'
            if ms == 6 or ts == 6:
                return f'{ms}v{ts}'

            # PP / SH
            if ts == 4 and ms == 5:
                return '5v4'
            if ts == 3 and ms == 5:
                return '5v3'
            if ts == 3 and ms == 4:
                return '4v3'
            if ms == 4 and ts == 5:
                return '4v5'
            if ms == 3 and ts == 5:
                return '3v5'
            if ms == 3 and ts == 4:
                return '3v4'

            # Two-man advantage/disadvantage cases that we want to preserve.
            if ms == 2 and ts in (3, 4, 5):
                return f'2v{ts}'
            if ts == 2 and ms in (3, 4, 5):
                return f'{ms}v2'

            # Even strength
            if ms == 4 and ts == 4:
                return '4v4'
            if ms == 3 and ts == 3:
                return '3v3'
            if ms == 5 and ts == 5:
                return '5v5'

            # Extreme low-count fallbacks.
            if ts == 0 and ms >= 1:
                return '1v0'
            if ms == 0 and ts >= 1:
                return '0v1'

            # If we're missing players due to scraping quirks, snap to closest even-strength.
            m = max(ms, ts)
            if m <= 3:
                return '3v3'
            if m == 4:
                return '4v4'
            return '5v5'

        # Emit rows with StrengthState
        for rec in active:
            team = rec['Team'] or ''
            # Determine opponent team abbrev
            if team == away_abbrev:
                opp = home_abbrev
            elif team == home_abbrev:
                opp = away_abbrev
            elif team.lower() == 'away':
                opp = 'Home'
            elif team.lower() == 'home':
                opp = 'Away'
            else:
                # Fallback: pick any other team in this slice
                opp = next((t for t in team_counts_clamped.keys() if t != team), '')

            my_raw = team_counts_raw.get(team, {'G': 0, 'S': 0})
            their_raw = team_counts_raw.get(opp, {'G': 0, 'S': 0})
            my = team_counts_clamped.get(team, {'G': 0, 'S': 0})
            their = team_counts_clamped.get(opp, {'G': 0, 'S': 0})

            # SeasonStats bucketing: only trust skater counts when both goalies are in.
            # Rules:
            # - both goalies in + both teams have 5+ skaters -> 5v5
            # - both goalies in + opponent has 3/4 skaters -> PP (if we have more skaters)
            # - both goalies in + we have 3/4 skaters -> SH (if opponent has more skaters)
            # - else -> Other
            try:
                my_g = int(my.get('G') or 0)
                their_g = int(their.get('G') or 0)
                # Some shift reports omit goalies entirely; if BOTH sides have 0 goalie rows,
                # treat goalie presence as unknown and assume both goalies are in.
                both_goalies_in = (my_g >= 1 and their_g >= 1) or (my_g == 0 and their_g == 0)
                my_s = int(my.get('S') or 0)
                their_s = int(their.get('S') or 0)
                if both_goalies_in and my_s >= 5 and their_s >= 5:
                    strength_bucket = '5v5'
                elif both_goalies_in and their_s in (3, 4) and my_s > their_s:
                    strength_bucket = 'PP'
                elif both_goalies_in and my_s in (3, 4) and their_s > my_s:
                    strength_bucket = 'SH'
                else:
                    strength_bucket = 'Other'
            except Exception:
                my_g = 0
                their_g = 0
                my_s = 0
                their_s = 0
                strength_bucket = 'Other'

            strength = _normalize_strength_state(my_s=my_s, their_s=their_s, my_g=my_g, their_g=their_g)
            strength_raw = f"{int(my_raw.get('S') or 0)}v{int(their_raw.get('S') or 0)}"

            period_calc = 1 + (s // 1200)
            split_rows.append({
                'ShiftIndex': shift_index,
                'PlayerID': rec['PlayerID'],
                'Name': rec['Name'],
                'Position': rec.get('Position'),
                'Team': rec['Team'],
                'Period': int(period_calc),
                'Start': int(s),
                'End': int(e),
                'Duration': int(e - s),
                'StrengthState': strength,
                'StrengthStateRaw': strength_raw,
                'StrengthStateBucket': strength_bucket,
                'SkatersOnIceFor': int(my_raw.get('S') or 0),
                'SkatersOnIceAgainst': int(their_raw.get('S') or 0),
                'GoaliesOnIceFor': int(my_raw.get('G') or 0),
                'GoaliesOnIceAgainst': int(their_raw.get('G') or 0),
            })

    out = {
        'gameId': game_id,
        'seasonDir': season_dir,
        'suffix': suffix,
        'source': urls,
        'shifts': split_rows,
    }

    try:
        # Include gameState for disk cache TTL check
        game_state = str((box.get('gameState') or box.get('gameStatus') or '')).upper()
        out['gameState'] = game_state
    except Exception:
        pass

    if not no_cache:
        try:
            _cache_set_multi_bounded(_SHIFTS_CACHE, int(game_id), out, ttl_s=std_ttl, max_items=max_items)
        except Exception:
            pass

    # Persist to disk
    if not no_disk:
        try:
            import json
            js = dict(out)
            js['_cachedAt'] = time.time()
            with open(disk_path, 'w', encoding='utf-8') as f:
                json.dump(js, f)
        except Exception:
            pass
    resp = jsonify(out)
    if force:
        try:
            resp.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
    return resp


# ── Shooting / Goaltending aggregate API ────────────────────────

@main_bp.route('/api/skaters/shooting')
def api_skaters_shooting():
    """Aggregate shot events for a team+season from the pbp table.

    Query params:
      team   – 3-letter team code (required)
      season – e.g. 20252026 (required)
      seasonState – regular (default) / playoffs / all
      strengthState – 5v5 (default) / PP / SH / all
      xgModel – xG_F (default) / xG_S / xG_F2

    Returns:
      kpis: {shots, xG, goals, gax, shPct, xShPct, dShPct}
      goalies: [{goalieId, name, shots, xG, goals, gax}, ...]
      events: [{x, y, boxId, goal, xG, goalieId, shotType}, ...]
      zones: {boxId: {shots, xG, goals}, ...}
    """
    team = str(request.args.get('team', '')).strip().upper()
    season = str(request.args.get('season', '')).strip()
    if not team or not season:
        return jsonify({'error': 'team and season required'}), 400
    season_ids = _parse_request_season_ids(season)

    ss = str(request.args.get('seasonState', 'regular')).lower()
    strength = str(request.args.get('strengthState', '5v5')).strip()
    xg_model = str(request.args.get('xgModel', 'xG_F')).strip()

    xg_col_map = {'xG_F': 'xg_f', 'xG_S': 'xg_s', 'xG_F2': 'xg_f2'}
    xg_col = xg_col_map.get(xg_model, 'xg_f')

    player_id = str(request.args.get('player', '')).strip()

    # Optional roster filter: comma-separated player IDs to restrict events
    # to the current roster (so KPIs match Table View totals).
    roster_ids_raw = str(request.args.get('roster', '')).strip()
    roster_ids: Optional[set] = None
    if roster_ids_raw and not player_id:
        try:
            roster_ids = {int(x) for x in roster_ids_raw.split(',') if x.strip()}
        except Exception:
            roster_ids = None

    filters_base: Dict[str, str] = {
        'event_team': f'eq.{team}',
    }
    if player_id:
        filters_base['player1_id'] = f'eq.{player_id}'
    if ss and ss != 'all':
        filters_base['season_state'] = f'eq.{ss}'

    all_events = []
    try:
        cols = f"event_index,x,y,box_id,shot,goal,fenwick,{xg_col},goalie_id,shot_type,player1_id,strength_state,event,highlight_url,period"
        for season_id in season_ids:
            rows = _sb_read('pbp', columns=cols, order='event_index', filters={
                **filters_base,
                'season': f'eq.{int(season_id)}',
                'fenwick': 'eq.1',
            })
            if rows:
                all_events.extend([e for e in rows if int(e.get('period') or 0) != 5])
    except Exception:
        pass

    # Restrict to current roster players when roster filter is provided
    if roster_ids:
        all_events = [e for e in all_events if _safe_int(e.get('player1_id')) in roster_ids]

    # Apply strength state filter
    if strength and strength.lower() != 'all':
        strength_map = {'5v5': '5v5', 'PP': '5v4', 'SH': '4v5'}
        target = strength_map.get(strength, strength)
        if strength == '5v5':
            all_events = [e for e in all_events if str(e.get('strength_state', '')) == '5v5']
        elif strength == 'PP':
            all_events = [e for e in all_events if str(e.get('strength_state', '')) in ('5v4', '5v3', '4v3')]
        elif strength == 'SH':
            all_events = [e for e in all_events if str(e.get('strength_state', '')) in ('4v5', '3v5', '3v4')]
        else:
            all_events = [e for e in all_events if str(e.get('strength_state', '')) == target]

    # Build response
    goalie_agg: Dict[int, Dict[str, Any]] = {}
    zone_agg: Dict[str, Dict[str, float]] = {}
    events_out = []
    total_shots = 0
    total_xg = 0.0
    total_goals = 0

    for e in all_events:
        xg_val = float(e.get(xg_col) or 0.0)
        is_goal = int(e.get('goal') or 0) == 1
        is_shot = int(e.get('shot') or 0) == 1 or is_goal
        gid = e.get('goalie_id')
        box = str(e.get('box_id') or '')

        total_shots += 1
        total_xg += xg_val
        if is_goal:
            total_goals += 1

        # Goalie aggregation
        if gid:
            gid = int(gid)
            if gid not in goalie_agg:
                goalie_agg[gid] = {'goalieId': gid, 'shots': 0, 'xG': 0.0, 'goals': 0}
            goalie_agg[gid]['shots'] += 1
            goalie_agg[gid]['xG'] += xg_val
            if is_goal:
                goalie_agg[gid]['goals'] += 1

        # Zone aggregation
        if box:
            if box not in zone_agg:
                zone_agg[box] = {'shots': 0, 'xG': 0.0, 'goals': 0}
            zone_agg[box]['shots'] += 1
            zone_agg[box]['xG'] += xg_val
            if is_goal:
                zone_agg[box]['goals'] += 1

        events_out.append({
            'x': float(e.get('x') or 0),
            'y': float(e.get('y') or 0),
            'boxId': box,
            'goal': 1 if is_goal else 0,
            'shot': 1 if is_shot else 0,
            'xG': round(xg_val, 4),
            'goalieId': int(gid) if gid else None,
            'shotType': str(e.get('shot_type') or ''),
            'playerId': int(e.get('player1_id')) if e.get('player1_id') else None,
            'highlightUrl': e.get('highlight_url') if is_goal else None,
        })

    # Compute KPIs
    sh_pct = (total_goals / total_shots * 100) if total_shots else 0.0
    x_sh_pct = (total_xg / total_shots * 100) if total_shots else 0.0
    d_sh_pct = sh_pct - x_sh_pct
    gax = total_goals - total_xg

    # Compute GAx for each goalie
    goalies_list = sorted(goalie_agg.values(), key=lambda g: g['goals'] - g['xG'], reverse=True)
    # Resolve goalie names from players table (DB), fall back to roster API
    player_names = _load_player_names_for_seasons(season_ids)
    for g in goalies_list:
        g['gax'] = round(g['goals'] - g['xG'], 2)
        g['xG'] = round(g['xG'], 2)
        g['name'] = player_names.get(g['goalieId'], str(g['goalieId']))

    # Round zone aggregates
    for z in zone_agg.values():
        z['xG'] = round(z['xG'], 2)

    return jsonify({
        'kpis': {
            'shots': total_shots,
            'xG': round(total_xg, 2),
            'goals': total_goals,
            'gax': round(gax, 2),
            'shPct': round(sh_pct, 1),
            'xShPct': round(x_sh_pct, 1),
            'dShPct': round(d_sh_pct, 1),
        },
        'goalies': goalies_list,
        'events': events_out,
        'zones': zone_agg,
    })


@main_bp.route('/api/goalies/goaltending')
def api_goalies_goaltending():
    """Aggregate shots-against for a team+season from the goalie perspective.

    Query params:
      team   – 3-letter team code (required)
      season – e.g. 20252026 (required)
      seasonState – regular (default) / playoffs / all
      strengthState – 5v5 (default) / PP / SH / all
      xgModel – xG_F (default) / xG_S / xG_F2

    Returns:
      kpis: {sa, xGA, ga, gsax, svPct, xSvPct, dSvPct}
      shooters: [{playerId, name, sa, xGA, ga, gsax}, ...]
      events: [{x, y, boxId, goal, xG, playerId, shotType}, ...]
      zones: {boxId: {sa, xGA, ga}, ...}
    """
    team = str(request.args.get('team', '')).strip().upper()
    season = str(request.args.get('season', '')).strip()
    if not team or not season:
        return jsonify({'error': 'team and season required'}), 400
    season_ids = _parse_request_season_ids(season)

    ss = str(request.args.get('seasonState', 'regular')).lower()
    strength = str(request.args.get('strengthState', '5v5')).strip()
    xg_model = str(request.args.get('xgModel', 'xG_F')).strip()

    xg_col_map = {'xG_F': 'xg_f', 'xG_S': 'xg_s', 'xG_F2': 'xg_f2'}
    xg_col = xg_col_map.get(xg_model, 'xg_f')

    goalie_id = str(request.args.get('player', '')).strip()

    # Optional roster filter: comma-separated goalie IDs to restrict events
    # to the current roster (so KPIs match Table View totals).
    roster_ids_raw = str(request.args.get('roster', '')).strip()
    roster_ids: Optional[set] = None
    if roster_ids_raw and not goalie_id:
        try:
            roster_ids = {int(x) for x in roster_ids_raw.split(',') if x.strip()}
        except Exception:
            roster_ids = None

    roster_key = tuple(sorted(int(x) for x in roster_ids)) if roster_ids else ()
    try:
        gt_ttl_s = max(30, int(os.getenv('GOALIES_GOALTENDING_CACHE_TTL_SECONDS', '180') or '180'))
    except Exception:
        gt_ttl_s = 180
    try:
        gt_max_items = max(1, int(os.getenv('GOALIES_GOALTENDING_CACHE_MAX_ITEMS', '128') or '128'))
    except Exception:
        gt_max_items = 128
    gt_cache_key = (
        str(team),
        tuple(_normalize_season_id_list(season_ids)),
        str(ss),
        str(strength),
        str(xg_model),
        str(goalie_id or ''),
        roster_key,
    )
    _cache_prune_ttl_and_size(_GOALIES_GOALTENDING_CACHE, ttl_s=gt_ttl_s, max_items=gt_max_items)
    gt_cached = _cache_get(_GOALIES_GOALTENDING_CACHE, gt_cache_key, gt_ttl_s)
    if gt_cached is not None:
        j = jsonify(gt_cached)
        try:
            j.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
        return j

    filters_base: Dict[str, str] = {}
    if goalie_id:
        filters_base['goalie_id'] = f'eq.{goalie_id}'
    if ss and ss != 'all':
        filters_base['season_state'] = f'eq.{ss}'

    all_events = []
    try:
        cols = f"event_index,x,y,box_id,shot,goal,fenwick,{xg_col},goalie_id,shot_type,player1_id,strength_state,highlight_url,period"
        for season_id in season_ids:
            rows = _sb_read('pbp', columns=cols, order='event_index', filters={
                **filters_base,
                'season': f'eq.{int(season_id)}',
                'opponent': f'eq.{team}',
                'fenwick': 'eq.1',
            })
            if rows:
                all_events.extend([e for e in rows if int(e.get('period') or 0) != 5])
    except Exception:
        pass

    # Restrict to current roster goalies when roster filter is provided
    if roster_ids:
        all_events = [e for e in all_events if _safe_int(e.get('goalie_id')) in roster_ids]

    # Apply strength state filter
    # NOTE: strength_state is from the SHOOTER's (event_team) perspective.
    # For goaltending, PP means the goalie's team has more skaters,
    # so the shooter is shorthanded → filter for 4v5/3v5/3v4.
    # SH means the goalie's team is shorthanded,
    # so the shooter has a power play → filter for 5v4/5v3/4v3.
    if strength and strength.lower() != 'all':
        if strength == '5v5':
            all_events = [e for e in all_events if str(e.get('strength_state', '')) == '5v5']
        elif strength == 'PP':
            # Goalie's team on PP → shooter is shorthanded
            all_events = [e for e in all_events if str(e.get('strength_state', '')) in ('4v5', '3v5', '3v4')]
        elif strength == 'SH':
            # Goalie's team on PK → shooter has power play
            all_events = [e for e in all_events if str(e.get('strength_state', '')) in ('5v4', '5v3', '4v3')]
        else:
            all_events = [e for e in all_events if str(e.get('strength_state', '')) == strength]

    # Build response
    shooter_agg: Dict[int, Dict[str, Any]] = {}
    zone_agg: Dict[str, Dict[str, float]] = {}
    events_out = []
    total_sa = 0
    total_xga = 0.0
    total_ga = 0

    for e in all_events:
        xg_val = float(e.get(xg_col) or 0.0)
        is_goal = int(e.get('goal') or 0) == 1
        pid = e.get('player1_id')
        box = str(e.get('box_id') or '')

        total_sa += 1
        total_xga += xg_val
        if is_goal:
            total_ga += 1

        # Shooter aggregation
        if pid:
            pid = int(pid)
            if pid not in shooter_agg:
                shooter_agg[pid] = {'playerId': pid, 'sa': 0, 'xGA': 0.0, 'ga': 0}
            shooter_agg[pid]['sa'] += 1
            shooter_agg[pid]['xGA'] += xg_val
            if is_goal:
                shooter_agg[pid]['ga'] += 1

        # Zone aggregation
        if box:
            if box not in zone_agg:
                zone_agg[box] = {'sa': 0, 'xGA': 0.0, 'ga': 0}
            zone_agg[box]['sa'] += 1
            zone_agg[box]['xGA'] += xg_val
            if is_goal:
                zone_agg[box]['ga'] += 1

        events_out.append({
            'x': float(e.get('x') or 0),
            'y': float(e.get('y') or 0),
            'boxId': box,
            'goal': 1 if is_goal else 0,
            'shot': 1 if (int(e.get('shot') or 0) == 1 or is_goal) else 0,
            'xG': round(xg_val, 4),
            'playerId': int(pid) if pid else None,
            'goalieId': int(e.get('goalie_id')) if e.get('goalie_id') else None,
            'shotType': str(e.get('shot_type') or ''),
            'highlightUrl': e.get('highlight_url') if is_goal else None,
        })

    # Compute KPIs
    sv_pct = ((total_sa - total_ga) / total_sa * 100) if total_sa else 0.0
    x_sv_pct = ((total_sa - total_xga) / total_sa * 100) if total_sa else 0.0
    d_sv_pct = sv_pct - x_sv_pct
    gsax = total_xga - total_ga  # positive = saved more than expected

    # Compute GSAx for each shooter
    shooters_list = sorted(shooter_agg.values(), key=lambda s: s['xGA'] - s['ga'], reverse=True)
    # Resolve shooter names from players table (DB), fall back to roster API
    player_names = _load_player_names_for_seasons(season_ids)
    for s in shooters_list:
        s['gsax'] = round(s['xGA'] - s['ga'], 2)
        s['xGA'] = round(s['xGA'], 2)
        s['name'] = player_names.get(s['playerId'], str(s['playerId']))

    # Round zone aggregates
    for z in zone_agg.values():
        z['xGA'] = round(z['xGA'], 2)

    payload = {
        'kpis': {
            'sa': total_sa,
            'xGA': round(total_xga, 2),
            'ga': total_ga,
            'gsax': round(gsax, 2),
            'svPct': round(sv_pct, 1),
            'xSvPct': round(x_sv_pct, 1),
            'dSvPct': round(d_sv_pct, 1),
        },
        'shooters': shooters_list,
        'events': events_out,
        'zones': zone_agg,
    }
    _cache_set_multi_bounded(_GOALIES_GOALTENDING_CACHE, gt_cache_key, payload, ttl_s=gt_ttl_s, max_items=gt_max_items)
    j = jsonify(payload)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


