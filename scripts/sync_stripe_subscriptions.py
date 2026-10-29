"""Reconcile Stripe subscriptions into the app's `user_accounts` rows.

The Stripe webhook is the primary writer, but when it fails (or never ran for an
account) paying customers stay on "trial ended". This script is the repair path:
it reads subscriptions straight from Stripe and writes the same account fields
the webhook would.

Dry-run by default — pass --apply to write.

Usage:
    python scripts/sync_stripe_subscriptions.py                 # show the plan
    python scripts/sync_stripe_subscriptions.py --apply         # write it
    python scripts/sync_stripe_subscriptions.py --apply --user <auth_user_id>

Requires STRIPE_SECRET_KEY plus SUPABASE_URL / SUPABASE_SERVICE_KEY.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app import supabase_client  # noqa: E402
from app.routes import (  # noqa: E402
    _build_account_payload,
    _stripe_updates_from_subscription,
)

STRIPE_API = "https://api.stripe.com/v1"

# Preference order when one account has several subscriptions: a live charge
# beats a lapsed one, and the newest period end wins among equals.
_LIVE_STATUSES = ("active", "trialing", "past_due")


def _stripe_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    key = os.getenv("STRIPE_SECRET_KEY", "").strip()
    if not key:
        raise SystemExit("STRIPE_SECRET_KEY is not set")
    resp = requests.get(
        f"{STRIPE_API}/{path}",
        params=params or {},
        headers={"Authorization": f"Bearer {key}"},
        timeout=30,
    )
    if resp.status_code != 200:
        raise SystemExit(f"Stripe {path} failed: {resp.status_code} {resp.text[:200]}")
    return resp.json()


def list_subscriptions() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    starting_after: Optional[str] = None
    while True:
        params: Dict[str, Any] = {"limit": 100, "status": "all"}
        if starting_after:
            params["starting_after"] = starting_after
        page = _stripe_get("subscriptions", params)
        data = page.get("data") or []
        out.extend(data)
        if not page.get("has_more") or not data:
            return out
        starting_after = data[-1]["id"]


def account_index() -> Dict[str, Dict[str, Any]]:
    """Existing user_accounts rows keyed by auth_user_id."""
    rows = supabase_client.list_user_accounts() or []
    return {str(r.get("auth_user_id")): r for r in rows if r.get("auth_user_id")}


def email_to_auth_user_id() -> Dict[str, str]:
    """Lowercased email -> auth_user_id, from the Supabase auth user list."""
    out: Dict[str, str] = {}
    try:
        users = supabase_client.auth_admin_list_users() or []
    except Exception as exc:  # pragma: no cover - depends on live Supabase
        print(f"  [warn] could not list auth users for email fallback: {exc}", file=sys.stderr)
        return out
    for user in users:
        email = str((user or {}).get("email") or "").strip().lower()
        uid = str((user or {}).get("id") or "").strip()
        if email and uid:
            out[email] = uid
    return out


def resolve_user_id(
    subscription: Dict[str, Any],
    customer_email: str,
    accounts: Dict[str, Dict[str, Any]],
    emails: Dict[str, str],
) -> Optional[str]:
    """metadata.auth_user_id, then stored customer id, then account email."""
    metadata = subscription.get("metadata") or {}
    candidate = str(metadata.get("auth_user_id") or "").strip()
    if candidate and (not accounts or candidate in accounts):
        return candidate

    customer_id = str(subscription.get("customer") or "").strip()
    sub_id = str(subscription.get("id") or "").strip()
    for uid, row in accounts.items():
        if sub_id and str(row.get("stripe_subscription_id") or "").strip() == sub_id:
            return uid
        if customer_id and str(row.get("stripe_customer_id") or "").strip() == customer_id:
            return uid

    if customer_email:
        return emails.get(customer_email.strip().lower())
    return None


def pick_subscription(subs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A live subscription wins; otherwise the newest one."""

    def key(sub: Dict[str, Any]):
        live = 1 if str(sub.get("status")) in _LIVE_STATUSES else 0
        return (live, int(sub.get("created") or 0), int(sub.get("current_period_end") or 0))

    return max(subs, key=key)


def _account_payload(
    auth_user_id: str,
    row: Optional[Dict[str, Any]],
    updates: Dict[str, Any],
    emails: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """A complete user_accounts row with the subscription fields overlaid.

    PostgREST's `merge-duplicates` builds `INSERT ... ON CONFLICT DO UPDATE`, so
    the insert half must satisfy the table's NOT NULL columns (`email` et al).
    A subscription-only payload fails with 23502 even when the row exists.
    """
    if row:
        payload = dict(row)
        # Overlay every subscription-derived field, including None: a NULL from
        # Stripe (e.g. no end date because it is not cancelling) must clear the
        # stale value, otherwise a re-activated account keeps an old end date.
        payload.update(updates)
        payload["auth_user_id"] = auth_user_id
        return payload

    try:
        user = supabase_client.auth_admin_get_user(auth_user_id) or {}
    except Exception:
        user = {}
    if not user.get("email"):
        return None
    payload = _build_account_payload(user, updates)
    payload["auth_user_id"] = auth_user_id
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Reconcile Stripe subscriptions into user_accounts")
    ap.add_argument("--apply", action="store_true", help="Write the changes (default: dry run)")
    ap.add_argument("--user", help="Only reconcile this auth_user_id")
    args = ap.parse_args(argv)

    subscriptions = list_subscriptions()
    accounts = account_index()
    emails = email_to_auth_user_id()
    print(f"Stripe subscriptions: {len(subscriptions)}  |  user_accounts rows: {len(accounts)}")

    # email lookups are per customer, cached
    customer_emails: Dict[str, str] = {}

    by_user: Dict[str, List[Dict[str, Any]]] = {}
    unresolved: List[Dict[str, Any]] = []
    for sub in subscriptions:
        customer_id = str(sub.get("customer") or "").strip()
        if customer_id and customer_id not in customer_emails:
            try:
                customer = _stripe_get(f"customers/{customer_id}")
                customer_emails[customer_id] = str(customer.get("email") or "")
            except SystemExit:
                customer_emails[customer_id] = ""
        uid = resolve_user_id(sub, customer_emails.get(customer_id, ""), accounts, emails)
        if not uid:
            unresolved.append(sub)
            continue
        by_user.setdefault(uid, []).append(sub)

    print()
    changed = 0
    for uid in sorted(by_user):
        if args.user and uid != args.user:
            continue
        chosen = pick_subscription(by_user[uid])
        live = [s for s in by_user[uid] if str(s.get("status")) in _LIVE_STATUSES]
        row = accounts.get(uid) or {}
        updates = _stripe_updates_from_subscription(chosen)

        before = (row.get("subscription_plan"), row.get("subscription_status"), row.get("billing_interval"))
        after = (updates.get("subscription_plan"), updates.get("subscription_status"), updates.get("billing_interval"))
        mark = "CHANGE" if before != after else "ok    "
        if before != after:
            changed += 1
        print(f"[{mark}] {uid}  {row.get('email') or '?'}")
        print(f"          now: plan={before[0]!r} status={before[1]!r} interval={before[2]!r}")
        print(f"          new: plan={after[0]!r} status={after[1]!r} interval={after[2]!r}")
        print(f"          subscription {chosen['id']} ({chosen.get('status')})"
              f" customer {chosen.get('customer')}")
        if len(by_user[uid]) > 1:
            others = [s for s in by_user[uid] if s["id"] != chosen["id"]]
            print(f"          NOTE: {len(others)} other subscription(s) on this account: "
                  + ", ".join(f"{s['id']}({s.get('status')})" for s in others))
        if live and chosen["id"] not in {s["id"] for s in live}:
            print("          NOTE: picking a non-live subscription although a live one exists")

        if args.apply:
            payload = _account_payload(uid, row, updates, emails)
            if payload is None:
                print("          written: NO (could not build a complete account row)")
                continue
            saved = supabase_client.upsert_user_account(payload)
            print(f"          written: {'yes' if saved else 'NO'}")

    if unresolved:
        print()
        print(f"Could not map {len(unresolved)} subscription(s) to an account:")
        for sub in unresolved:
            print(f"  {sub['id']} customer={sub.get('customer')} "
                  f"email={customer_emails.get(str(sub.get('customer')), '')!r} "
                  f"status={sub.get('status')} "
                  f"metadata.auth_user_id={(sub.get('metadata') or {}).get('auth_user_id')!r}")

    print()
    if args.apply:
        print(f"Applied. {changed} account(s) changed.")
    else:
        print(f"Dry run. {changed} account(s) would change. Re-run with --apply to write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
