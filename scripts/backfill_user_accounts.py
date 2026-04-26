"""Backfill Supabase public.user_accounts from Supabase Auth users.

Usage:
  c:/Apps/NHL/.venv/Scripts/python.exe scripts/backfill_user_accounts.py
  c:/Apps/NHL/.venv/Scripts/python.exe scripts/backfill_user_accounts.py --include-existing
  c:/Apps/NHL/.venv/Scripts/python.exe scripts/backfill_user_accounts.py --limit 20
"""

import argparse
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from app.supabase_client import auth_admin_list_users, get_user_account, upsert_user_account


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _isoformat_utc(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_username(value: Any) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", str(value or "").strip().lower()).strip("-")


def _valid_username(value: Any) -> bool:
    username = _normalize_username(value)
    return bool(re.fullmatch(r"[a-z0-9](?:[a-z0-9._-]{1,30}[a-z0-9])?", username))


def _record_from_auth_user(user: Dict[str, Any]) -> Dict[str, Any]:
    user_id = str(user.get("id") or "").strip()
    email = str(user.get("email") or "").strip().lower()
    user_meta = user.get("user_metadata") or {}
    app_meta = user.get("app_metadata") or {}

    created_at = _parse_iso_datetime(user.get("created_at")) or datetime.now(timezone.utc)
    trial_started = _parse_iso_datetime(user_meta.get("trial_started_at")) or created_at
    trial_expires = _parse_iso_datetime(user_meta.get("trial_expires_at")) or (trial_started + timedelta(days=7))

    raw_username = user_meta.get("username") or email.split("@", 1)[0]
    username = _normalize_username(raw_username)
    if not _valid_username(username):
        username = ""

    display_name = str(
        user_meta.get("display_name")
        or user_meta.get("name")
        or username
        or email
        or "Account"
    ).strip()

    subscription_status = str(app_meta.get("subscription_status") or user_meta.get("subscription_status") or "trialing").strip().lower() or "trialing"
    subscription_plan = str(app_meta.get("subscription_plan") or user_meta.get("subscription_plan") or "trial").strip() or "trial"
    billing_interval = str(app_meta.get("billing_interval") or user_meta.get("billing_interval") or "").strip().lower() or None

    return {
        "auth_user_id": user_id,
        "email": email,
        "username": username or None,
        "display_name": display_name,
        "is_admin": bool(app_meta.get("is_admin") or user_meta.get("is_admin") or False),
        "subscription_status": subscription_status,
        "subscription_plan": subscription_plan,
        "billing_interval": billing_interval,
        "trial_started_at": _isoformat_utc(trial_started),
        "trial_expires_at": _isoformat_utc(trial_expires),
        "updated_at": _isoformat_utc(datetime.now(timezone.utc)),
    }


def _minimal_record(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "auth_user_id": payload.get("auth_user_id"),
        "email": payload.get("email"),
        "username": payload.get("username"),
        "display_name": payload.get("display_name") or payload.get("email") or "Account",
        "is_admin": bool(payload.get("is_admin")),
        "subscription_status": payload.get("subscription_status") or "trialing",
        "subscription_plan": payload.get("subscription_plan") or "trial",
        "billing_interval": payload.get("billing_interval"),
        "trial_started_at": payload.get("trial_started_at"),
        "trial_expires_at": payload.get("trial_expires_at"),
        "updated_at": payload.get("updated_at"),
    }


def run_backfill(include_existing: bool, limit: int) -> int:
    users = auth_admin_list_users() or []
    if limit > 0:
        users = users[:limit]

    scanned = 0
    skipped = 0
    saved = 0
    failed = 0
    errors: List[str] = []

    for user in users:
        if not isinstance(user, dict):
            continue
        user_id = str(user.get("id") or "").strip()
        if not user_id:
            continue

        scanned += 1
        existing = get_user_account(user_id)
        if existing and not include_existing:
            skipped += 1
            continue

        payload = _record_from_auth_user(user)

        try:
            out = upsert_user_account(payload)
            if not out:
                # Retry with the smallest possible payload in case of schema mismatch.
                out = upsert_user_account(_minimal_record(payload))
            if out and str(out.get("auth_user_id") or "").strip() == user_id:
                saved += 1
            else:
                failed += 1
                errors.append(f"{user_id}: upsert returned no row")
        except Exception as exc:
            failed += 1
            errors.append(f"{user_id}: {exc}")

    print(f"[user-backfill] scanned={scanned} saved={saved} skipped={skipped} failed={failed}")
    if errors:
        print("[user-backfill] first errors:")
        for line in errors[:15]:
            print(f"  - {line}")

    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill public.user_accounts from Supabase Auth users")
    parser.add_argument("--include-existing", action="store_true", help="Also update users that already exist in user_accounts")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N users")
    args = parser.parse_args()

    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_SERVICE_KEY"):
        print("[user-backfill] Missing SUPABASE_URL or SUPABASE_SERVICE_KEY environment variables.")
        return 2

    return run_backfill(include_existing=bool(args.include_existing), limit=max(0, int(args.limit)))


if __name__ == "__main__":
    raise SystemExit(main())
