#!/usr/bin/env python3
"""Mint per-person demo invite links (account-free access).

Each link carries an unguessable token; clicking it gives that person their own
isolated, tracked session. Email the printed URL to the invitee.

    python3 mint_invite.py --name "Prof Jane Doe"
    python3 mint_invite.py --name "Workshop seat" --count 20      # batch
    python3 mint_invite.py --list                                  # show issued invites
    python3 mint_invite.py --revoke inv_xxx
"""
from __future__ import annotations

import argparse
import datetime as _dt

from config import CONFIG
from tenants import TenantStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="invitee label (shown in notifications + reports)")
    ap.add_argument("--count", type=int, default=1, help="mint N links (e.g. workshop seats)")
    ap.add_argument("--max-sessions", type=int, default=0, help="cap sessions per invite (0=unlimited)")
    ap.add_argument("--list", action="store_true", help="list issued invites")
    ap.add_argument("--revoke", metavar="TOKEN", help="revoke an invite token")
    args = ap.parse_args()

    store = TenantStore(CONFIG.state_path)

    if args.revoke:
        store.revoke_invite(args.revoke)
        print(f"revoked {args.revoke}")
        return

    if args.list:
        for r in store.list_invites():
            when = _dt.datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d %H:%M")
            flag = " [REVOKED]" if r["revoked"] else ""
            print(f"{r['token']}  {when}  {r['name']!r}{flag}")
        return

    if not args.name:
        ap.error("--name is required when minting")

    for i in range(args.count):
        label = args.name if args.count == 1 else f"{args.name} #{i+1}"
        token = store.mint_invite(label, max_sessions=args.max_sessions)
        print(f"{CONFIG.public_base.rstrip('/')}/start?t={token}")


if __name__ == "__main__":
    main()
