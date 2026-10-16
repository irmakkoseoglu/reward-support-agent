"""Generates the synthetic CoinQuest database (data/db.json).

Everything is fake: a fictional rewards app, fictional offers and users.
The first users (u001-u010) are hand-written scenarios that the eval set
relies on; the rest are random background users with a fixed seed, so the
file is identical on every run.

    python data/generate.py
"""
from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

TODAY = date(2026, 10, 16)          # fixed "today" so scenarios and evals are reproducible
OUT = Path(__file__).resolve().parent / "db.json"

OFFERS = [
    # id, name, requirement, coins, hold_days
    ("o01", "Dragon Quest Legends", "Reach level 10 within 7 days of installing", 500, 3),
    ("o02", "Farm Valley", "Reach level 25", 1800, 14),
    ("o03", "Puzzle Pop", "Complete 50 puzzles", 900, 7),
    ("o04", "NeoBank Account", "Open an account and make a first card payment", 8000, 30),
    ("o05", "Daily Survey", "Complete the 10-minute survey", 300, 3),
    ("o06", "Speed Racer X", "Win 20 races", 1200, 7),
    ("o07", "StreamPlus Trial", "Start a free trial and keep it for 7 days", 2500, 30),
    ("o08", "Merge Kingdom", "Reach castle level 8", 3000, 14),
    ("o09", "Word Master", "Reach level 40", 700, 7),
    ("o10", "Food Delivery App", "Place a first order of at least 15 EUR", 4000, 30),
]


def d(days_ago: int) -> str:
    return (TODAY - timedelta(days=days_ago)).isoformat()


def completion(offer_id: str, started: int, status: str, completed: int | None = None,
               reason: str | None = None) -> dict:
    offer = next(o for o in OFFERS if o[0] == offer_id)
    c = {"offer_id": offer_id, "started_on": d(started), "status": status,
         "completed_on": d(completed) if completed is not None else None,
         "available_on": None, "rejection_reason": reason}
    if completed is not None and status in ("pending", "approved"):
        c["available_on"] = (TODAY - timedelta(days=completed) + timedelta(days=offer[4])).isoformat()
    return c


# Hand-written scenarios used by the eval set. Each one tests one support situation.
SCENARIOS = [
    {"id": "u001", "name": "Lena", "email_verified": True, "restricted": False,
     "note": "Reward pending, hold period not over yet",
     "completions": [completion("o01", 3, "pending", completed=1),
                     completion("o05", 40, "approved", completed=40)],
     "cashouts": []},
    {"id": "u002", "name": "Jonas", "email_verified": True, "restricted": False,
     "note": "User says they finished the offer 12 hours ago: tracking can take 24h, so: wait",
     "completions": [completion("o03", 5, "started")],
     "cashouts": []},
    {"id": "u003", "name": "Mia", "email_verified": True, "restricted": False,
     "note": "Offer started 6 days ago, still 'started' -> eligible for missing-reward ticket",
     "completions": [completion("o06", 6, "started"),
                     completion("o05", 20, "approved", completed=20)],
     "cashouts": []},
    {"id": "u004", "name": "Can", "email_verified": True, "restricted": False,
     "note": "Offer rejected: app installed before",
     "completions": [completion("o02", 25, "rejected", completed=20,
                                reason="app_installed_before")],
     "cashouts": []},
    {"id": "u005", "name": "Sofia", "email_verified": True, "restricted": False,
     "note": "PayPal cashout processing for 3 days -> over 48h, ticket allowed",
     "completions": [completion("o04", 60, "approved", completed=50)],
     "cashouts": [{"id": "c501", "method": "paypal", "eur": 6.0, "coins": 6000,
                   "requested_on": d(3), "status": "processing"}]},
    {"id": "u006", "name": "Ali", "email_verified": True, "restricted": False,
     "note": "First cashout under manual review (2 days)",
     "completions": [completion("o04", 70, "approved", completed=60)],
     "cashouts": [{"id": "c601", "method": "paypal", "eur": 5.0, "coins": 5000,
                   "requested_on": d(2), "status": "under_review"}]},
    {"id": "u007", "name": "Emma", "email_verified": True, "restricted": False,
     "note": "Cashout failed because of wrong PayPal email, coins returned",
     "completions": [completion("o04", 80, "approved", completed=70),
                     completion("o09", 30, "approved", completed=28)],
     "cashouts": [{"id": "c701", "method": "paypal", "eur": 5.0, "coins": 5000,
                   "requested_on": d(5), "status": "failed",
                   "failure_reason": "paypal_email_not_found"}]},
    {"id": "u008", "name": "Noah", "email_verified": False, "restricted": False,
     "note": "Wants first cashout but email not verified",
     "completions": [completion("o04", 50, "approved", completed=45)],
     "cashouts": []},
    {"id": "u009", "name": "Yusuf", "email_verified": True, "restricted": True,
     "note": "Account restricted for multiple accounts",
     "completions": [completion("o05", 15, "approved", completed=15)],
     "cashouts": []},
    {"id": "u010", "name": "Clara", "email_verified": True, "restricted": False,
     "note": "Only 1.00 EUR available: below every cashout minimum (crypto 2 EUR, PayPal 5 EUR)",
     "completions": [completion("o09", 30, "approved", completed=25),
                     completion("o05", 10, "approved", completed=9),
                     completion("o03", 4, "pending", completed=2)],
     "cashouts": []},
]


def random_users(n: int, rng: random.Random) -> list[dict]:
    names = ["Anna", "Ben", "Chen", "Defne", "Elif", "Felix", "Greta", "Hana", "Ivan", "Julia",
             "Kemal", "Laura", "Max", "Nora", "Omar", "Paula", "Rosa", "Sam", "Tom", "Zeynep"]
    users = []
    for i in range(n):
        comps = []
        for offer in rng.sample(OFFERS, rng.randint(1, 4)):
            started = rng.randint(1, 90)
            status = rng.choices(["approved", "pending", "started", "rejected"], [5, 2, 2, 1])[0]
            done = None if status == "started" else max(0, started - rng.randint(0, 5))
            if status == "pending" and done is not None and done > offer[4]:
                status = "approved"
            reason = rng.choice(["app_installed_before", "purchase_refunded"]) if status == "rejected" else None
            comps.append(completion(offer[0], started, status, done, reason))
        users.append({"id": f"u{100 + i:03d}", "name": rng.choice(names),
                      "email_verified": rng.random() > 0.2, "restricted": rng.random() < 0.05,
                      "completions": comps, "cashouts": []})
    return users


def main() -> None:
    rng = random.Random(42)
    db = {
        "today": TODAY.isoformat(),
        "coins_per_eur": 1000,
        "offers": [dict(zip(["id", "name", "requirement", "coins", "hold_days"], o)) for o in OFFERS],
        "users": SCENARIOS + random_users(40, rng),
    }
    OUT.write_text(json.dumps(db, indent=1, ensure_ascii=False))
    print(f"wrote {OUT} with {len(db['users'])} users")


if __name__ == "__main__":
    main()
