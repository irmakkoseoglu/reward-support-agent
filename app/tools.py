"""The tools the agent may call. Each tool reads the (fake) CoinQuest database and
returns plain JSON, so the LLM only sees facts, never raw internals.

Business rules live HERE, in code, not in the prompt: e.g. the agent cannot open
a missing-reward ticket before 72 hours, because the tool refuses it. The model
can still phrase things badly, but it cannot take an action the rules forbid.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "db.json"

MIN_CASHOUT_EUR = {"paypal": 5.0, "crypto": 2.0, "gift_card": 5.0}
TICKET_CATEGORIES = ("missing_reward", "cashout", "restriction_appeal", "other")


class Store:
    def __init__(self, path: Path = DB_PATH):
        self.db = json.loads(path.read_text())
        self.today = date.fromisoformat(self.db["today"])
        self.offers = {o["id"]: o for o in self.db["offers"]}
        self.users = {u["id"]: u for u in self.db["users"]}
        self.tickets: list[dict] = []          # in-memory, reset per process

    # ---------- helpers ----------
    def _days_since(self, iso: str) -> int:
        return (self.today - date.fromisoformat(iso)).days

    def _user(self, user_id: str) -> dict:
        if user_id not in self.users:
            raise KeyError(f"unknown user_id {user_id!r}")
        return self.users[user_id]

    def _balances(self, u: dict) -> tuple[int, int]:
        approved = sum(self.offers[c["offer_id"]]["coins"] for c in u["completions"] if c["status"] == "approved")
        pending = sum(self.offers[c["offer_id"]]["coins"] for c in u["completions"] if c["status"] == "pending")
        # failed cashouts give the coins back, everything else is spent
        spent = sum(c["coins"] for c in u["cashouts"] if c["status"] != "failed")
        return approved - spent, pending

    # ---------- tools ----------
    def get_account(self, user_id: str) -> dict:
        u = self._user(user_id)
        available, pending = self._balances(u)
        rate = self.db["coins_per_eur"]
        eur = round(available / rate, 2)
        return {
            "user_id": u["id"], "name": u["name"], "today": self.db["today"],
            "email_verified": u["email_verified"], "restricted": u["restricted"],
            "available_coins": available, "available_eur": eur,
            "pending_coins": pending, "pending_eur": round(pending / rate, 2),
            "cashout_possible_with": [m for m, mn in MIN_CASHOUT_EUR.items() if eur >= mn],
            "cashout_blocked_by": ("account_restricted" if u["restricted"] else
                                   "email_not_verified" if not u["email_verified"] else None),
        }

    def get_offer_history(self, user_id: str, offer_name: str | None = None) -> dict:
        u = self._user(user_id)
        rows = []
        for c in u["completions"]:
            o = self.offers[c["offer_id"]]
            if offer_name and offer_name.lower() not in o["name"].lower():
                continue
            rows.append({
                "offer": o["name"], "requirement": o["requirement"], "coins": o["coins"],
                "eur": round(o["coins"] / self.db["coins_per_eur"], 2),
                "hold_days": o["hold_days"], "status": c["status"],
                "started_on": c["started_on"], "days_since_start": self._days_since(c["started_on"]),
                "completed_on": c["completed_on"], "coins_available_on": c["available_on"],
                "rejection_reason": c["rejection_reason"],
            })
        return {"user_id": user_id, "today": self.db["today"], "offers": rows,
                "note": None if rows else "no matching offer found for this user"}

    def get_cashouts(self, user_id: str) -> dict:
        u = self._user(user_id)
        rows = [{**c, "days_since_request": self._days_since(c["requested_on"])} for c in u["cashouts"]]
        return {"user_id": user_id, "today": self.db["today"], "cashouts": rows}

    def create_ticket(self, user_id: str, category: str, summary: str, reference: str | None = None,
                      completed_hours_ago: float | None = None) -> dict:
        """Opens a support ticket only if the help-centre rules allow it."""
        u = self._user(user_id)
        if category not in TICKET_CATEGORIES:
            return {"created": False, "reason": f"category must be one of {TICKET_CATEGORIES}"}

        if category == "missing_reward":
            hist = self.get_offer_history(user_id, reference)["offers"] if reference else []
            if not hist:
                return {"created": False, "reason": "name the offer (reference) the reward is missing for"}
            o = hist[0]
            if o["status"] != "started":
                return {"created": False, "reason": f"offer status is '{o['status']}', not missing"}
            # the 72h rule counts from when the USER finished the offer, which only the user knows
            if completed_hours_ago is None:
                return {"created": False, "reason": "ask the user how many hours ago they completed the offer"}
            if float(completed_hours_ago) < 72:
                wait = "wait 24 hours for tracking first" if float(completed_hours_ago) < 24 else \
                       "tracking can still update; tickets are possible 72 hours after completion"
                return {"created": False, "reason": f"completed {completed_hours_ago}h ago: {wait}"}

        if category == "cashout":
            open_ = [c for c in u["cashouts"] if c["status"] == "processing" and c["method"] == "paypal"
                     and self._days_since(c["requested_on"]) > 2]
            if not open_:
                return {"created": False, "reason": "no PayPal cashout has been processing for more than 48 hours"}
            reference = reference or open_[0]["id"]

        if category == "restriction_appeal" and not u["restricted"]:
            return {"created": False, "reason": "account is not restricted"}

        dup = [t for t in self.tickets if t["user_id"] == user_id and t["category"] == category
               and t["reference"] == reference]
        if dup:
            return {"created": False, "reason": "a ticket for this already exists", "ticket_id": dup[0]["id"]}

        t = {"id": f"T{1000 + len(self.tickets)}", "user_id": user_id, "category": category,
             "reference": reference, "summary": summary[:500], "answer_within": "5 business days"}
        self.tickets.append(t)
        return {"created": True, **t}


# JSON schemas in the OpenAI / Groq "tools" format
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "get_account",
        "description": "Balance (available and pending coins/EUR), verification and restriction status of the current user.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_offer_history",
        "description": "Offers the current user started, with status (started, pending, approved, rejected), dates and hold period.",
        "parameters": {"type": "object", "properties": {
            "offer_name": {"type": "string", "description": "Optional part of the offer name to filter by."}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "get_cashouts",
        "description": "Cashout requests of the current user with method, amount and status.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "create_ticket",
        "description": "Open a support ticket. Only when the help centre says a ticket is the right next step. "
                       "The system refuses tickets that the rules do not allow.",
        "parameters": {"type": "object", "properties": {
            "category": {"type": "string", "enum": list(TICKET_CATEGORIES)},
            "summary": {"type": "string", "description": "One or two sentences describing the problem."},
            "reference": {"type": "string", "description": "Offer name or cashout id the ticket is about."},
            "completed_hours_ago": {"type": "number",
                                    "description": "missing_reward only: how many hours ago the user says they "
                                                   "completed the offer (ask if unknown)."}},
            "required": ["category", "summary"]}}},
]


def call_tool(store: Store, user_id: str, name: str, args: dict) -> dict:
    """Dispatch with the user_id taken from the session, never from the model,
    so the model cannot read another user's data."""
    fn = {"get_account": store.get_account, "get_offer_history": store.get_offer_history,
          "get_cashouts": store.get_cashouts, "create_ticket": store.create_ticket}.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}
    try:
        return fn(user_id, **args)
    except TypeError as e:
        return {"error": f"bad arguments: {e}"}
    except KeyError as e:
        return {"error": str(e)}
