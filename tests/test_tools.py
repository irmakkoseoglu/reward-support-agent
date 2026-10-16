"""Business-rule tests: each hand-written scenario must behave as the help centre says."""
import pytest

from app.tools import Store, call_tool


@pytest.fixture
def s():
    return Store()


def test_pending_reward_has_release_date(s):
    o = s.get_offer_history("u001", "Dragon")["offers"][0]
    assert o["status"] == "pending" and o["coins_available_on"] == "2026-10-18"


def test_missing_reward_ticket_blocked_before_72h(s):
    # u001's offer is pending, not missing
    assert not s.create_ticket("u001", "missing_reward", "x", "Dragon")["created"]


def test_missing_reward_ticket_allowed_after_72h(s):
    assert not s.create_ticket("u003", "missing_reward", "x", "Speed Racer")["created"]          # hours unknown
    assert not s.create_ticket("u003", "missing_reward", "x", "Speed Racer", 12)["created"]      # wait 24h
    r = s.create_ticket("u003", "missing_reward", "won 20 races", "Speed Racer", 100)
    assert r["created"]
    assert not s.create_ticket("u003", "missing_reward", "again", "Speed Racer", 100)["created"]  # no duplicates


def test_rejected_offer_reason(s):
    o = s.get_offer_history("u004")["offers"][0]
    assert o["status"] == "rejected" and o["rejection_reason"] == "app_installed_before"


def test_cashout_ticket_only_after_48h(s):
    assert s.create_ticket("u005", "cashout", "paypal not arrived")["created"]
    assert not s.create_ticket("u006", "cashout", "review too slow")["created"]


def test_failed_cashout_returns_coins(s):
    acc = s.get_account("u007")
    assert acc["available_coins"] == 8000 + 700   # failed 5000 came back


def test_balance_below_minimum(s):
    acc = s.get_account("u010")
    assert acc["available_eur"] == 1.0 and acc["cashout_possible_with"] == []


def test_unverified_and_restricted_flags(s):
    assert s.get_account("u008")["email_verified"] is False
    assert s.get_account("u009")["restricted"] is True
    assert not s.create_ticket("u001", "restriction_appeal", "x")["created"]


def test_dispatch_uses_session_user_and_handles_bad_input(s):
    assert call_tool(s, "u001", "get_account", {})["user_id"] == "u001"
    assert "error" in call_tool(s, "u001", "get_account", {"user_id": "u002"})  # model cannot switch user
    assert "error" in call_tool(s, "u001", "delete_account", {})
