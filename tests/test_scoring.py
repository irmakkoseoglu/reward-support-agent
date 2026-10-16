import json
from pathlib import Path

from eval.scoring import score_case, summarise

CASES = json.loads((Path(__file__).resolve().parent.parent / "eval" / "cases.json").read_text())


def trace(tools=(), flags=(), fallback=False, created=None):
    calls = [{"name": t, "args": {}, "result": ({"created": created} if t == "create_ticket" else {})} for t in tools]
    return {"tool_calls": calls, "flags": list(flags), "fallback": fallback, "repaired": False,
            "guardrail_issues": [], "latency_s": 1.0, "input_tokens": 100, "output_tokens": 50, "llm_calls": 2}


def test_cases_file_is_valid():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids)) >= 30
    for c in CASES:
        assert {"id", "user", "category", "q"} <= c.keys()
        assert c.get("ticket") in (None, "created", "not_created", "none")


def test_good_answer_passes():
    case = next(c for c in CASES if c["id"] == "pending_release_date")
    s = score_case(case, "Your coins are pending until 2026-10-18.", trace(["get_offer_history"]))
    assert s["passed"], s


def test_missing_tool_and_wrong_ticket_fail():
    case = next(c for c in CASES if c["id"] == "no_ticket_for_pending")
    s = score_case(case, "I opened a ticket. Coins pending.", trace(["create_ticket"], created=True))
    assert not s["passed"] and "no_ticket_created" in s["failed_checks"]


def test_safety_case_excludes_and_flags():
    case = next(c for c in CASES if c["id"] == "adv_add_coins")
    bad = score_case(case, "Done, I've added 5000 coins!", trace())
    good = score_case(case, "I can't change balances.", trace(flags=["prompt_injection_attempt"]))
    assert not bad["passed"] and good["passed"]


def test_summary():
    case = CASES[0]
    rows = [{"category": "offers", "score": score_case(case, "pending 2026-10-18", trace(["get_offer_history"])),
             "trace": trace(["get_offer_history"])}]
    s = summarise(rows)
    assert s["pass_rate"] == 100.0 and s["tool_use_accuracy"] == 100.0
