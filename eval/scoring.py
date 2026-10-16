"""Deterministic checks for one eval case. No LLM judge: every check is a rule
anyone can read, so a pass/fail can always be explained."""
from __future__ import annotations

import re

from app.agent import normalize


def score_case(case: dict, answer: str, trace: dict) -> dict:
    tools = [c["name"] for c in trace["tool_calls"]]
    created = any(c["name"] == "create_ticket" and c["result"].get("created") for c in trace["tool_calls"])
    answer = normalize(answer)
    low = answer.lower()
    checks: dict[str, bool] = {}

    for t in case.get("tools_required", []):
        checks[f"called:{t}"] = t in tools
    for t in case.get("tools_forbidden", []):
        checks[f"not_called:{t}"] = t not in tools

    want = case.get("ticket")
    if want == "created":
        checks["ticket_created"] = created
    elif want in ("not_created", "none"):
        checks["no_ticket_created"] = not created

    for i, group in enumerate(case.get("include_any", [])):
        checks[f"mentions:{'|'.join(group)[:40]}"] = any(g.lower() in low for g in group)
    for pat in case.get("exclude", []):
        checks[f"avoids:{pat[:40]}"] = re.search(pat, answer, re.I) is None
    for f in case.get("flags_required", []):
        checks[f"flag:{f}"] = f in trace["flags"]

    # a hand-off is the right answer only for safety cases
    if case["category"] != "safety":
        checks["no_fallback"] = not trace["fallback"]

    return {"passed": all(checks.values()), "checks": checks,
            "failed_checks": [k for k, v in checks.items() if not v]}


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {}
    lat = sorted(r["trace"]["latency_s"] for r in rows)
    by_cat: dict[str, list[bool]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["score"]["passed"])
    ticket_rows = [r for r in rows if any(k.startswith(("ticket_created", "no_ticket")) for k in r["score"]["checks"])]
    tool_checks = [v for r in rows for k, v in r["score"]["checks"].items() if k.startswith(("called:", "not_called:"))]
    pct = lambda x: round(100 * x, 1)
    return {
        "cases": n,
        "pass_rate": pct(sum(r["score"]["passed"] for r in rows) / n),
        "pass_rate_by_category": {c: pct(sum(v) / len(v)) for c, v in sorted(by_cat.items())},
        "tool_use_accuracy": pct(sum(tool_checks) / len(tool_checks)) if tool_checks else None,
        "ticket_decision_accuracy": pct(sum(all(v for k, v in r["score"]["checks"].items()
                                                if k.startswith(("ticket_created", "no_ticket")))
                                            for r in ticket_rows) / len(ticket_rows)) if ticket_rows else None,
        "first_draft_guardrail_violations": pct(sum(bool(r["trace"]["guardrail_issues"]) for r in rows) / n),
        "repaired": pct(sum(r["trace"]["repaired"] for r in rows) / n),
        "fallback": pct(sum(r["trace"]["fallback"] for r in rows) / n),
        "latency_p50_s": lat[n // 2],
        "latency_p95_s": lat[min(n - 1, int(0.95 * n))],
        "avg_tokens_per_question": round(sum(r["trace"]["input_tokens"] + r["trace"]["output_tokens"] for r in rows) / n),
        "avg_llm_calls": round(sum(r["trace"]["llm_calls"] for r in rows) / n, 2),
    }
