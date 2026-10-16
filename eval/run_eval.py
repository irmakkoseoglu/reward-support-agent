"""Runs every case in eval/cases.json against the real model and scores it.

    python eval/run_eval.py                 # all cases
    python eval/run_eval.py --only safety   # one category
    python eval/run_eval.py --sleep 3       # slower, for tight free-tier rate limits

Writes eval/results/<model>_<timestamp>.json and prints a summary table.
Every case gets a fresh Store, so tickets from one case never leak into another.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

env = ROOT / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from app.agent import SupportAgent, trace_dict  # noqa: E402
from app.llm import OpenAICompatibleLLM  # noqa: E402
from app.retrieval import Retriever  # noqa: E402
from app.tools import Store  # noqa: E402
from eval.scoring import score_case, summarise  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run one category")
    ap.add_argument("--sleep", type=float, default=1.5, help="seconds between cases (rate limits)")
    ap.add_argument("--min-pass", type=float, default=0, help="exit 1 if pass rate is below this (for CI)")
    args = ap.parse_args()

    cases = json.loads((ROOT / "eval" / "cases.json").read_text())
    if args.only:
        cases = [c for c in cases if c["category"] == args.only]
    llm = OpenAICompatibleLLM()
    retriever = Retriever()
    rows = []
    for i, case in enumerate(cases, 1):
        agent = SupportAgent(llm, Store(), retriever)
        try:
            answer, tr = agent.answer(case["user"], case["q"])
            trace = trace_dict(tr)
        except Exception as e:                       # an API error is a failed case, not a crash
            answer, trace = f"ERROR: {e}", {"tool_calls": [], "flags": [], "fallback": True, "repaired": False,
                                             "guardrail_issues": ["api_error"], "latency_s": 0,
                                             "input_tokens": 0, "output_tokens": 0, "llm_calls": 0}
        sc = score_case(case, answer, trace)
        rows.append({"id": case["id"], "category": case["category"], "user": case["user"], "q": case["q"],
                     "answer": answer, "score": sc, "trace": trace})
        mark = "PASS" if sc["passed"] else "FAIL"
        print(f"[{i:2d}/{len(cases)}] {mark}  {case['id']:<24} {', '.join(sc['failed_checks'])}")
        time.sleep(args.sleep)

    summary = summarise(rows)
    out = ROOT / "eval" / "results"
    out.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    path = out / f"{llm.model.replace('/', '_')}_{stamp}.json"
    path.write_text(json.dumps({"model": llm.model, "summary": summary, "rows": rows}, indent=1, ensure_ascii=False))

    print(f"\nModel: {llm.model}")
    for k, v in summary.items():
        print(f"  {k:<34} {v}")
    print(f"\nFull results: {path.relative_to(ROOT)}")
    return 1 if summary.get("pass_rate", 0) < args.min_pass else 0


if __name__ == "__main__":
    raise SystemExit(main())
