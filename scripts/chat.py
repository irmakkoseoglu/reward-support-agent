"""Chat with the agent in the terminal, as one of the demo users.

    python scripts/chat.py u001
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

env = Path(__file__).resolve().parent.parent / ".env"          # tiny .env loader, no extra package
if env.exists():
    for line in env.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from app.agent import SupportAgent  # noqa: E402
from app.llm import OpenAICompatibleLLM  # noqa: E402

user = sys.argv[1] if len(sys.argv) > 1 else "u001"
agent = SupportAgent(OpenAICompatibleLLM())
u = agent.store.users[user]
print(f"Chatting as {u['name']} ({user}). Situation: {u.get('note', '-')}. Empty line to quit.\n")
history = []
while (q := input("you> ").strip()):
    answer, tr = agent.answer(user, q, history)
    print(f"\nagent> {answer}\n")
    print(f"  [tools: {[c['name'] for c in tr.tool_calls]} | docs: {[d['id'] for d in tr.retrieved]} | "
          f"issues: {tr.guardrail_issues or '-'}{' (repaired)' if tr.repaired else ''}{' (FALLBACK)' if tr.fallback else ''} | {tr.latency_s}s | {tr.input_tokens}+{tr.output_tokens} tokens]\n")
    history += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]
