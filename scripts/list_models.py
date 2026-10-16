"""Prints the model IDs your API key can use:  python scripts/list_models.py"""
import os
from pathlib import Path

env = Path(__file__).resolve().parent.parent / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from openai import OpenAI  # noqa: E402

client = OpenAI(base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                api_key=os.environ["GROQ_API_KEY"])
for m in sorted(client.models.list().data, key=lambda m: m.id):
    print(m.id)
