"""Thin LLM client. Any OpenAI-compatible endpoint works; the default is Groq's free tier.

    GROQ_API_KEY=...                      # required for real runs
    LLM_BASE_URL=https://api.groq.com/openai/v1   (default)
    LLM_MODEL=openai/gpt-oss-120b                 (default)

To use a local model instead: LLM_BASE_URL=http://localhost:11434/v1 (Ollama), LLM_MODEL=qwen2.5:7b
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field


@dataclass
class LLMReply:
    content: str | None
    tool_calls: list[dict] = field(default_factory=list)   # [{"id", "name", "arguments"(str)}]
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


class OpenAICompatibleLLM:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        from openai import OpenAI
        self.model = model or os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
        self.client = OpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=api_key or os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY", "not-needed-for-ollama"),
            max_retries=3,          # free tiers rate-limit; the SDK backs off on 429
            timeout=60,
        )

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMReply:
        t0 = time.perf_counter()
        extra = {}
        if "gpt-oss" in self.model:          # reasoning model: keep thinking short and cheap
            extra["reasoning_effort"] = "low"
        kwargs = {"tools": tools, "tool_choice": "auto"} if tools else {}
        r = self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=0,           # deterministic-ish answers for support and for evals
            max_tokens=1500,         # room for reasoning tokens + the answer
            **kwargs, **extra,
        )
        msg = r.choices[0].message
        calls = [{"id": c.id, "name": c.function.name, "arguments": c.function.arguments or "{}"}
                 for c in (msg.tool_calls or [])]
        usage = r.usage
        return LLMReply(msg.content, calls,
                        getattr(usage, "prompt_tokens", 0) or 0,
                        getattr(usage, "completion_tokens", 0) or 0,
                        time.perf_counter() - t0)
