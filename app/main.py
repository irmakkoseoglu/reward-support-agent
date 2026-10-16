"""HTTP API.

    uvicorn app.main:app --reload
    POST /chat  {"user_id": "u001", "message": "Where are my coins?"}
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import SupportAgent, trace_dict
from .llm import OpenAICompatibleLLM

app = FastAPI(title="CoinQuest Reward Support Agent")


class ChatIn(BaseModel):
    user_id: str = Field(examples=["u001"])
    message: str = Field(min_length=1, max_length=2000)
    history: list[dict] = Field(default_factory=list, description="previous turns: role + content")


@lru_cache
def get_agent() -> SupportAgent:
    return SupportAgent(OpenAICompatibleLLM())


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/users")
def users():
    """Demo users with a one-line description of their situation (for the UI)."""
    s = get_agent().store
    return [{"id": u["id"], "name": u["name"], "note": u.get("note", "")}
            for u in s.users.values() if "note" in u]


@app.post("/chat")
def chat(body: ChatIn):
    agent = get_agent()
    if body.user_id not in agent.store.users:
        raise HTTPException(404, "unknown user")
    history = [m for m in body.history if m.get("role") in ("user", "assistant")]
    answer, trace = agent.answer(body.user_id, body.message, history)
    return {"answer": answer, "trace": trace_dict(trace)}
