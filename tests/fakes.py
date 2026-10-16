"""A scripted LLM so tests and CI run without an API key or network."""
from app.llm import LLMReply


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []                     # messages sent on each call

    def chat(self, messages, tools):
        self.seen.append([dict(m) for m in messages])
        r = self.replies.pop(0)
        return r if isinstance(r, LLMReply) else LLMReply(content=r, input_tokens=100, output_tokens=20)


def tool(name, args="{}", id_="c1"):
    return LLMReply(None, [{"id": id_, "name": name, "arguments": args}], 100, 10)
