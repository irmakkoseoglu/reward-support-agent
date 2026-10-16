# Reward Support Agent

An LLM support agent for a **fictional** rewards app ("CoinQuest": users earn coins for trying
partner apps and cash them out). It answers questions like *"I reached level 10, where are my
coins?"* by combining help-centre retrieval (RAG), tool calls on the user's own data, and
guardrails that stop it from promising things or inventing facts.

All data is synthetic. Full write-up, eval results and demo follow.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env            # put your free Groq API key in .env
python scripts/chat.py u001     # chat in the terminal as demo user u001
uvicorn app.main:app --reload   # or start the API: http://127.0.0.1:8000/docs
pytest -q                       # tests run offline, no API key needed
```
