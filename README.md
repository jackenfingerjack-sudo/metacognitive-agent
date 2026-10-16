# Metacognitive Agent

Generate → Reflect → Revise agent with a persistent self-model, tools (safe calculator, DuckDuckGo search) and 26 markdown skills.

## Local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...        # optional: OPENAI_BASE_URL, OPENAI_MODEL (default gpt-4o-mini)
python metacognitive_agent.py       # CLI: /quit /self /reset /skills
streamlit run app.py                # web UI at http://localhost:8501
```
Self-model is stored at `~/.metacognitive_agent/self_model.json` (override with `METACOG_HOME`). Skills load from `./skills` (override with `SKILLS_DIR`).

## Docker
```bash
export OPENAI_API_KEY=sk-...
docker compose up --build           # http://localhost:8501
```
The self-model persists in the `agent_data` volume; `./skills` is mounted so you can edit skills without rebuilding.

## Live test results (Oct 16, 2026)

Model configured: `gpt-4o-mini` (default). Environment: Python 3.13 venv, openai 3.31, streamlit 1.66, ddgs 9.16.

- **Passed:** calculator tool (`(1234*5678)/7 + sqrt(2025)` = 1000995.2857…); `web_search` returning live results via `ddgs`; skill matching (code-review prompt → `code_review`, planning prompt → `planning`); 26 skills loaded; CLI with piped `/self`, `/skills`, `/reset`, `/quit`, plus clean exit on EOF; Streamlit headless on :8501 (`/` → 200, `/_stcore/health` → `ok`).
- **Fixed:** switched from the renamed `duckduckgo-search` package to `ddgs`, with a fallback import, and added backend retries because single searches were intermittently returning "No results found".
- **Not verified:** the full LLM generate → critique → revise loop (including the JSON `response_format` critique). The test API key had no credits (HTTP 429 `insufficient_quota`). The CLI reports this error and keeps running.
- **Caveat:** generic phrases like "what is" match `technical_explanation` (for example "What is 2+2").
