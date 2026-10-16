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
