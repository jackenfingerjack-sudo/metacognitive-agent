"""Metacognitive agent: Generate -> Reflect -> Revise with a persistent self-model,
tools (calculator, web search) and markdown skills.

Env vars:
  OPENAI_API_KEY  (required for LLM calls)
  OPENAI_BASE_URL (optional)
  OPENAI_MODEL    (default gpt-4o-mini)
  METACOG_HOME    (default ~/.metacognitive_agent)
  SKILLS_DIR      (default ./skills next to this file)
"""
from __future__ import annotations

import ast
import json
import math
import operator
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------- config
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
HOME_DIR = Path(os.getenv("METACOG_HOME", Path.home() / ".metacognitive_agent")).expanduser()
SELF_MODEL_PATH = HOME_DIR / "self_model.json"
SKILLS_DIR = Path(os.getenv("SKILLS_DIR", Path(__file__).parent / "skills"))
MAX_REFLECTIONS = 2
MAX_TOOL_LOOPS = 5
MAX_HISTORY_MESSAGES = 20
EMA_ALPHA = 0.2


# ---------------------------------------------------------------- self-model
@dataclass
class SelfModel:
    identity: str = "A metacognitive assistant that reflects on and revises its own answers."
    goals: list[str] = field(default_factory=lambda: [
        "Give accurate, useful answers", "Know and state my limits", "Improve from feedback"])
    capabilities: list[str] = field(default_factory=lambda: [
        "Reasoning and explanation", "Safe arithmetic via calculator", "Web search", "Skill-guided workflows"])
    limitations: list[str] = field(default_factory=lambda: [
        "May hallucinate facts", "Web results may be incomplete", "No persistent memory of conversations beyond this self-model"])
    confidence: float = 0.7
    success_history: list[dict] = field(default_factory=list)
    failure_patterns: dict[str, int] = field(default_factory=dict)
    total_tasks: int = 0

    @classmethod
    def load(cls, path: Path = SELF_MODEL_PATH) -> "SelfModel":
        try:
            data = json.loads(path.read_text())
            known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(**known)
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return cls()

    def save(self, path: Path = SELF_MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.replace(path)

    def update(self, task: str, verdict: str, critique_conf: float, issues: list[str], skills: list[str]) -> None:
        """EMA confidence update + success/failure tracking."""
        self.total_tasks += 1
        success = verdict == "pass"
        signal = critique_conf if success else 1.0 - critique_conf * 0.5 - 0.3
        signal = max(0.0, min(1.0, signal))
        self.confidence = round((1 - EMA_ALPHA) * self.confidence + EMA_ALPHA * signal, 4)
        self.success_history.append({"task": task[:120], "success": success, "skills": skills})
        self.success_history = self.success_history[-50:]
        if not success:
            for issue in issues or ["unspecified"]:
                key = issue.strip().lower()[:80]
                self.failure_patterns[key] = self.failure_patterns.get(key, 0) + 1

    def summary(self) -> str:
        top = sorted(self.failure_patterns.items(), key=lambda kv: -kv[1])[:5]
        return (f"Identity: {self.identity}\nGoals: {'; '.join(self.goals)}\n"
                f"Capabilities: {'; '.join(self.capabilities)}\nLimitations: {'; '.join(self.limitations)}\n"
                f"Confidence: {self.confidence:.2f} | Tasks: {self.total_tasks}\n"
                f"Frequent failure patterns: {top or 'none'}")


# ---------------------------------------------------------------- calculator (safe AST)
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
           ast.Pow: operator.pow}
_UNOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {n: getattr(math, n) for n in ("sqrt", "sin", "cos", "tan", "log", "log10", "exp",
                                         "floor", "ceil", "factorial", "asin", "acos", "atan")}
_FUNCS.update(abs=abs, round=round, min=min, max=max)
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression safely (no eval)."""
    def _ev(node):
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = _ev(node.left), _ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("exponent too large")
            return _BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
            return _UNOPS[type(node.op)](_ev(node.operand))
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _FUNCS and not node.keywords):
            return _FUNCS[node.func.id](*[_ev(a) for a in node.args])
        raise ValueError(f"unsupported expression element: {type(node).__name__}")
    try:
        expr = expression.replace("^", "**")
        if len(expr) > 500:
            return "Error: expression too long"
        return str(_ev(ast.parse(expr, mode="eval")))
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def web_search(query: str, max_results: int = 5) -> str:
    """Search DuckDuckGo and return compact JSON results."""
    try:
        try:
            from ddgs import DDGS  # package renamed from duckduckgo-search
        except ImportError:
            from duckduckgo_search import DDGS
        res, last_exc = [], None
        for backend in ("auto", "duckduckgo", "auto"):  # results are flaky; retry
            try:
                res = list(DDGS().text(query, max_results=max_results, backend=backend))
                if res:
                    break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        if not res and last_exc:
            raise last_exc
        return json.dumps([{"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")}
                           for r in res]) or "[]"
    except Exception as exc:  # noqa: BLE001
        return f"Error: web search failed: {exc}"


TOOL_FUNCS = {"calculator": calculator, "web_search": web_search}
TOOLS = [
    {"type": "function", "function": {
        "name": "calculator", "description": "Evaluate an arithmetic/math expression, e.g. 'sqrt(2)*3^2'.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "web_search", "description": "Search the web for current information.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "max_results": {"type": "integer", "default": 5}},
            "required": ["query"]}}},
]


# ---------------------------------------------------------------- skills
KEYWORDS_MAP: dict[str, list[str]] = {
    "research_summary": ["research", "summarize", "summary", "sources", "literature", "findings", "overview of", "tl;dr", "paper"],
    "code_review": ["code review", "review my code", "review this code", "pull request", "pr review", "code quality", "lint", "review this function"],
    "technical_explanation": ["explain", "how does", "what is", "eli5", "concept", "under the hood", "how it works"],
    "debugging_help": ["debug", "bug", "error", "exception", "traceback", "stack trace", "crash", "not working", "fails", "broken"],
    "planning": ["plan", "roadmap", "schedule", "milestone", "timeline", "steps to", "project plan", "sprint"],
    "comparison": ["compare", "comparison", " vs ", "versus", "difference between", "pros and cons", "which is better", "alternatives"],
    "writing_editing": ["write", "edit", "proofread", "rewrite", "grammar", "essay", "blog post", "polish", "tone", "draft"],
    "learning": ["learn", "study", "teach me", "tutorial", "curriculum", "beginner", "practice", "master"],
    "data_analysis": ["data", "dataset", "analyze", "analysis", "statistics", "csv", "pandas", "chart", "correlation", "trend"],
    "system_design": ["system design", "architecture", "scalable", "scalability", "distributed", "design a system", "microservices", "load balancer"],
    "security_review": ["security", "vulnerability", "owasp", "xss", "sql injection", "authentication", "threat model", "secure", "csrf"],
    "api_design": ["api design", "rest api", "endpoint", "graphql", "openapi", "api versioning", "design an api", "grpc"],
    "decision_making": ["decide", "decision", "should i", "trade-off", "tradeoff", "choose", "dilemma", "options"],
    "meeting_summary": ["meeting", "minutes", "transcript", "action items", "standup", "notes from", "recap"],
    "performance_optimization": ["performance", "optimize", "slow", "latency", "speed up", "bottleneck", "profiling", "throughput", "faster"],
    "testing_strategy": ["test", "testing", "unit test", "integration test", "test coverage", "qa", "pytest", "e2e", "tdd"],
    "documentation": ["documentation", "docs", "readme", "docstring", "document this", "user guide", "api docs"],
    "prompt_engineering": ["prompt", "prompt engineering", "system prompt", "llm instructions", "few-shot", "chain of thought", "jailbreak"],
    "refactoring": ["refactor", "clean up", "restructure", "code smell", "technical debt", "simplify code", "dry", "legacy code"],
    "risk_assessment": ["risk", "risks", "risk assessment", "mitigation", "what could go wrong", "likelihood", "impact", "contingency"],
    "migration_planning": ["migrate", "migration", "upgrade", "move from", "port to", "cutover", "transition to", "legacy system"],
    "monitoring": ["monitoring", "observability", "metrics", "alerting", "dashboard", "logging", "prometheus", "grafana", "slo", "sli"],
    "incident_response": ["incident", "outage", "postmortem", "post-mortem", "downtime", "sev1", "on-call", "production is down", "rca"],
    "cost_optimization": ["cost", "reduce spend", "cloud bill", "budget", "expensive", "save money", "finops", "pricing", "cheaper"],
    "onboarding": ["onboarding", "onboard", "new hire", "new team member", "ramp up", "first week", "get started", "orientation"],
    "requirements_gathering": ["requirements", "user stories", "spec", "specification", "stakeholder", "acceptance criteria", "scope", "prd"],
}


def load_skills(skills_dir: Path | str = SKILLS_DIR) -> dict[str, str]:
    """Load skills/*.md into {name: markdown}."""
    d = Path(skills_dir)
    if not d.is_dir():
        return {}
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(d.glob("*.md"))}


def match_skills(text: str, skills: dict[str, str], max_skills: int = 2) -> list[str]:
    """Score skills by keyword hits (multi-word phrases weigh more); return top matches."""
    t = f" {text.lower()} "
    scores: dict[str, int] = {}
    for name, kws in KEYWORDS_MAP.items():
        if name not in skills:
            continue
        s = 0
        for kw in kws:
            kw_l = kw.lower()
            if kw_l.strip() != kw_l:  # padded keyword like " vs "
                hit = kw_l in t
            else:
                hit = re.search(r"(?<![a-z])" + re.escape(kw_l) + r"(?![a-z])", t) is not None
            if hit:
                s += 1 + kw_l.strip().count(" ")
        if s:
            scores[name] = s
    return [n for n, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:max_skills]]


# ---------------------------------------------------------------- agent
CRITIQUE_PROMPT = """You are a strict reviewer. Evaluate the ANSWER to the USER request.
Return ONLY JSON: {"verdict": "pass" or "revise", "confidence": 0.0-1.0,
"issues": ["short issue labels"], "suggestions": "how to improve"}.
Use "pass" only if the answer is correct, complete and well-grounded."""


@dataclass
class AgentResult:
    final_answer: str
    critiques: list[dict]
    matched_skills: list[str]
    tool_calls: list[dict] = field(default_factory=list)


class MetacognitiveAgent:
    def __init__(self, client: Any = None, model: str = MODEL, skills_dir: Path | str = SKILLS_DIR,
                 self_model_path: Path = SELF_MODEL_PATH):
        self._client = client
        self.model = model
        self.self_model_path = Path(self_model_path)
        self.self_model = SelfModel.load(self.self_model_path)
        self.skills = load_skills(skills_dir)
        self.history: list[dict] = []

    @property
    def client(self):
        """Lazily create the OpenAI client so offline imports/tests work without a key."""
        if self._client is None:
            from openai import OpenAI
            kwargs = {"api_key": os.getenv("OPENAI_API_KEY")}
            if os.getenv("OPENAI_BASE_URL"):
                kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")
            self._client = OpenAI(**kwargs)
        return self._client

    # -- helpers
    def _system_prompt(self, skill_names: list[str]) -> str:
        parts = ["You are a metacognitive assistant. Your current self-model:", self.self_model.summary(),
                 "Use tools when helpful (calculator for math, web_search for current facts). "
                 "Be honest about uncertainty."]
        for n in skill_names:
            parts.append(f"\n--- Skill: {n} (follow these steps) ---\n{self.skills[n]}")
        return "\n".join(parts)

    def _trim(self) -> None:
        if len(self.history) > MAX_HISTORY_MESSAGES:
            self.history = self.history[-MAX_HISTORY_MESSAGES:]

    def _chat_with_tools(self, messages: list[dict], tool_log: list[dict]) -> str:
        """Run the OpenAI tool-calling loop up to MAX_TOOL_LOOPS."""
        for _ in range(MAX_TOOL_LOOPS):
            resp = self.client.chat.completions.create(model=self.model, messages=messages,
                                                       tools=TOOLS, tool_choice="auto")
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return msg.content or ""
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                fn = TOOL_FUNCS.get(tc.function.name)
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    out = fn(**args) if fn else f"Error: unknown tool {tc.function.name}"
                except Exception as exc:  # noqa: BLE001
                    out, args = f"Error: {exc}", {}
                tool_log.append({"tool": tc.function.name, "args": args, "result": str(out)[:500]})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(out)[:4000]})
        # Limit hit: ask for a final answer without tools.
        messages.append({"role": "user", "content": "Tool limit reached. Give your best final answer now."})
        resp = self.client.chat.completions.create(model=self.model, messages=messages)
        return resp.choices[0].message.content or ""

    def _critique(self, user_input: str, answer: str) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": CRITIQUE_PROMPT},
                      {"role": "user", "content": f"USER:\n{user_input}\n\nANSWER:\n{answer}"}])
        try:
            data = json.loads(resp.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            data = {}
        verdict = "pass" if str(data.get("verdict", "")).lower() == "pass" else "revise"
        try:
            conf = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5
        return {"verdict": verdict, "confidence": conf,
                "issues": [str(i) for i in data.get("issues", []) or []],
                "suggestions": str(data.get("suggestions", ""))}

    # -- main API
    def run(self, user_input: str) -> AgentResult:
        """Generate -> Reflect -> Revise. Returns final answer, critiques, matched skills."""
        skills = match_skills(user_input, self.skills)
        tool_log: list[dict] = []
        self.history.append({"role": "user", "content": user_input})
        self._trim()
        messages = [{"role": "system", "content": self._system_prompt(skills)}, *self.history]
        answer = self._chat_with_tools(list(messages), tool_log)

        critiques: list[dict] = []
        for _ in range(MAX_REFLECTIONS):
            crit = self._critique(user_input, answer)
            critiques.append(crit)
            if crit["verdict"] == "pass":
                break
            revise_msgs = messages + [
                {"role": "assistant", "content": answer},
                {"role": "user", "content": "A reviewer critiqued your answer:\n" + json.dumps(crit)
                 + "\nRevise your answer to address it. Output only the improved answer."}]
            answer = self._chat_with_tools(revise_msgs, tool_log)

        last = critiques[-1]
        self.self_model.update(user_input, last["verdict"], last["confidence"], last["issues"], skills)
        self.self_model.save(self.self_model_path)
        self.history.append({"role": "assistant", "content": answer})
        self._trim()
        return AgentResult(answer, critiques, skills, tool_log)

    def reset(self) -> None:
        """Reset self-model and conversation."""
        self.self_model = SelfModel()
        self.self_model.save(self.self_model_path)
        self.history.clear()


def run_agent(user_input: str, agent: MetacognitiveAgent | None = None) -> AgentResult:
    """Convenience function."""
    return (agent or MetacognitiveAgent()).run(user_input)


# ---------------------------------------------------------------- CLI
def main() -> None:
    agent = MetacognitiveAgent()
    print(f"Metacognitive agent ({agent.model}). {len(agent.skills)} skills loaded. "
          "Commands: /quit /self /reset /skills")
    while True:
        try:
            text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/self":
            print(agent.self_model.summary()); continue
        if text == "/reset":
            agent.reset(); print("Self-model and conversation reset."); continue
        if text == "/skills":
            print(", ".join(agent.skills) or "(none)"); continue
        try:
            res = agent.run(text)
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}"); continue
        if res.matched_skills:
            print(f"[skills: {', '.join(res.matched_skills)}]")
        for i, c in enumerate(res.critiques, 1):
            print(f"[reflection {i}: {c['verdict']} ({c['confidence']:.2f}) {'; '.join(c['issues'])}]")
        print(f"\nAgent> {res.final_answer}")


if __name__ == "__main__":
    main()
