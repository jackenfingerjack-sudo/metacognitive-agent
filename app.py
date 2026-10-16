"""Streamlit chat UI for the metacognitive agent."""
import streamlit as st

from metacognitive_agent import MetacognitiveAgent

st.set_page_config(page_title="Metacognitive Agent", page_icon="🧠")

if "agent" not in st.session_state:
    st.session_state.agent = MetacognitiveAgent()
    st.session_state.messages = []
agent: MetacognitiveAgent = st.session_state.agent

with st.sidebar:
    st.header("Self-model")
    sm = agent.self_model
    st.metric("Confidence", f"{sm.confidence:.2f}")
    st.metric("Total tasks", sm.total_tasks)
    st.caption(sm.identity)
    with st.expander("Goals / capabilities / limitations"):
        st.write("**Goals**", sm.goals)
        st.write("**Capabilities**", sm.capabilities)
        st.write("**Limitations**", sm.limitations)
    with st.expander("Failure patterns"):
        st.json(sm.failure_patterns or {})
    with st.expander(f"Loaded skills ({len(agent.skills)})"):
        st.write(list(agent.skills))
    if st.button("Reset self-model & chat"):
        agent.reset()
        st.session_state.messages = []
        st.rerun()

st.title("🧠 Metacognitive Agent")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("meta"):
            meta = m["meta"]
            if meta["skills"]:
                st.caption("Skills: " + ", ".join(meta["skills"]))
            with st.expander(f"Critiques ({len(meta['critiques'])})"):
                for i, c in enumerate(meta["critiques"], 1):
                    st.markdown(f"**Reflection {i}:** `{c['verdict']}` (confidence {c['confidence']:.2f})")
                    st.json(c)
            if meta["tools"]:
                with st.expander("Tool calls"):
                    st.json(meta["tools"])

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Generating, reflecting, revising..."):
        try:
            res = agent.run(prompt)
            st.session_state.messages.append({"role": "assistant", "content": res.final_answer, "meta": {
                "skills": res.matched_skills, "critiques": res.critiques, "tools": res.tool_calls}})
        except Exception as exc:  # noqa: BLE001
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Error: {exc}"})
    st.rerun()
