import streamlit as st
from app import run_agents

st.set_page_config(
    page_title="Agent Orchestration Framework",
    layout="wide",
)

st.title("🤖 Agent Orchestration Framework")

task = st.text_area(
    "Enter your query",
    height=150,
    placeholder="Example: Analyze the impact of AI in healthcare and draft an email to my manager.",
)

if st.button("🚀 Run Orchestration"):
    if not task.strip():
        st.warning("Please enter a goal / topic.")
    else:
        with st.spinner("Agents are working..."):
            result = run_agents(
                user_goal=task,
            )

        st.subheader("🔍 Research Agent Output")
        st.write(result.get("research", ""))

        st.subheader("📝 Summary Agent Output")
        st.write(result.get("summary", ""))

        st.subheader("📧 Email Agent Output")
        st.write(result.get("email", "No email generated."))
