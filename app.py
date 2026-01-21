import os
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS

# -------------------------------------------------
# 1) Load environment & init LLM / Embeddings
# -------------------------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in .env")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2,
)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# -------------------------------------------------
# 2) Shared Vector Memory (FAISS) – simplified
# -------------------------------------------------
FAISS_PATH = "shared_memory_faiss"

try:
    vectorstore = FAISS.load_local(
        FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True,
    )
except Exception:
    vectorstore = FAISS.from_texts(
        ["_initial_shared_memory_"],
        embedding_model,
        metadatas=[{"source": "system"}],
    )

shared_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# -------------------------------------------------
# 3) Tools (web search, calculator, weather)
# -------------------------------------------------
web_search_tool = DuckDuckGoSearchRun(name="web_search")

@tool("calculator")
def calculator(expression: str) -> str:
    """Evaluate basic arithmetic expressions like '2+3*4'."""
    try:
        allowed = "0123456789+-*/(). "
        if any(ch not in allowed for ch in expression):
            return "Only basic arithmetic is allowed."
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

@tool("weather")
def weather(location: str) -> str:
    """Return a dummy weather report for the given location."""
    return f"Mock weather for {location}: clear sky, 28°C, light breeze."


TOOLS = [web_search_tool, calculator, weather]

# -------------------------------------------------
# 4) Agent prompts
# -------------------------------------------------
RESEARCH_PROMPT = PromptTemplate.from_template(
    """
You are a **Research Agent**.
Your job is to research the user's topic in depth using tools (web_search, calculator, weather) when needed.

TOOLS:
{tools}
Available tool names:
{tool_names}

Use ReAct format:
Thought: ...
Action: <tool name>
Action Input: <tool input>
Observation: <tool result>
...
Final Answer: <your final research report>

User query:
{input}

Previous reasoning:
{agent_scratchpad}
"""
)

SUMMARY_PROMPT = PromptTemplate.from_template(
    """
You are a **Summarizer Agent**.
You receive a detailed research report and must:
- Write a concise summary.
- Provide 3–5 key bullet points.
- Keep it clear and structured.

TOOLS (for compatibility, usually not needed):
{tools}
{tool_names}

Use:
Thought: ...
Final Answer: ...

Original user request:
{input}

Research content:
{research_notes}

Previous reasoning:
{agent_scratchpad}
"""
)

EMAIL_PROMPT = PromptTemplate.from_template(
    """
You are an **Email Agent**.
Write a professional email based on the summary.

TOOLS (not needed but kept for compatibility):
{tools}
{tool_names}

Use:
Thought: ...
Final Answer: ...

Original goal:
{input}

Summary to include:
{summary_text}

Previous reasoning:
{agent_scratchpad}
"""
)

# -------------------------------------------------
# 5) Shared functions (if any)
# -------------------------------------------------


# -------------------------------------------------
# 6) Specialized agents with create_react_agent + AgentExecutor
# -------------------------------------------------
research_agent_core = create_react_agent(
    llm=llm,
    tools=TOOLS,
    prompt=RESEARCH_PROMPT,
)
research_agent = AgentExecutor(
    agent=research_agent_core,
    tools=TOOLS,
    verbose=True,
    handle_parsing_errors=True,
)

summary_agent_core = create_react_agent(
    llm=llm,
    tools=[],
    prompt=SUMMARY_PROMPT,
)
summary_agent = AgentExecutor(
    agent=summary_agent_core,
    tools=[],
    verbose=True,
    handle_parsing_errors=True,
)

email_agent_core = create_react_agent(
    llm=llm,
    tools=[],
    prompt=EMAIL_PROMPT,
)
email_agent = AgentExecutor(
    agent=email_agent_core,
    tools=[],
    verbose=True,
    handle_parsing_errors=True,
)


# -------------------------------------------------
# 7) Orchestration (Supervisor) – Pydantic input
# -------------------------------------------------
class OrchestratorInput(BaseModel):
    user_goal: str
    # generate_email flag తీసేసాం; email ఎప్పుడూ run అవుతుంది
    extra_instructions: Optional[str] = None


def orchestrate_task(payload: OrchestratorInput) -> Dict[str, Any]:
    """Supervisor: research → summary → email (always)."""
    user_goal = payload.user_goal
    extra = payload.extra_instructions or ""

    # 1) Research
    research_input = {
        "input": f"{user_goal}\nExtra instructions: {extra}",
    }
    research_result = research_agent.invoke(research_input)
    research_text = research_result.get("output", "")

    vectorstore.add_texts(
        [research_text],
        metadatas=[{"source": "research_agent", "goal": user_goal}],
    )
    vectorstore.save_local(FAISS_PATH)

    # 2) Summary
    summary_input = {
        "input": user_goal,
        "research_notes": research_text,
    }
    summary_result = summary_agent.invoke(summary_input)
    summary_text = summary_result.get("output", "")

    vectorstore.add_texts(
        [summary_text],
        metadatas=[{"source": "summary_agent", "goal": user_goal}],
    )
    vectorstore.save_local(FAISS_PATH)

    # 3) Email (always)
    email_input = {
        "input": user_goal,
        "summary_text": summary_text,
    }
    email_result = email_agent.invoke(email_input)
    email_text = email_result.get("output", "")

    vectorstore.add_texts(
        [email_text],
        metadatas=[{"source": "email_agent", "goal": user_goal}],
    )
    vectorstore.save_local(FAISS_PATH)

    return {
        "goal": user_goal,
        "research": research_text,
        "summary": summary_text,
        "email": email_text,
    }


# Convenience function (used by API & Streamlit)
def run_agents(
    user_goal: str,
    extra_instructions: Optional[str] = None,
) -> Dict[str, Any]:
    payload = OrchestratorInput(
        user_goal=user_goal,
        extra_instructions=extra_instructions,
    )
    return orchestrate_task(payload)
