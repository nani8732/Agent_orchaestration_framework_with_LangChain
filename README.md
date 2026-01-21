🤖 Agent Orchestration Framework 

An AI-powered multi-agent orchestration system built using LangChain, Google Gemini, FastAPI, and Streamlit.
This project demonstrates how multiple specialized agents (Research, Summary, Email) can collaborate to solve a task using a supervisor-style workflow.

🚀 Features

🔍 Research Agent – Gathers information using web search & tools

🧠 Summary Agent – Produces structured summaries

📧 Email Agent – Drafts professional emails from summaries

🧠 Shared Memory (FAISS) – Stores agent outputs for reuse

🌐 FastAPI Backend – REST API interface

🖥 Streamlit UI – Simple frontend for interaction

🔑 Google Gemini (Generative AI) integration

📁 Project Structure
.
├── app.py                # Core agent orchestration logic
├── api.py                # FastAPI backend
├── streamlit_app.py      # Streamlit frontend
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
├── shared_memory_faiss/  # Vector store (auto-created)
└── README.md
⚙️ Tech Stack

Python 3.10+

LangChain

Google Gemini (Generative AI)

FastAPI

Streamlit

FAISS (Vector Database)

DuckDuckGo Search Tool

🔐 Environment Setup
1️⃣ Create .env file
GOOGLE_API_KEY=your_google_gemini_api_key

⚠️ Make sure your API key has access to Gemini models.

📦 Install Dependencies
pip install -r requirements.txt
▶️ Running the Application
✅ Option 1: Run Streamlit App (Recommended)
streamlit run streamlit_app.py

Then open:

http://localhost:8501
✅ Option 2: Run FastAPI Backend
uvicorn api:app --reload

API will be available at:

http://localhost:8000
