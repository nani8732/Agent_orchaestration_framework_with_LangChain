from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from app import run_agents

app = FastAPI(title="Agent Orchestration API")

class RunRequest(BaseModel):
    task: str
    extra_instructions: Optional[str] = None


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Agent Orchestration API running"}


@app.post("/run")
def run_task(req: RunRequest):
    result = run_agents(
        user_goal=req.task,
        extra_instructions=req.extra_instructions,
    )
    return result
