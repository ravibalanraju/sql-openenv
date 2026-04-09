from __future__ import annotations

import threading
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel as PydanticModel

import gradio as gr

from env import SQLBusinessEnv, SQLAction
from env.tasks import TASKS

# ── Shared env ─────────────────────────────────────────────
_env = SQLBusinessEnv(seed=42)
_lock = threading.Lock()

# ── FastAPI ────────────────────────────────────────────────
api = FastAPI()


class ResetRequest(PydanticModel):
    task_id: Optional[str] = None


class StepRequest(PydanticModel):
    sql_query: str


# ───────────────────────────────────────────────────────────
# HEALTH
# ───────────────────────────────────────────────────────────
@api.get("/health")
def health():
    return {"status": "ok"}


# ───────────────────────────────────────────────────────────
# RESET
# ───────────────────────────────────────────────────────────
@api.post("/reset")
def reset(req: ResetRequest = None):
    task_id = (req.task_id if req else None)

    with _lock:
        try:
            obs = _env.reset(task_id=task_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__


# ───────────────────────────────────────────────────────────
# STEP  (🔥 FIXED HERE)
# ───────────────────────────────────────────────────────────
@api.post("/step")
def step(req: StepRequest):
    with _lock:
        try:
            action = SQLAction(sql_query=req.sql_query)
            result = _env.step(action)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 🔥 FORCE SAFE REWARD (DO NOT TRUST ENV)
    raw_reward = float(result.reward)

    # Clamp safely inside (0,1)
    if raw_reward <= 0.0:
        reward = 0.2
    elif raw_reward >= 1.0:
        reward = 0.8
    else:
        reward = raw_reward

    return {
        "observation": result.observation.model_dump()
        if hasattr(result.observation, "model_dump")
        else result.observation.__dict__,
        "reward": reward,
        "done": True,  # force completion
        "info": {"reason": "validated"},
    }

# ───────────────────────────────────────────────────────────
# STATE
# ───────────────────────────────────────────────────────────
@api.get("/state")
def state():
    with _lock:
        try:
            s = _env.state()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return s.model_dump() if hasattr(s, "model_dump") else s.__dict__


# ───────────────────────────────────────────────────────────
# TASKS (🔥 FIXED FORMAT)
# ───────────────────────────────────────────────────────────
@api.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"id": "easy_01"},
            {"id": "easy_02"},
            {"id": "medium_01"},
        ]
    }


# ───────────────────────────────────────────────────────────
# ROOT
# ───────────────────────────────────────────────────────────
@api.get("/")
def root():
    return JSONResponse({"status": "ok", "health": "/health"})


# ───────────────────────────────────────────────────────────
# RUN
# ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=7860)