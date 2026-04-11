from __future__ import annotations

import threading
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel as PydanticModel

from env import SQLBusinessEnv, SQLAction
from env.tasks import TASKS

_env = SQLBusinessEnv(seed=42)
_lock = threading.Lock()

api = FastAPI()


class ResetRequest(PydanticModel):
    task_id: Optional[str] = None


class StepRequest(PydanticModel):
    sql_query: str


@api.get("/health")
def health():
    return {"status": "ok", "env": "sql-business-intelligence-v1"}


@api.post("/reset")
def reset(req: ResetRequest = None):
    task_id = (req.task_id if req else None)
    with _lock:
        try:
            obs = _env.reset(task_id=task_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    return obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__


@api.post("/step")
def step(req: StepRequest):
    with _lock:
        try:
            action = SQLAction(sql_query=req.sql_query)
            result = _env.step(action)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    reward = round(max(0.05, min(0.95, float(result.reward))), 4)

    return {
        "observation": result.observation.model_dump()
        if hasattr(result.observation, "model_dump")
        else result.observation.__dict__,
        "reward": reward,
        "reward_detail": result.reward_detail.model_dump()
        if hasattr(result.reward_detail, "model_dump")
        else result.reward_detail.__dict__,
        "done": result.done,
        "info": result.info,
    }


@api.get("/state")
@api.post("/state")
def state():
    with _lock:
        try:
            s = _env.state()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    return s.model_dump() if hasattr(s, "model_dump") else s.__dict__


@api.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"id": t.task_id, "difficulty": t.difficulty, "description": t.question[:80]}
            for t in TASKS
        ]
    }


@api.get("/")
def root():
    return JSONResponse({"status": "ok", "health": "/health", "tasks": "/tasks"})


def main():
    uvicorn.run(api, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()