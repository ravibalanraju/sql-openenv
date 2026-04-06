"""
app.py — SQL Business Intelligence OpenEnv server.

Serves TWO things on port 7860:
  1. REST API  — /reset  /step  /state  /tasks  /health
     Required by the OpenEnv validator (validate-submission.sh pings POST /reset)
  2. Gradio UI — mounted at /ui
     Required for HuggingFace Spaces visual demo

Stack: FastAPI + Gradio mounted via gr.mount_gradio_app()

Run:  python app.py
"""
from __future__ import annotations

import json
import textwrap
import threading
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel as PydanticModel

import gradio as gr

from env import SQLBusinessEnv, SQLAction
from env.tasks import TASKS
from env.database import SCHEMA_INFO

# ── Shared env (one instance per worker; fine for demo) ───────────────────────
_env = SQLBusinessEnv(seed=42)
_lock = threading.Lock()

# ══════════════════════════════════════════════════════════════════════════════
#  REST API  (required by validate-submission.sh)
# ══════════════════════════════════════════════════════════════════════════════

api = FastAPI(
    title="SQL Business Intelligence — OpenEnv",
    description="OpenEnv REST API: /reset  /step  /state  /tasks  /health",
    version="1.0.0",
)


# ── Request / response schemas ─────────────────────────────────────────────────

class ResetRequest(PydanticModel):
    task_id: Optional[str] = None


class StepRequest(PydanticModel):
    sql_query: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@api.get("/health")
def health():
    """Liveness probe."""
    return {"status": "ok", "env": "sql-business-intelligence-v1"}


@api.post("/reset")
def reset(req: ResetRequest = None):
    """
    Start a new episode.

    Body (optional JSON):
      { "task_id": "medium_01" }   — specific task
      {}                           — random task

    Returns: Observation (JSON)
    """
    task_id = (req.task_id if req else None)
    with _lock:
        try:
            obs = _env.reset(task_id=task_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    return obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__


@api.post("/step")
def step(req: StepRequest):
    """
    Submit a SQL query.

    Body:
      { "sql_query": "SELECT COUNT(*) FROM customers WHERE state='NY'" }

    Returns: StepResult (observation, reward, done, info)
    """
    with _lock:
        try:
            action = SQLAction(sql_query=req.sql_query)
            result = _env.step(action)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return result.model_dump() if hasattr(result, "model_dump") else {
        "observation" : result.observation.model_dump() if hasattr(result.observation, "model_dump") else result.observation.__dict__,
        "reward"      : result.reward,
        "reward_detail": result.reward_detail.model_dump() if hasattr(result.reward_detail, "model_dump") else result.reward_detail.__dict__,
        "done"        : result.done,
        "info"        : result.info,
    }


@api.get("/state")
@api.post("/state")
def state():
    """Return full episode state snapshot."""
    with _lock:
        try:
            s = _env.state()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    return s.model_dump() if hasattr(s, "model_dump") else s.__dict__


@api.get("/tasks")
def tasks():
    """List all available tasks with metadata."""
    return [
        {
            "task_id"   : t.task_id,
            "difficulty": t.difficulty,
            "question"  : t.question,
            "max_attempts": t.max_attempts,
        }
        for t in TASKS
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  GRADIO UI  (HuggingFace Spaces visual interface)
# ══════════════════════════════════════════════════════════════════════════════

_ui_env = SQLBusinessEnv(seed=42)
_ui_obs = None


def ui_start_task(task_id: str):
    global _ui_obs
    _ui_obs = _ui_env.reset(task_id=task_id)
    return _fmt_obs(_ui_obs), "", "", ""


def ui_submit(sql: str):
    global _ui_obs
    if _ui_obs is None:
        return "⚠️ Load a task first.", "", "", ""
    action = SQLAction(sql_query=sql.strip())
    result = _ui_env.step(action)
    _ui_obs = result.observation

    done_msg = (
        "✅ Solved!" if result.done and result.reward >= 0.9
        else "⏰ Attempts exhausted." if result.done
        else f"🔄 Attempt {_ui_obs.attempt}/{_ui_obs.max_attempts}"
    )
    rd = result.reward_detail
    reward_md = textwrap.dedent(f"""
        **Reward:** `{result.reward:.4f}`  {'█' * int(result.reward * 10)}{'░' * (10 - int(result.reward * 10))}

        | Component | Value |
        |---|---|
        | Correctness | `{rd.correctness:.4f}` |
        | Syntax bonus | `+{rd.syntax_bonus:.4f}` |
        | Attempt penalty | `-{rd.attempt_penalty:.4f}` |

        **Grader:** {result.info['reason']}
        **Status:** {done_msg}
    """).strip()

    try:
        state_json = _ui_env.state().model_dump_json(indent=2)
    except Exception:
        state_json = "{}"

    return _fmt_obs(_ui_obs), reward_md, _ui_obs.previous_result or "", state_json


def ui_benchmark():
    from inference import RuleBasedAgent
    bench = SQLBusinessEnv(seed=42)
    bm = bench.run_benchmark(RuleBasedAgent(), verbose=False)
    rows = "\n".join(
        f"- {'🟢' if r.difficulty=='easy' else '🟡' if r.difficulty=='medium' else '🔴'} "
        f"`{r.task_id}` {'█'*int(r.score*10)}{'░'*(10-int(r.score*10))} `{r.score:.4f}`"
        for r in bm.per_task
    )
    return textwrap.dedent(f"""
        ## Rule-based baseline results
        | | Score |
        |---|---|
        | Easy avg | `{bm.easy_avg:.4f}` |
        | Medium avg | `{bm.medium_avg:.4f}` |
        | Hard avg | `{bm.hard_avg:.4f}` |
        | **Normalized** | **`{bm.normalized_score:.4f}`** |

        {rows}
    """).strip()


def _fmt_obs(obs) -> str:
    hist = ""
    if obs.previous_query:
        hist = f"\n\n---\n**Prev SQL:** `{obs.previous_query[:80]}`\n**Score:** `{obs.previous_score:.3f}`"
    hint_block = f"\n\n> 💡 **Hint:** {obs.hint}" if obs.hint else ""
    return textwrap.dedent(f"""
        ### `{obs.task_id}` — {obs.difficulty.upper()}
        **Attempt** {obs.attempt} / {obs.max_attempts}

        **Question**
        > {obs.question}

        **Schema**
        ```
        {obs.schema_info}
        ```{hist}{hint_block}
    """).strip()


CHOICES = [
    (f"[{t.difficulty.upper()}] {t.task_id} — {t.question[:65]}…", t.task_id)
    for t in TASKS
]

with gr.Blocks(title="SQL BI — OpenEnv") as gradio_ui:
    gr.Markdown("# 🗄️ SQL Business Intelligence — OpenEnv\n"
                "Answer business questions with SQL. Scored 0.0–1.0 with partial credit.")
    with gr.Row():
        with gr.Column(scale=1):
            task_dd   = gr.Dropdown(choices=CHOICES, value=CHOICES[0][1], label="Task")
            start_btn = gr.Button("▶ Load Task", variant="primary")
            sql_box   = gr.Code(language="sql", label="Your SQL", lines=8,
                                value="SELECT COUNT(*) FROM customers WHERE state = 'NY'")
            sub_btn   = gr.Button("⚡ Submit", variant="primary")
            prev_box  = gr.Textbox(label="Previous result", lines=3, interactive=False)
        with gr.Column(scale=1):
            obs_md    = gr.Markdown("*Load a task to begin.*")
            reward_md = gr.Markdown()
    with gr.Accordion("Episode state (JSON)", open=False):
        state_box = gr.Code(language="json", interactive=False)
    with gr.Accordion("Run full benchmark", open=False):
        gr.Button("Run rule-based baseline").click(fn=ui_benchmark, outputs=gr.Markdown())
    gr.Markdown("---\n**REST API:** `POST /reset` · `POST /step` · `GET /state` · `GET /tasks`")

    start_btn.click(ui_start_task, [task_dd], [obs_md, reward_md, prev_box, state_box])
    sub_btn.click(ui_submit, [sql_box], [obs_md, reward_md, prev_box, state_box])
    task_dd.change(ui_start_task, [task_dd], [obs_md, reward_md, prev_box, state_box])
    gradio_ui.load(lambda: ui_start_task(CHOICES[0][1]),
                   outputs=[obs_md, reward_md, prev_box, state_box])


# ── Mount Gradio onto FastAPI ──────────────────────────────────────────────────
app = gr.mount_gradio_app(api, gradio_ui, path="/ui")

# Also serve Gradio at root so HF Spaces shows the UI by default
app = gr.mount_gradio_app(app, gradio_ui, path="/")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    port = int(os.getenv("GRADIO_SERVER_PORT", 7860))
    print(f"\n  REST API  →  http://localhost:{port}/reset  (POST)")
    print(f"  Gradio UI →  http://localhost:{port}/\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
