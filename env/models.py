"""
OpenEnv typed models — Pydantic v2 (falls back to dataclasses when pydantic is not installed).

In production (Docker / HF Space) pydantic IS installed, so the real BaseModel path runs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field

    _USING_PYDANTIC = True

    # ── Action ────────────────────────────────────────────────────────────────

    class SQLAction(BaseModel):
        """The one action an agent can take: submit a SQL SELECT query."""
        sql_query: str = Field(..., description="A SQLite SELECT (or WITH…SELECT) query.")

    # ── Observation ───────────────────────────────────────────────────────────

    class Observation(BaseModel):
        """Everything the agent can see at each timestep."""
        task_id: str
        difficulty: str = Field(..., description="easy | medium | hard")
        question: str = Field(..., description="Natural-language business question.")
        schema_info: str = Field(..., description="Database schema DDL.")
        attempt: int = Field(0, ge=0, description="0-indexed attempt counter.")
        max_attempts: int = Field(3, ge=1)
        previous_query: Optional[str] = None
        previous_result: Optional[str] = None
        previous_score: Optional[float] = Field(None, ge=0.0, le=1.0)
        hint: Optional[str] = None

    # ── Reward ────────────────────────────────────────────────────────────────

    class Reward(BaseModel):
        """Structured reward breakdown — the OpenEnv spec requires a typed Reward model."""
        value: float = Field(..., ge=0.0, le=1.0, description="Final reward for this step.")
        correctness: float = Field(0.0, ge=0.0, le=1.0)
        attempt_penalty: float = Field(0.0, ge=0.0, le=1.0)
        syntax_bonus: float = Field(0.0, ge=0.0, le=1.0)
        reason: str = ""

    # ── Step Result ───────────────────────────────────────────────────────────

    class StepResult(BaseModel):
        observation: Observation
        reward: float = Field(..., ge=0.0, le=1.0)
        reward_detail: Reward
        done: bool
        info: Dict[str, Any] = Field(default_factory=dict)

    # ── State ─────────────────────────────────────────────────────────────────

    class EnvState(BaseModel):
        task_id: str
        difficulty: str
        current_observation: Observation
        total_reward: float = 0.0
        best_reward: float = 0.0
        best_query: Optional[str] = None
        steps_taken: int = 0
        completed: bool = False
        episode_history: List[Dict[str, Any]] = Field(default_factory=list)

        def model_dump_json(self, **kw):
            import json
            return json.dumps(self.model_dump(), **kw)

    # ── Task Definition ───────────────────────────────────────────────────────

    class TaskDefinition(BaseModel):
        task_id: str
        difficulty: str
        question: str
        expected_answer: Any
        answer_type: str   # scalar_int | scalar_float | scalar_string | ordered_list | set_match
        hint: Optional[str] = None
        max_attempts: int = 3
        points: float = 1.0

    # ── Benchmark ─────────────────────────────────────────────────────────────

    class TaskResult(BaseModel):
        task_id: str
        difficulty: str
        score: float
        attempts_used: int
        final_query: Optional[str] = None
        notes: str = ""

    class BenchmarkResult(BaseModel):
        agent: str
        total_score: float
        max_score: float
        normalized_score: float
        per_task: List[TaskResult]
        easy_avg: float
        medium_avg: float
        hard_avg: float

except ModuleNotFoundError:
    # ── stdlib fallback (sandbox / CI without pydantic installed) ─────────────
    from dataclasses import dataclass, field as dc_field

    _USING_PYDANTIC = False

    def Field(default=None, **_):  # noqa: N802
        return default

    @dataclass
    class SQLAction:
        sql_query: str
        def model_dump(self): return {"sql_query": self.sql_query}

    @dataclass
    class Observation:
        task_id: str
        difficulty: str
        question: str
        schema_info: str
        attempt: int = 0
        max_attempts: int = 3
        previous_query: object = None
        previous_result: object = None
        previous_score: object = None
        hint: object = None
        def model_dump(self): return self.__dict__.copy()

    @dataclass
    class Reward:
        value: float
        correctness: float = 0.0
        attempt_penalty: float = 0.0
        syntax_bonus: float = 0.0
        reason: str = ""
        def model_dump(self): return self.__dict__.copy()

    @dataclass
    class StepResult:
        observation: Observation
        reward: float
        reward_detail: Reward
        done: bool
        info: dict = dc_field(default_factory=dict)
        def model_dump(self):
            return {"observation": self.observation.model_dump(),
                    "reward": self.reward,
                    "reward_detail": self.reward_detail.model_dump(),
                    "done": self.done, "info": self.info}

    @dataclass
    class EnvState:
        task_id: str
        difficulty: str
        current_observation: Observation
        total_reward: float = 0.0
        best_reward: float = 0.0
        best_query: object = None
        steps_taken: int = 0
        completed: bool = False
        episode_history: list = dc_field(default_factory=list)
        def model_dump(self):
            return {"task_id": self.task_id, "difficulty": self.difficulty,
                    "current_observation": self.current_observation.model_dump(),
                    "total_reward": self.total_reward, "best_reward": self.best_reward,
                    "best_query": self.best_query, "steps_taken": self.steps_taken,
                    "completed": self.completed, "episode_history": self.episode_history}
        def model_dump_json(self, indent=2):
            import json; return json.dumps(self.model_dump(), indent=indent)

    @dataclass
    class TaskDefinition:
        task_id: str
        difficulty: str
        question: str
        expected_answer: object
        answer_type: str
        hint: object = None
        max_attempts: int = 3
        points: float = 1.0

    @dataclass
    class TaskResult:
        task_id: str
        difficulty: str
        score: float
        attempts_used: int
        final_query: object = None
        notes: str = ""
        def model_dump(self): return self.__dict__.copy()

    @dataclass
    class BenchmarkResult:
        agent: str
        total_score: float
        max_score: float
        normalized_score: float
        per_task: list
        easy_avg: float
        medium_avg: float
        hard_avg: float
        def model_dump(self):
            return {"agent": self.agent, "total_score": self.total_score,
                    "max_score": self.max_score, "normalized_score": self.normalized_score,
                    "per_task": [r.model_dump() for r in self.per_task],
                    "easy_avg": self.easy_avg, "medium_avg": self.medium_avg,
                    "hard_avg": self.hard_avg}
