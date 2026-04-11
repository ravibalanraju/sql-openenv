"""
OpenEnv typed models — Pydantic v2
All reward values are strictly between 0.0 and 1.0 (exclusive).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field

    _USING_PYDANTIC = True

    class SQLAction(BaseModel):
        sql_query: str = Field(..., description="A SQLite SELECT (or WITH…SELECT) query.")

    class Observation(BaseModel):
        task_id: str
        difficulty: str
        question: str
        schema_info: str
        attempt: int = Field(0, ge=0)
        max_attempts: int = Field(3, ge=1)
        previous_query: Optional[str] = None
        previous_result: Optional[str] = None
        previous_score: Optional[float] = None
        hint: Optional[str] = None

    class Reward(BaseModel):
        # strictly between 0 and 1 — gt/lt instead of ge/le
        value: float = Field(..., gt=0.0, lt=1.0,
                             description="Final reward strictly between 0 and 1.")
        correctness: float = Field(0.05, ge=0.0, le=1.0)
        attempt_penalty: float = Field(0.0, ge=0.0, le=1.0)
        syntax_bonus: float = Field(0.0, ge=0.0, le=1.0)
        reason: str = ""

        def model_post_init(self, __context: Any) -> None:
            # Clamp value to be strictly between 0 and 1
            object.__setattr__(self, 'value',
                               round(max(0.05, min(0.95, self.value)), 4))

    class StepResult(BaseModel):
        observation: Observation
        reward: float = Field(..., ge=0.0, le=1.0)
        reward_detail: Reward
        done: bool
        info: Dict[str, Any] = Field(default_factory=dict)

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

    class TaskDefinition(BaseModel):
        task_id: str
        difficulty: str
        question: str
        expected_answer: Any
        answer_type: str
        hint: Optional[str] = None
        max_attempts: int = 3
        points: float = 1.0

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
    from dataclasses import dataclass, field as dc_field

    _USING_PYDANTIC = False

    def Field(default=None, **_):
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
        correctness: float = 0.05
        attempt_penalty: float = 0.0
        syntax_bonus: float = 0.0
        reason: str = ""
        def __post_init__(self):
            self.value = round(max(0.05, min(0.95, self.value)), 4)
        def model_dump(self): return self.__dict__.copy()

    @dataclass
    class StepResult:
        observation: Observation
        reward: float
        reward_detail: Reward
        done: bool
        info: dict = dc_field(default_factory=dict)

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