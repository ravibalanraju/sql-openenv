"""
SQLBusinessEnv — OpenEnv-compliant SQL Business Intelligence environment.

Full spec:
  reset(task_id?)  →  Observation
  step(action)     →  StepResult
  state()          →  EnvState
  available_tasks  →  list[str]
  run_benchmark    →  BenchmarkResult
"""
from __future__ import annotations

import random
import sqlite3
from typing import List, Optional

from env.database import SCHEMA_INFO, build_connection, safe_execute
from env.graders import compute_reward
from env.models import (
    BenchmarkResult,
    EnvState,
    Observation,
    Reward,
    SQLAction,
    StepResult,
    TaskDefinition,
    TaskResult,
)
from env.tasks import TASK_MAP, TASKS

# ── Constants ──────────────────────────────────────────────────────────────────
ENV_ID      = "sql-business-intelligence-v1"
ENV_VERSION = "1.0.0"


class SQLBusinessEnv:
    """
    Real-world OpenEnv environment: SQL Business Intelligence Agent.

    The agent receives a natural-language business question and must write
    a SQL SELECT query against a seeded SQLite retail database to answer it.

    Episode lifecycle
    ─────────────────
    1. reset(task_id?)       – loads task, returns first Observation
    2. step(SQLAction) × N   – agent submits SQL; gets Observation + Reward
    3. done=True when:
         a) correctness == 1.0 (solved), or
         b) attempts exhausted
    """

    # Reward shaping constants (exposed so they appear in openenv.yaml)
    ATTEMPT_PENALTY: float = 0.10   # deducted per attempt after the first
    SYNTAX_BONUS: float    = 0.05   # awarded when SQL runs without error

    def __init__(
        self,
        db_path: str = ":memory:",
        seed: Optional[int] = 42,
    ):
        self._conn: sqlite3.Connection = build_connection(db_path)
        self._rng  = random.Random(seed)
        self._task: Optional[TaskDefinition] = None
        self._state: Optional[EnvState] = None

    # ══ Public API ════════════════════════════════════════════════════════════

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """
        Start a new episode.

        Parameters
        ----------
        task_id : str | None
            Explicit task to load.  Pass None to sample randomly.

        Returns
        -------
        Observation
        """
        if task_id is not None:
            if task_id not in TASK_MAP:
                raise ValueError(
                    f"Unknown task_id '{task_id}'. "
                    f"Available: {sorted(TASK_MAP)}"
                )
            task = TASK_MAP[task_id]
        else:
            task = self._rng.choice(TASKS)

        self._task = task
        obs = self._make_obs(attempt=0)
        self._state = EnvState(
            task_id=task.task_id,
            difficulty=task.difficulty,
            current_observation=obs,
        )
        return obs

    def step(self, action: SQLAction) -> StepResult:
        """
        Submit a SQL query.

        Parameters
        ----------
        action : SQLAction

        Returns
        -------
        StepResult
            observation, reward (float), reward_detail (Reward), done, info
        """
        self._require_reset()
        if self._state.completed:
            raise RuntimeError("Episode is already done — call reset() to start a new one.")

        task   = self._task
        attempt = self._state.steps_taken   # 0-indexed

        # Execute query
        rows, _cols, error = safe_execute(self._conn, action.sql_query)

        # Structured reward
        reward_obj: Reward = compute_reward(
            rows=rows,
            error=error,
            expected=task.expected_answer,
            answer_type=task.answer_type,
            attempt=attempt,
            max_attempts=task.max_attempts,
            attempt_penalty=self.ATTEMPT_PENALTY,
        )

        # Update state
        self._state.steps_taken += 1
        self._state.total_reward += reward_obj.value
        if reward_obj.value > self._state.best_reward:
            self._state.best_reward = reward_obj.value
            self._state.best_query  = action.sql_query

        # Episode termination
        perfect   = reward_obj.correctness >= 1.0
        exhausted = self._state.steps_taken >= task.max_attempts
        done      = perfect or exhausted
        self._state.completed = done

        # History (useful for debugging / analysis)
        self._state.episode_history.append({
            "attempt"    : attempt,
            "query"      : action.sql_query,
            "correctness": reward_obj.correctness,
            "reward"     : reward_obj.value,
            "reason"     : reward_obj.reason,
            "done"       : done,
        })

        # Next observation
        result_snippet = _fmt_result(rows, error)
        next_obs = self._make_obs(
            attempt      = self._state.steps_taken,
            prev_query   = action.sql_query,
            prev_result  = result_snippet,
            prev_score   = reward_obj.value,
        )
        self._state.current_observation = next_obs

        info = {
            "correctness"      : reward_obj.correctness,
            "syntax_bonus"     : reward_obj.syntax_bonus,
            "attempt_penalty"  : reward_obj.attempt_penalty,
            "reason"           : reward_obj.reason,
            "error"            : error,
            "attempts_remaining": max(0, task.max_attempts - self._state.steps_taken),
        }

        return StepResult(
            observation  = next_obs,
            reward       = reward_obj.value,
            reward_detail= reward_obj,
            done         = done,
            info         = info,
        )

    def state(self) -> EnvState:
        """Return a full snapshot of the current episode state."""
        self._require_reset()
        return self._state

    def available_tasks(self) -> List[str]:
        """Return all task IDs."""
        return [t.task_id for t in TASKS]

    # ══ Benchmark ════════════════════════════════════════════════════════════

    def run_benchmark(
        self,
        agent_fn,
        verbose: bool = False,
    ) -> BenchmarkResult:
        """
        Run agent_fn against every task and return aggregate scores.

        Parameters
        ----------
        agent_fn : Callable[[Observation], SQLAction]
        verbose  : print per-task lines

        Returns
        -------
        BenchmarkResult
        """
        results: list[TaskResult] = []

        for task in TASKS:
            obs  = self.reset(task.task_id)
            done = False

            while not done:
                action = agent_fn(obs)
                step   = self.step(action)
                obs    = step.observation
                done   = step.done

            s = self._state
            tr = TaskResult(
                task_id     = task.task_id,
                difficulty  = task.difficulty,
                score       = round(s.best_reward, 4),
                attempts_used = s.steps_taken,
                final_query = s.best_query,
                notes       = s.episode_history[-1]["reason"] if s.episode_history else "",
            )
            results.append(tr)
            if verbose:
                bar = "█" * int(tr.score * 10) + "░" * (10 - int(tr.score * 10))
                print(f"  [{tr.task_id:12s}] [{bar}] {tr.score:.4f}  ({tr.attempts_used} attempts)")

        total  = sum(r.score for r in results)
        max_s  = float(len(results))
        easy   = [r.score for r in results if r.difficulty == "easy"]
        medium = [r.score for r in results if r.difficulty == "medium"]
        hard   = [r.score for r in results if r.difficulty == "hard"]

        return BenchmarkResult(
            agent           = "unknown",
            total_score     = round(total, 4),
            max_score       = max_s,
            normalized_score= round(total / max_s, 4),
            per_task        = results,
            easy_avg        = round(sum(easy)   / len(easy),   4) if easy   else 0.0,
            medium_avg      = round(sum(medium) / len(medium), 4) if medium else 0.0,
            hard_avg        = round(sum(hard)   / len(hard),   4) if hard   else 0.0,
        )

    # ══ Private helpers ═══════════════════════════════════════════════════════

    def _require_reset(self) -> None:
        if self._state is None:
            raise RuntimeError("Call reset() before interacting with the environment.")

    def _make_obs(
        self,
        attempt: int,
        prev_query : Optional[str]   = None,
        prev_result: Optional[str]   = None,
        prev_score : Optional[float] = None,
    ) -> Observation:
        task = self._task
        return Observation(
            task_id        = task.task_id,
            difficulty     = task.difficulty,
            question       = task.question,
            schema_info    = SCHEMA_INFO,
            attempt        = attempt,
            max_attempts   = task.max_attempts,
            previous_query = prev_query,
            previous_result= prev_result,
            previous_score = prev_score,
            hint           = task.hint if attempt > 0 else None,  # hint revealed after first attempt
        )


# ── Formatting ─────────────────────────────────────────────────────────────────

def _fmt_result(rows, error, max_rows: int = 5) -> str:
    if error:
        return f"ERROR: {error}"
    if rows is None:
        return "No result."
    if not rows:
        return "Query returned 0 rows."
    lines = [str(r) for r in rows[:max_rows]]
    suffix = f"\n... +{len(rows) - max_rows} rows" if len(rows) > max_rows else ""
    return "\n".join(lines) + suffix
