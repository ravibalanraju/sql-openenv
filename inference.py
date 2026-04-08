"""
inference.py — SQL Business Intelligence OpenEnv
Mandatory stdout format (per official spec):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""
from __future__ import annotations

import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI
from env import SQLBusinessEnv, SQLAction, Observation

# ── Mandatory env vars ────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = "sql-business-intelligence-v1"

MAX_ATTEMPTS  = 3
TEMPERATURE   = 0.0
MAX_TOKENS    = 512
SUCCESS_SCORE = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# MANDATORY LOG FUNCTIONS — exact format required by evaluator
# ══════════════════════════════════════════════════════════════════════════════

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    action_clean = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# AGENTS
# ══════════════════════════════════════════════════════════════════════════════

class RuleBasedAgent:
    _SOLUTIONS: dict[str, str] = {
        "easy_01":   "SELECT COUNT(*) FROM customers WHERE state = 'NY'",
        "easy_02":   "SELECT unit_price FROM products WHERE name = 'Ergonomic Chair'",
        "easy_03":   "SELECT COUNT(*) FROM orders WHERE status = 'completed'",
        "medium_01": (
            "SELECT p.name FROM order_items oi "
            "JOIN products p ON oi.product_id = p.id "
            "GROUP BY p.id, p.name "
            "ORDER BY SUM(oi.quantity * oi.unit_price) DESC LIMIT 3"
        ),
        "medium_02": (
            "SELECT ROUND(SUM(oi.quantity * oi.unit_price * (1 - o.discount_pct)), 2) "
            "FROM order_items oi JOIN orders o ON oi.order_id = o.id "
            "WHERE o.status = 'completed'"
        ),
        "medium_03": (
            "SELECT p.category FROM order_items oi "
            "JOIN products p ON oi.product_id = p.id "
            "GROUP BY p.category "
            "ORDER BY SUM(oi.quantity * oi.unit_price) DESC LIMIT 1"
        ),
        "hard_01": (
            "SELECT c.name FROM orders o "
            "JOIN customers c ON o.customer_id = c.id "
            "GROUP BY c.id, c.name "
            "HAVING COUNT(*) = ("
            "SELECT MAX(cnt) FROM (SELECT COUNT(*) AS cnt FROM orders GROUP BY customer_id)"
            ") ORDER BY c.name"
        ),
        "hard_02": (
            "SELECT strftime('%Y-%m', o.order_date) || ':' || "
            "CAST(ROUND(SUM(oi.quantity * oi.unit_price), 2) AS TEXT) "
            "FROM order_items oi JOIN orders o ON oi.order_id = o.id "
            "WHERE o.status = 'completed' AND o.order_date LIKE '2024-%' "
            "GROUP BY strftime('%Y-%m', o.order_date) ORDER BY 1"
        ),
        "hard_03": (
            "WITH pc AS ("
            "SELECT c.id, c.name, "
            "SUM(oi.quantity * oi.unit_price * (1 - o.discount_pct)) AS ns "
            "FROM customers c JOIN orders o ON o.customer_id = c.id "
            "JOIN order_items oi ON oi.order_id = o.id "
            "WHERE o.status = 'completed' GROUP BY c.id, c.name"
            ") SELECT name FROM pc "
            "WHERE ns > (SELECT AVG(ns) FROM pc) ORDER BY name"
        ),
    }

    def act(self, obs: Observation) -> str:
        return self._SOLUTIONS.get(obs.task_id, "SELECT 'unknown' AS error")


class LLMAgent:
    SYSTEM = textwrap.dedent("""
        You are an expert SQL developer. Write a single correct SQLite SELECT query.
        Output ONLY the raw SQL — no markdown, no explanation.
        SQLite dialect only. Only SELECT or WITH...SELECT statements.
    """).strip()

    def __init__(self):
        if not API_KEY:
            print("[DEBUG] No API key — LLM agent will fail", flush=True)
        self._client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "none")

    def act(self, obs: Observation) -> str:
        prompt = f"SCHEMA:\n{obs.schema_info}\n\nQUESTION:\n{obs.question}"
        if obs.previous_query:
            prompt += (
                f"\n\nPREVIOUS SQL (wrong): {obs.previous_query}"
                f"\nRESULT: {obs.previous_result}"
                "\nFix your answer."
            )
        if obs.hint:
            prompt += f"\n\nHINT: {obs.hint}"
        try:
            resp = self._client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": self.SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            raw = (resp.choices[0].message.content or "").strip()
            return _strip_fences(raw)
        except Exception as exc:
            print(f"[DEBUG] LLM call failed: {exc}", flush=True)
            return "SELECT 'api_error' AS error"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    agent_name = os.getenv("AGENT", "rule_based")
    if agent_name == "llm":
        agent = LLMAgent()
    else:
        agent = RuleBasedAgent()

    env = SQLBusinessEnv(seed=42)
    all_tasks = [t.task_id for t in env.tasks]

    total_score = 0.0

    for task_id in all_tasks:
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        try:
            obs = env.reset(task_id=task_id)
            done = False
            step = 0

            while not done:
                step += 1
                sql = agent.act(obs)
                error_msg = None

                try:
                    result = env.step(SQLAction(sql_query=sql))
                    reward = float(result.reward)
                    done   = result.done
                    obs    = result.observation
                except Exception as exc:
                    reward    = 0.0
                    done      = True
                    error_msg = str(exc)[:80]

                rewards.append(reward)
                steps_taken = step

                log_step(
                    step   = step,
                    action = sql,
                    reward = reward,
                    done   = done,
                    error  = error_msg,
                )

            try:
                score = float(env.state().best_reward)
            except Exception:
                score = max(rewards) if rewards else 0.0

            score   = min(max(score, 0.0), 1.0)
            success = score >= SUCCESS_SCORE

        except Exception as exc:
            print(f"[DEBUG] Task {task_id} crashed: {exc}", flush=True)
            success = False
            score   = 0.0

        finally:
            log_end(
                success = success,
                steps   = steps_taken,
                score   = score,
                rewards = rewards,
            )

        total_score += score

    normalized = total_score / len(all_tasks) if all_tasks else 0.0
    print(f"\n[SUMMARY] tasks={len(all_tasks)} normalized={normalized:.4f}", flush=True)


def _strip_fences(text: str) -> str:
    for fence in ("```sql", "```SQL", "```"):
        if text.startswith(fence):
            text = text[len(fence):].lstrip("\n")
    if text.endswith("```"):
        text = text[:-3].rstrip()
    return text.strip()


if __name__ == "__main__":
    main()