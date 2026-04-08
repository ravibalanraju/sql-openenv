"""
inference.py — Baseline agents and benchmark runner.

Mandatory env vars (per OpenEnv spec):
  API_BASE_URL  The LLM API endpoint (e.g. https://router.huggingface.co/v1)
  MODEL_NAME    Model identifier (e.g. Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN      HuggingFace API key (also accepted: API_KEY)

Usage:
  python inference.py --agent rule_based --verbose
  python inference.py --agent random
  python inference.py --agent llm --verbose
  python inference.py --agent llm --task hard_03
  python inference.py --agent llm --output results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from typing import Optional

from env import SQLBusinessEnv, SQLAction, Observation

# ── Mandatory credentials (per provided inference spec) ──────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

MAX_TOKENS  = 512
TEMPERATURE = 0.0   # deterministic for reproducibility


# ══════════════════════════════════════════════════════════════════════════════
# AGENTS
# ══════════════════════════════════════════════════════════════════════════════

class RuleBasedAgent:
    """
    Hard-coded SQL solutions for every task.
    Deterministic upper-bound baseline — scores 1.0 on all tasks.
    No API key required.
    """
    _SOLUTIONS: dict[str, str] = {
        "easy_01": "SELECT COUNT(*) FROM customers WHERE state = 'NY'",
        "easy_02": "SELECT unit_price FROM products WHERE name = 'Ergonomic Chair'",
        "easy_03": "SELECT COUNT(*) FROM orders WHERE status = 'completed'",
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
            " SELECT MAX(cnt) FROM (SELECT COUNT(*) AS cnt FROM orders GROUP BY customer_id)"
            ") ORDER BY c.name"
        ),
        "hard_02": (
            "SELECT strftime('%Y-%m', o.order_date) || ':' || "
            " CAST(ROUND(SUM(oi.quantity * oi.unit_price), 2) AS TEXT) "
            "FROM order_items oi JOIN orders o ON oi.order_id = o.id "
            "WHERE o.status = 'completed' AND o.order_date LIKE '2024-%' "
            "GROUP BY strftime('%Y-%m', o.order_date) ORDER BY 1"
        ),
        "hard_03": (
            "WITH pc AS ("
            " SELECT c.id, c.name, "
            " SUM(oi.quantity * oi.unit_price * (1 - o.discount_pct)) AS ns "
            " FROM customers c JOIN orders o ON o.customer_id = c.id "
            " JOIN order_items oi ON oi.order_id = o.id "
            " WHERE o.status = 'completed' GROUP BY c.id, c.name"
            ") SELECT name FROM pc "
            "WHERE ns > (SELECT AVG(ns) FROM pc) ORDER BY name"
        ),
    }

    def __call__(self, obs: Observation) -> SQLAction:
        sql = self._SOLUTIONS.get(obs.task_id, "SELECT 'unknown task' AS error")
        return SQLAction(sql_query=sql)


class RandomAgent:
    """Submits random (almost always wrong) SQL — floor baseline."""
    _POOL = [
        "SELECT 42",
        "SELECT COUNT(*) FROM customers",
        "SELECT name FROM products LIMIT 1",
        "SELECT id FROM orders WHERE id = 1",
        "SELECT AVG(unit_price) FROM products",
    ]

    def __init__(self, seed: int = 42):
        import random
        self._rng = random.Random(seed)

    def __call__(self, obs: Observation) -> SQLAction:
        return SQLAction(sql_query=self._rng.choice(self._POOL))


class LLMAgent:
    """
    Uses the OpenAI-compatible API client with the mandatory env vars:
    API_BASE_URL, HF_TOKEN (or API_KEY), MODEL_NAME.
    """

    SYSTEM_PROMPT = textwrap.dedent("""
        You are an expert SQL developer helping answer business intelligence questions.
        You have access to a SQLite retail database.

        Your job: write a single, correct SQLite SELECT query that answers the question.

        Rules:
        - Output ONLY the raw SQL query. No markdown, no explanation, no comments.
        - SQLite dialect only (use strftime for dates, CAST for type conversion).
        - Only SELECT or WITH...SELECT statements (no INSERT/UPDATE/DELETE/DROP).
        - The query result must directly answer the question with the right columns and aggregation.
        - If a previous attempt was wrong, fix your logic — do not repeat the same mistake.
    """).strip()

    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("openai package not installed. Run: pip install openai")

        if not API_KEY:
            sys.exit(
                "No API key found.\n"
                "Set HF_TOKEN or API_KEY environment variable.\n"
                "Example: export HF_TOKEN=hf_..."
            )

        self._client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        self._model  = MODEL_NAME

    def __call__(self, obs: Observation) -> SQLAction:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": self._build_user_message(obs)},
        ]

        try:
            completion = self._client.chat.completions.create(
                model       = self._model,
                messages    = messages,
                temperature = TEMPERATURE,
                max_tokens  = MAX_TOKENS,
                stream      = False,
            )
            raw = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [LLMAgent] API call failed: {exc}. Using fallback.")
            raw = "SELECT 'api_error' AS error"

        return SQLAction(sql_query=_strip_fences(raw.strip()))

    def _build_user_message(self, obs: Observation) -> str:
        parts = [
            f"DATABASE SCHEMA\n{obs.schema_info}",
            f"\nQUESTION\n{obs.question}",
        ]
        if obs.previous_query:
            parts.append(
                f"\nPREVIOUS ATTEMPT (attempt {obs.attempt})\n"
                f"SQL    : {obs.previous_query}\n"
                f"Result : {obs.previous_result}\n"
                f"Score  : {obs.previous_score:.3f}\n"
                "\nYour previous answer was wrong. Analyse the result and fix it."
            )
        if obs.hint:
            parts.append(f"\nHINT: {obs.hint}")
        return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# MANDATORY STRUCTURED LOGGING — [START] [STEP] [END]
# Required by Scaler evaluator — any deviation causes scoring failure
# ══════════════════════════════════════════════════════════════════════════════

def log_start(task_id: str, difficulty: str, question: str):
    print(json.dumps({
        "event":      "START",
        "task_id":    task_id,
        "difficulty": difficulty,
        "question":   question,
    }))
    sys.stdout.flush()


def log_step(task_id: str, attempt: int, sql_query: str, reward: float,
             done: bool, info: dict):
    print(json.dumps({
        "event":     "STEP",
        "task_id":   task_id,
        "attempt":   attempt,
        "sql_query": sql_query,
        "reward":    round(reward, 4),
        "done":      done,
        "info":      info,
    }))
    sys.stdout.flush()


def log_end(task_id: str, best_reward: float, attempts_used: int):
    print(json.dumps({
        "event":         "END",
        "task_id":       task_id,
        "best_reward":   round(best_reward, 4),
        "attempts_used": attempts_used,
    }))
    sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_single_task(
    env: SQLBusinessEnv,
    agent,
    task_id: str,
    verbose: bool = False,
) -> float:
    obs = env.reset(task_id=task_id)

    # Mandatory [START] log
    log_start(obs.task_id, obs.difficulty, obs.question)

    if verbose:
        print(f"\n{'─'*64}")
        print(f"  Task: {task_id} | difficulty: {obs.difficulty}")
        print(f"  Question: {obs.question}\n")

    done = False
    attempts_used = 0

    while not done:
        action = agent(obs)
        result = env.step(action)
        attempts_used += 1

        # Mandatory [STEP] log
        log_step(
            task_id    = obs.task_id,
            attempt    = obs.attempt + 1,
            sql_query  = action.sql_query,
            reward     = result.reward,
            done       = result.done,
            info       = result.info,
        )

        if verbose:
            print(f"  Attempt {obs.attempt + 1}/{obs.max_attempts}")
            print(f"  SQL    : {action.sql_query[:100]}")
            print(f"  Reward : {result.reward:.4f}  correctness={result.info.get('correctness', 0):.2f}")
            print(f"  Reason : {result.info.get('reason', '')}\n")

        obs  = result.observation
        done = result.done

    best = env.state().best_reward

    # Mandatory [END] log
    log_end(obs.task_id, best, attempts_used)

    if verbose:
        print(f"  ► Best reward: {best:.4f}")

    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SQL BI OpenEnv — Benchmark Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--agent", choices=["rule_based", "random", "llm"],
        default="rule_based",
        help=(
            "rule_based : deterministic optimal SQL (no API key)\n"
            "random     : random SQL (floor baseline)\n"
            "llm        : uses API_BASE_URL + HF_TOKEN + MODEL_NAME"
        ),
    )
    parser.add_argument("--task",    default=None,  help="Run one task only (e.g. hard_01)")
    parser.add_argument("--seed",    type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--verbose", action="store_true",  help="Print per-step detail")
    parser.add_argument("--output",  default=None,  help="Save JSON results to this path")
    args = parser.parse_args()

    # Print active config for reproducibility
    print(f"\nConfig:")
    print(f"  agent        = {args.agent}")
    print(f"  API_BASE_URL = {API_BASE_URL}")
    print(f"  MODEL_NAME   = {MODEL_NAME}")
    print(f"  HF_TOKEN set = {'yes' if API_KEY else 'NO (llm agent will fail)'}")
    print(f"  seed         = {args.seed}")
    sys.stdout.flush()

    if args.agent == "rule_based":
        agent = RuleBasedAgent()
    elif args.agent == "random":
        agent = RandomAgent(seed=args.seed)
    else:
        agent = LLMAgent()

    env = SQLBusinessEnv(seed=args.seed)

    if args.task:
        run_single_task(env, agent, args.task, verbose=True)
        return

    # Full benchmark
    print(f"\nRunning full benchmark...")
    print("─" * 64)
    sys.stdout.flush()

    all_tasks  = [t.task_id for t in env.tasks]
    results    = []
    total      = 0.0

    for task_id in all_tasks:
        score = run_single_task(env, agent, task_id, verbose=args.verbose)
        results.append({"task_id": task_id, "score": score})
        total += score

    normalized = total / len(all_tasks) if all_tasks else 0.0

    print(f"\n{'═'*64}")
    print(f"  BENCHMARK — {args.agent.upper()} (seed={args.seed})")
    print(f"{'═'*64}")

    easy_scores   = [r["score"] for r in results if "easy"   in r["task_id"]]
    medium_scores = [r["score"] for r in results if "medium" in r["task_id"]]
    hard_scores   = [r["score"] for r in results if "hard"   in r["task_id"]]

    easy_avg   = sum(easy_scores)   / len(easy_scores)   if easy_scores   else 0.0
    medium_avg = sum(medium_scores) / len(medium_scores) if medium_scores else 0.0
    hard_avg   = sum(hard_scores)   / len(hard_scores)   if hard_scores   else 0.0

    print(f"  Easy avg   : {easy_avg:.4f}")
    print(f"  Medium avg : {medium_avg:.4f}")
    print(f"  Hard avg   : {hard_avg:.4f}")
    print(f"  {'─'*36}")
    print(f"  Total      : {total:.4f} / {len(all_tasks):.1f}")
    print(f"  Normalized : {normalized:.4f}\n")

    for r in results:
        diff  = "easy" if "easy" in r["task_id"] else "medium" if "medium" in r["task_id"] else "hard"
        icon  = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(diff, "")
        bar   = "█" * int(r["score"] * 10) + "░" * (10 - int(r["score"] * 10))
        print(f"  {icon} {r['task_id']:12s} [{bar}] {r['score']:.4f}")

    if args.output:
        out = {
            "agent":      args.agent,
            "seed":       args.seed,
            "easy_avg":   easy_avg,
            "medium_avg": medium_avg,
            "hard_avg":   hard_avg,
            "normalized": normalized,
            "per_task":   results,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Results saved → {args.output}")

    sys.stdout.flush()


def _strip_fences(text: str) -> str:
    for fence in ("```sql", "```SQL", "```"):
        if text.startswith(fence):
            text = text[len(fence):].lstrip("\n")
    if text.endswith("```"):
        text = text[:-3].rstrip()
    return text.strip()


if __name__ == "__main__":
    main()