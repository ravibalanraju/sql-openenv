"""
inference.py — Baseline agents and benchmark runner.
Mandatory env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN
"""
from __future__ import annotations
import argparse, json, os, sys
from typing import Optional

from env import SQLBusinessEnv, SQLAction, Observation

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
MAX_TOKENS   = 512
TEMPERATURE  = 0.0


class RuleBasedAgent:
    """Hard-coded optimal SQL. No API key needed. Scores 1.0 on all tasks."""
    _SOLUTIONS = {
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
            "  SELECT MAX(cnt) FROM ("
            "    SELECT COUNT(*) AS cnt FROM orders GROUP BY customer_id"
            "  )"
            ") ORDER BY c.name"
        ),
        "hard_02": (
            "SELECT strftime('%Y-%m', o.order_date) || ':' || "
            "       CAST(ROUND(SUM(oi.quantity * oi.unit_price), 2) AS TEXT) "
            "FROM order_items oi JOIN orders o ON oi.order_id = o.id "
            "WHERE o.status = 'completed' AND o.order_date LIKE '2024-%' "
            "GROUP BY strftime('%Y-%m', o.order_date) ORDER BY 1"
        ),
        "hard_03": (
            "WITH pc AS ("
            "  SELECT c.id, c.name, "
            "         SUM(oi.quantity * oi.unit_price * (1 - o.discount_pct)) AS ns "
            "  FROM customers c "
            "  JOIN orders o ON o.customer_id = c.id "
            "  JOIN order_items oi ON oi.order_id = o.id "
            "  WHERE o.status = 'completed' GROUP BY c.id, c.name"
            ") SELECT name FROM pc "
            "WHERE ns > (SELECT AVG(ns) FROM pc) ORDER BY name"
        ),
    }

    def __call__(self, obs: Observation) -> SQLAction:
        sql = self._SOLUTIONS.get(obs.task_id, "SELECT 'unknown task' AS error")
        return SQLAction(sql_query=sql)


class RandomAgent:
    """Random SQL — floor baseline."""
    _POOL = [
        "SELECT 42",
        "SELECT COUNT(*) FROM customers",
        "SELECT name FROM products LIMIT 1",
        "SELECT AVG(unit_price) FROM products",
    ]
    def __init__(self, seed=42):
        import random
        self._rng = random.Random(seed)
    def __call__(self, obs: Observation) -> SQLAction:
        return SQLAction(sql_query=self._rng.choice(self._POOL))


class LLMAgent:
    """Uses OpenAI-compatible API with API_BASE_URL + HF_TOKEN + MODEL_NAME."""

    SYSTEM = (
        "You are an expert SQLite developer. "
        "Output ONLY a raw SQL SELECT query — no markdown, no explanation."
    )

    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("Run: pip install openai")
        if not API_KEY:
            sys.exit("Set HF_TOKEN or API_KEY environment variable.")
        self._client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    def __call__(self, obs: Observation) -> SQLAction:
        parts = [
            f"SCHEMA\n{obs.schema_info}",
            f"\nQUESTION\n{obs.question}",
        ]
        if obs.previous_query:
            parts.append(
                f"\nPREVIOUS ATTEMPT\n"
                f"SQL: {obs.previous_query}\n"
                f"Result: {obs.previous_result}\n"
                f"Score: {obs.previous_score:.3f}\n"
                f"Your previous answer was wrong. Fix it."
            )
        if obs.hint:
            parts.append(f"\nHINT: {obs.hint}")

        try:
            resp = self._client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stream=False,
                messages=[
                    {"role": "system", "content": self.SYSTEM},
                    {"role": "user",   "content": "\n".join(parts)},
                ],
            )
            raw = resp.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [LLMAgent] API error: {exc}")
            raw = "SELECT 'api_error' AS error"

        # Strip markdown fences if model added them
        raw = raw.strip()
        for fence in ("```sql", "```SQL", "```"):
            if raw.startswith(fence):
                raw = raw[len(fence):].lstrip("\n")
        if raw.endswith("```"):
            raw = raw[:-3].rstrip()

        return SQLAction(sql_query=raw.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="SQL BI OpenEnv Benchmark")
    parser.add_argument(
        "--agent",
        choices=["rule_based", "random", "llm"],
        default="rule_based",
    )
    parser.add_argument("--task",    default=None)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output",  default=None)
    args = parser.parse_args()

    print(f"\nConfig:")
    print(f"  agent        = {args.agent}")
    print(f"  API_BASE_URL = {API_BASE_URL}")
    print(f"  MODEL_NAME   = {MODEL_NAME}")
    print(f"  HF_TOKEN set = {'yes' if API_KEY else 'NO'}")
    print(f"  seed         = {args.seed}")

    if args.agent == "rule_based":
        agent = RuleBasedAgent()
    elif args.agent == "random":
        agent = RandomAgent(seed=args.seed)
    else:
        agent = LLMAgent()

    env = SQLBusinessEnv(seed=args.seed)

    if args.task:
        obs  = env.reset(task_id=args.task)
        print(f"\nTask: {args.task} | {obs.question}")
        done = False
        while not done:
            action = agent(obs)
            result = env.step(action)
            if args.verbose:
                print(f"  Attempt {obs.attempt + 1}: {action.sql_query[:80]}")
                print(f"  Reward={result.reward:.4f}  {result.info['reason']}")
            obs  = result.observation
            done = result.done
        print(f"Best reward: {env.state().best_reward:.4f}")
        return

    # Full benchmark
    print(f"\nRunning full benchmark...\n{'─'*60}")
    bm = env.run_benchmark(agent, verbose=args.verbose)
    bm.agent = args.agent

    print(f"\n{'═'*60}")
    print(f"  RESULTS — {args.agent.upper()}  (seed={args.seed})")
    print(f"{'═'*60}")
    print(f"  Easy   avg : {bm.easy_avg:.4f}")
    print(f"  Medium avg : {bm.medium_avg:.4f}")
    print(f"  Hard   avg : {bm.hard_avg:.4f}")
    print(f"  Normalized : {bm.normalized_score:.4f}  "
          f"({bm.total_score:.2f}/{bm.max_score:.1f})\n")

    for r in bm.per_task:
        icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(r.difficulty, "")
        bar  = "█" * int(r.score * 10) + "░" * (10 - int(r.score * 10))
        print(f"  {icon} {r.task_id:12s} [{bar}] {r.score:.4f}  ({r.attempts_used} att)")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(bm.model_dump(), f, indent=2)
        print(f"\n  Saved → {args.output}")


if __name__ == "__main__":
    main()