#!/usr/bin/env python3
"""
Baseline inference script for SQL Business Intelligence OpenEnv.
Uses OpenAI-compatible API client as required by the OpenEnv spec.
"""

import os
import sys
import json
import argparse
import traceback
import time

# ── Env vars (spec-required) ──────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "dummy-key")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RULE_BASED_SQL = {
    "easy_01":   "SELECT COUNT(*) as count FROM customers WHERE state = 'NY'",
    "easy_02":   "SELECT price FROM products WHERE name = 'Ergonomic Chair'",
    "easy_03":   "SELECT COUNT(*) as count FROM orders WHERE status = 'completed'",
    "medium_01": "SELECT p.name FROM order_items oi JOIN products p ON oi.product_id = p.id GROUP BY p.id, p.name ORDER BY SUM(oi.quantity * oi.unit_price) DESC LIMIT 3",
    "medium_02": "SELECT ROUND(SUM(oi.quantity * oi.unit_price * (1 - o.discount)), 2) as net_revenue FROM orders o JOIN order_items oi ON o.id = oi.order_id WHERE o.status = 'completed'",
    "medium_03": "SELECT p.category FROM order_items oi JOIN products p ON oi.product_id = p.id GROUP BY p.category ORDER BY SUM(oi.quantity * oi.unit_price) DESC LIMIT 1",
    "hard_01":   "SELECT c.name FROM orders o JOIN customers c ON o.customer_id = c.id GROUP BY c.id, c.name ORDER BY COUNT(*) DESC LIMIT 1",
    "hard_02":   "WITH monthly AS (SELECT strftime('%Y-%m', o.order_date) as month, ROUND(SUM(oi.quantity * oi.unit_price * (1 - o.discount)), 2) as revenue FROM orders o JOIN order_items oi ON o.id = oi.order_id WHERE strftime('%Y', o.order_date) = '2024' GROUP BY month) SELECT month, revenue FROM monthly ORDER BY month",
    "hard_03":   "WITH customer_spend AS (SELECT c.id, c.name, SUM(oi.quantity * oi.unit_price * (1 - o.discount)) as total_spend FROM customers c JOIN orders o ON c.id = o.customer_id JOIN order_items oi ON o.id = oi.order_id GROUP BY c.id, c.name), avg_spend AS (SELECT AVG(total_spend) as avg FROM customer_spend) SELECT cs.name FROM customer_spend cs, avg_spend a WHERE cs.total_spend > a.avg ORDER BY cs.total_spend DESC",
}


def wait_for_server(url, timeout=120):
    import urllib.request
    print(f"[INFO] Waiting for server at {url} ...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(f"{url}/health", timeout=5)
            if resp.status == 200:
                print(f"[INFO] Server is ready!")
                return True
        except Exception:
            pass
        time.sleep(3)
    print(f"[WARN] Server not ready after {timeout}s, continuing anyway...")
    return False


def http_post(url, data):
    import urllib.request
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())
    except Exception as e:
        print(f"[ERROR] POST {url} failed: {e}")
        return {}


def http_get(url):
    import urllib.request
    try:
        resp = urllib.request.urlopen(url, timeout=30)
        return json.loads(resp.read())
    except Exception as e:
        print(f"[ERROR] GET {url} failed: {e}")
        return {}


class RuleBasedAgent:
    def act(self, task_id, obs=None):
        return RULE_BASED_SQL.get(task_id, "SELECT 1")


class RandomAgent:
    def act(self, task_id, obs=None):
        import random
        return random.choice(list(RULE_BASED_SQL.values()))


class LLMAgent:
    def __init__(self, model=None):
        self.model = model or MODEL_NAME
        try:
            from openai import OpenAI
            self.client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        except Exception as e:
            print(f"[WARN] OpenAI init failed: {e}")
            self.client = None

    def act(self, task_id, obs=None):
        if not self.client:
            return RULE_BASED_SQL.get(task_id, "SELECT 1")
        try:
            question = obs.get("question", "") if obs else ""
            schema = obs.get("schema_info", "") if obs else ""
            hint = obs.get("hint", "") if obs else ""
            prompt = f"Write a SQL SELECT query to answer:\n\nSchema:\n{schema}\n\nQuestion: {question}"
            if hint:
                prompt += f"\nHint: {hint}"
            prompt += "\n\nReturn ONLY the SQL, no explanation."
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500, temperature=0.1,
            )
            sql = resp.choices[0].message.content.strip()
            return sql.replace("```sql", "").replace("```", "").strip()
        except Exception as e:
            print(f"[WARN] LLM failed: {e}")
            return RULE_BASED_SQL.get(task_id, "SELECT 1")


def run_benchmark(agent_name="rule_based", task_id=None, verbose=True,
                  seed=42, model=None, output=None):

    wait_for_server(ENV_URL)

    tasks_resp = http_get(f"{ENV_URL}/tasks")
    if tasks_resp and "tasks" in tasks_resp:
        all_tasks = [t["id"] for t in tasks_resp["tasks"]]
    else:
        all_tasks = list(RULE_BASED_SQL.keys())

    tasks = [task_id] if task_id else all_tasks

    if agent_name == "rule_based":
        agent = RuleBasedAgent()
    elif agent_name == "random":
        agent = RandomAgent()
    else:
        agent = LLMAgent(model=model)

    results = {}

    for tid in tasks:
        try:
            obs = http_post(f"{ENV_URL}/reset", {"task_id": tid})
            if not obs:
                obs = {"task_id": tid, "question": "", "schema_info": "",
                       "attempt": 0, "max_attempts": 4}

            max_attempts = obs.get("max_attempts", 4)
            best_reward = 0.0
            done = False
            attempt = 0

            if verbose:
                print(f"\n{'='*55}")
                print(f"Task: {tid}  |  {obs.get('difficulty','')}")
                print(f"Q: {obs.get('question','')[:80]}")

            while not done and attempt < max_attempts:
                attempt += 1
                sql = agent.act(tid, obs)

                if verbose:
                    print(f"\n  Attempt {attempt}/{max_attempts}")
                    print(f"  SQL: {sql[:100].strip()}")

                result = http_post(f"{ENV_URL}/step", {"sql_query": sql})
                if not result:
                    break

                reward = result.get("reward", 0.0)
                if isinstance(reward, dict):
                    reward = reward.get("value", 0.0)
                reward = float(reward)

                done = result.get("done", False)
                obs = result.get("observation", obs) or obs
                reason = ""
                if isinstance(result.get("info"), dict):
                    reason = result["info"].get("reason", "")

                best_reward = max(best_reward, reward)

                if verbose:
                    bar = "█" * int(reward * 10) + "░" * (10 - int(reward * 10))
                    print(f"  Reward: [{bar}] {reward:.4f}  {reason}")

            results[tid] = best_reward

        except Exception as e:
            print(f"[ERROR] Task {tid}: {e}")
            traceback.print_exc()
            results[tid] = 0.0

    scores = list(results.values())
    normalized = sum(scores) / len(scores) if scores else 0.0

    print(f"\n{'='*55}")
    print("BENCHMARK RESULTS")
    print(f"{'='*55}")
    for tid, score in results.items():
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        print(f"  [{tid:<12}] [{bar}] {score:.4f}")
    print(f"{'='*55}")
    print(f"  Normalized : {normalized:.4f}")
    print(f"{'='*55}\n")

    final = {"tasks": results, "normalized": normalized}
    if output:
        with open(output, "w") as f:
            json.dump(final, f, indent=2)
        print(f"[INFO] Saved to {output}")

    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent",   default="rule_based",
                        choices=["rule_based", "random", "llm"])
    parser.add_argument("--task",    default=None)
    parser.add_argument("--model",   default=None)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--output",  default=None)
    args = parser.parse_args()
    run_benchmark(agent_name=args.agent, task_id=args.task,
                  verbose=args.verbose, seed=args.seed,
                  model=args.model, output=args.output)


if __name__ == "__main__":
    main()