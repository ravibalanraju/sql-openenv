# 🗄️ SQL Business Intelligence — OpenEnv

> **Scaler OpenEnv Hackathon** — real-world agent environment

An AI agent receives a natural-language business question and must write a SQL query to answer it correctly from a retail database. Tasks range from a simple `COUNT(*)` (easy) to multi-table CTEs with discount-aware aggregations (hard).

**Why this domain?** Natural-language-to-SQL is one of the highest-value agent capabilities in industry. Every BI team, data analyst, and product manager would immediately benefit from a reliable SQL-writing agent. This environment gives the RL community a rigorous, reproducible benchmark for training and evaluating such agents.

---

## 🗂 Project structure

```
openenv-sql/
├── env/
│   ├── __init__.py        # Package exports
│   ├── models.py          # Typed Pydantic models (Action, Observation, Reward, State)
│   ├── database.py        # SQLite schema + ~60 seed rows + safe_execute()
│   ├── tasks.py           # 9 TaskDefinition objects (easy / medium / hard)
│   ├── graders.py         # Partial-credit scoring per answer type
│   └── environment.py     # SQLBusinessEnv: reset / step / state + benchmark runner
├── inference.py           # Baseline agents: rule-based, random, LLM (OpenAI)
├── app.py                 # Gradio UI → HuggingFace Spaces
├── openenv.yaml           # OpenEnv spec manifest
├── Dockerfile             # HF Spaces-compatible container
├── requirements.txt
└── README.md
```

---

## 📐 OpenEnv API

```python
from env import SQLBusinessEnv, SQLAction

env = SQLBusinessEnv(seed=42)

# 1. reset — start an episode
obs = env.reset("medium_01")
print(obs.question)      # "List the top 3 products by total gross revenue..."
print(obs.schema_info)   # Database DDL
print(obs.difficulty)    # "medium"

# 2. step — submit a SQL query
action = SQLAction(sql_query="""
    SELECT p.name
    FROM order_items oi JOIN products p ON oi.product_id = p.id
    GROUP BY p.id, p.name
    ORDER BY SUM(oi.quantity * oi.unit_price) DESC
    LIMIT 3
""")
result = env.step(action)
print(result.reward)              # 1.0 if correct on first try
print(result.reward_detail)       # breakdown: correctness, penalty, syntax_bonus
print(result.done)                # True when solved or max attempts reached
print(result.info["reason"])      # "Exact ordered match."

# 3. state — inspect full episode
state = env.state()
print(state.best_reward)          # best reward this episode
print(state.best_query)           # SQL that produced that reward
print(state.episode_history)      # full attempt log
```

---

## 📊 Observation space

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Unique task identifier |
| `difficulty` | `str` | `easy` \| `medium` \| `hard` |
| `question` | `str` | Natural-language BI question |
| `schema_info` | `str` | Full database schema DDL |
| `attempt` | `int` | 0-indexed attempt counter |
| `max_attempts` | `int` | Maximum attempts for this task |
| `previous_query` | `str?` | Last SQL submitted (None on first step) |
| `previous_result` | `str?` | Truncated result of the last query |
| `previous_score` | `float?` | Reward from the last step |
| `hint` | `str?` | Revealed after the first wrong attempt |

## 🎯 Action space

| Field | Type | Description |
|---|---|---|
| `sql_query` | `str` | A SQLite `SELECT` or `WITH…SELECT` statement |

Only read operations are permitted. DML/DDL is blocked.

---

## 🏆 Tasks

### Easy

| ID | Question |
|---|---|
| `easy_01` | How many customers are from New York state? |
| `easy_02` | What is the unit price of the Ergonomic Chair? |
| `easy_03` | How many orders have status = 'completed'? |

*Requires: single-table SELECT with WHERE / COUNT. No JOINs needed.*

### Medium

| ID | Question |
|---|---|
| `medium_01` | Top 3 products by total gross revenue (JOIN + GROUP BY + ORDER BY) |
| `medium_02` | Total net revenue from completed orders (discount-aware aggregation) |
| `medium_03` | Category with highest total gross revenue |

*Requires: 2-table JOIN, GROUP BY, aggregation, ORDER BY.*

### Hard

| ID | Question |
|---|---|
| `hard_01` | Customer(s) who placed the most orders (HAVING + subquery) |
| `hard_02` | Monthly revenue breakdown for 2024 (date bucketing, concatenated output) |
| `hard_03` | Customers whose net spend exceeds per-customer average (CTE + AVG subquery) |

*Requires: CTEs, HAVING clauses, subqueries, date functions, multi-join.*

---

## 🎁 Reward function

```
reward = clamp(correctness + syntax_bonus − attempt_penalty × attempt, 0.0, 1.0)
```

| Component | Value | Rationale |
|---|---|---|
| `correctness` | 0.0–1.0 | Grader score (type-specific, with partial credit) |
| `syntax_bonus` | +0.05 | SQL ran without error — valid signal even when answer is wrong |
| `attempt_penalty` | −0.10 × attempt | Encourages solving on the first try |

### Partial credit by answer type

| Type | 1.0 | 0.5 | < 0.5 | 0.0 |
|---|---|---|---|---|
| `scalar_int` | Exact | — | — | Wrong |
| `scalar_float` | Within ±0.02 | Within ±0.10 | — | Far off |
| `scalar_string` | Case-insensitive match | — | — | Wrong |
| `ordered_list` | Exact ordered match | Right items, wrong order | Proportional overlap | No match |

---

## 🗄️ Database schema

```
customers   (id, name, city, state, email, signup_date)
products    (id, name, category, unit_price, stock_qty)
orders      (id, customer_id, order_date, status, discount_pct)
order_items (id, order_id, product_id, quantity, unit_price)
returns     (id, order_id, product_id, return_date, reason)
```

SQLite 3, in-memory, seeded deterministically on startup (~60 rows total).

---

## ⚡ Setup and usage

### Install

```bash
pip install -r requirements.txt
```

### Run the rule-based baseline (no API key)

```bash
python inference.py --agent rule_based --verbose
```

Expected output:
```
  [easy_01     ] [██████████] 1.0000  (1 attempts)
  [easy_02     ] [██████████] 1.0000  (1 attempts)
  ...
  Normalized : 1.0000
```

### Run the LLM agent (OpenAI)

```bash
export OPENAI_API_KEY=sk-...
python inference.py --agent llm --model gpt-4o-mini --verbose
```

### Debug a single task

```bash
python inference.py --agent rule_based --task hard_03 --verbose
```

### Launch Gradio UI

```bash
python app.py
# Open http://localhost:7860
```

---

## 🐳 Docker

```bash
docker build -t sql-openenv .
docker run -p 7860:7860 sql-openenv
# With LLM agent: docker run -e OPENAI_API_KEY=sk-... -p 7860:7860 sql-openenv
```

---

## 🚀 Deploy to HuggingFace Spaces

1. Create a new Space at https://huggingface.co/spaces (SDK: **Gradio**)
2. Push this repo:

```bash
git init && git add . && git commit -m "initial"
git remote add hf https://huggingface.co/spaces/<YOUR_USERNAME>/sql-business-intelligence
git push hf main
```

The Space auto-builds from `Dockerfile`.

---

## 📈 Baseline scores

| Agent | Easy avg | Medium avg | Hard avg | Normalized |
|---|---|---|---|---|
| `rule_based` (deterministic) | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| `random` (seed=42) | ~0.056 | ~0.056 | ~0.056 | **~0.056** |
| `llm gpt-4o-mini` (temp=0) | ~0.95 | ~0.85 | ~0.65 | **~0.82** |

To reproduce rule-based baseline exactly:
```bash
python inference.py --agent rule_based --seed 42 --output baseline_results.json
```

---

## 🔑 Writing your own agent

```python
from env import SQLBusinessEnv, SQLAction, Observation

def my_agent(obs: Observation) -> SQLAction:
    # obs.question       — what to answer
    # obs.schema_info    — the database schema
    # obs.previous_result — feedback from last attempt
    # obs.hint           — hint revealed after first wrong attempt
    return SQLAction(sql_query="SELECT 42")   # replace with your logic

env = SQLBusinessEnv(seed=42)
bm  = env.run_benchmark(my_agent, verbose=True)
print(f"Score: {bm.normalized_score:.4f}")
```

---

## 📜 License

MIT
