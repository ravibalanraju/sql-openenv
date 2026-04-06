---
title: SQL Business Intelligence OpenEnv
emoji: 🗄️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🗄️ SQL Business Intelligence — OpenEnv

An AI agent training environment where agents learn to answer real business questions by writing SQL queries against a retail database.

## 🎯 Environment Description

The agent is given a business question (e.g. *"Which are the top 3 products by revenue?"*) and must write a SQL query to answer it correctly. The environment scores the query based on correctness, syntax validity, and number of attempts.

**Why this task is real-world:**
- Every company has databases with business data
- SQL is a top-10 skill used by analysts daily
- Answering business questions from data is a core enterprise workflow

---

## 🗃️ Database Schema

A seeded SQLite retail database with 5 tables:

| Table | Rows | Description |
|-------|------|-------------|
| customers | 10 | name, city, state, email |
| products | 10 | name, category, price |
| orders | 15 | customer, date, status, discount |
| order_items | 25 | product, quantity, unit_price |
| returns | 3 | order, reason, date |

---

## 📋 Action Space

```python
class SQLAction:
    sql_query: str   # A valid SQL SELECT statement
```

The agent submits a SQL query string. Only `SELECT` and `WITH...SELECT` (CTEs) are allowed. Destructive statements (`DROP`, `INSERT`, `UPDATE`, etc.) are blocked.

---

## 👁️ Observation Space

```python
class Observation:
    task_id: str          # e.g. "medium_01"
    difficulty: str       # "easy" | "medium" | "hard"
    question: str         # The business question to answer
    schema_info: str      # Full database schema description
    attempt: int          # Current attempt number (max 4)
    max_attempts: int     # Always 4
    previous_query: str   # Last submitted SQL (or None)
    previous_result: str  # Last query result (or None)
    previous_score: float # Last reward score (or None)
    hint: str             # Revealed after first failed attempt
```

---

## 🎁 Reward Function

```
reward = clamp(correctness + syntax_bonus - attempt_penalty × attempt, 0.0, 1.0)
```

| Signal | Value | Description |
|--------|-------|-------------|
| correctness | 0.0 – 1.0 | How correct the answer is |
| syntax_bonus | +0.05 | SQL ran without error (valid syntax) |
| attempt_penalty | −0.10 per retry | Encourages getting it right first try |

**Partial credit is supported** — agents get signal on every attempt, not just at episode end.

---

## 📝 Tasks

### 🟢 Easy (single table, basic SQL)
| ID | Question |
|----|----------|
| easy_01 | How many customers are from New York state? |
| easy_02 | What is the price of the Ergonomic Chair? |
| easy_03 | How many orders have status "completed"? |

### 🟡 Medium (joins + aggregation)
| ID | Question |
|----|----------|
| medium_01 | List the top 3 products by total revenue |
| medium_02 | What is the total net revenue from completed orders after discounts? |
| medium_03 | Which product category has earned the most revenue? |

### 🔴 Hard (subqueries, CTEs, complex logic)
| ID | Question |
|----|----------|
| hard_01 | Which customer has placed the most orders? |
| hard_02 | What is the total revenue per month for 2024? |
| hard_03 | List customers whose total net spend is above the average |

---

## 🚀 Setup & Usage

### Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run rule-based baseline (no API key needed)
python inference.py --agent rule_based --verbose

# Run web UI
python app.py
# Open http://localhost:7860
```

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode |
| `/step` | POST | Submit a SQL action |
| `/state` | GET | Get current environment state |
| `/tasks` | GET | List all available tasks |
| `/health` | GET | Health check |

### Example API Usage

```bash
# Reset environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "medium_01"}'

# Submit a SQL query
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"sql_query": "SELECT p.name FROM products p LIMIT 3"}'

# Check state
curl http://localhost:7860/state
```

### Python Usage

```python
from env import SQLBusinessEnv, SQLAction

env = SQLBusinessEnv(seed=42)
obs = env.reset("medium_01")
print(obs.question)

action = SQLAction(sql_query="""
    SELECT p.name
    FROM order_items oi
    JOIN products p ON oi.product_id = p.id
    GROUP BY p.id, p.name
    ORDER BY SUM(oi.quantity * oi.unit_price) DESC
    LIMIT 3
""")

result = env.step(action)
print(result.reward)         # 1.0
print(result.info["reason"]) # "Exact ordered match."
```

---

## 📊 Baseline Scores

| Agent | Score | Description |
|-------|-------|-------------|
| RuleBasedAgent | 1.0000 | Hard-coded correct SQL for all tasks |
| RandomAgent | ~0.0000 | Random SQL — almost always fails |
| LLMAgent (GPT-4o-mini) | ~0.75 | Uses OpenAI API to generate SQL |

### Run Baselines

```bash
# Rule-based (no API key)
python inference.py --agent rule_based --verbose

# Random agent
python inference.py --agent random --verbose

# LLM agent (requires API key)
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_token_here
python inference.py --agent llm --verbose
```

---

## 🐳 Docker

```bash
docker build -t sql-openenv .
docker run -p 7860:7860 sql-openenv
```

---

## 📁 Project Structure

```
├── env/
│   ├── __init__.py        # Package exports
│   ├── models.py          # Typed models: SQLAction, Observation, Reward, EnvState
│   ├── database.py        # SQLite schema, seed data, safe_execute()
│   ├── tasks.py           # 9 TaskDefinition objects (easy/medium/hard)
│   ├── graders.py         # Scoring with partial credit
│   └── environment.py     # SQLBusinessEnv: reset/step/state/run_benchmark
├── app.py                 # FastAPI + Gradio UI (HuggingFace Spaces)
├── inference.py           # Baseline agents: rule_based, random, llm
├── openenv.yaml           # OpenEnv spec manifest
├── Dockerfile             # HF Spaces container
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## 📜 License

MIT
