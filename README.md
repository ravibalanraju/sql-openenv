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

## Environment Description
The agent is given a business question and must write a SQL query to answer it correctly against a seeded SQLite retail database.

## Action Space
- `sql_query`: A valid SQL SELECT statement

## Observation Space
- `task_id`, `difficulty`, `question`, `schema_info`, `attempt`, `hint`

## Tasks
- 3 Easy tasks (single table)
- 3 Medium tasks (joins + aggregation)  
- 3 Hard tasks (CTEs + subqueries)

## Reward
- Correctness: 0.0 – 1.0
- Syntax bonus: +0.05
- Attempt penalty: -0.10 per retry

## Setup
```bash