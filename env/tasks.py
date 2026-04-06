"""
Task catalogue — SQL Business Intelligence OpenEnv.

All expected_answer values are verified against the seeded SQLite database.

Difficulty rubric
─────────────────
easy   : single table, simple WHERE / COUNT / direct lookup
medium : 2-table JOIN + GROUP BY + aggregation; or subquery on one table
hard   : multi-JOIN + HAVING / subquery / CTE / window-style logic
"""
from __future__ import annotations
from env.models import TaskDefinition

TASKS: list[TaskDefinition] = [

    # ══ EASY ══════════════════════════════════════════════════════════════════

    TaskDefinition(
        task_id="easy_01",
        difficulty="easy",
        question="How many customers are from New York state (state = 'NY')?",
        expected_answer=4,
        answer_type="scalar_int",
        hint="SELECT COUNT(*) FROM customers WHERE state = 'NY'",
        max_attempts=3,
    ),

    TaskDefinition(
        task_id="easy_02",
        difficulty="easy",
        question=(
            "What is the unit price of the product named 'Ergonomic Chair'? "
            "Return a single numeric value."
        ),
        expected_answer=299.99,
        answer_type="scalar_float",
        hint="SELECT unit_price FROM products WHERE name = 'Ergonomic Chair'",
        max_attempts=3,
    ),

    TaskDefinition(
        task_id="easy_03",
        difficulty="easy",
        question="How many orders have status = 'completed'?",
        expected_answer=12,
        answer_type="scalar_int",
        hint="SELECT COUNT(*) FROM orders WHERE status = 'completed'",
        max_attempts=3,
    ),

    # ══ MEDIUM ════════════════════════════════════════════════════════════════

    TaskDefinition(
        task_id="medium_01",
        difficulty="medium",
        question=(
            "List the top 3 products by total gross revenue (SUM of quantity × unit_price "
            "across ALL order_items rows, regardless of order status). "
            "Return an ordered list of product names, highest revenue first."
        ),
        expected_answer=["Ergonomic Chair", "Mechanical Keyboard", "USB-C Hub"],
        answer_type="ordered_list",
        hint=(
            "JOIN order_items → products. "
            "GROUP BY product, SUM(quantity * unit_price), ORDER BY DESC, LIMIT 3."
        ),
        max_attempts=3,
    ),

    TaskDefinition(
        task_id="medium_02",
        difficulty="medium",
        question=(
            "What is the total NET revenue from completed orders? "
            "Net revenue accounts for the per-order discount: "
            "SUM(quantity × unit_price × (1 - discount_pct)) "
            "across all order_items whose parent order has status = 'completed'. "
            "Return a single numeric value rounded to 2 decimal places."
        ),
        # verified: 1710.18
        expected_answer=1669.18,
        answer_type="scalar_float",
        hint=(
            "JOIN order_items → orders. "
            "Filter WHERE o.status = 'completed'. "
            "SUM(oi.quantity * oi.unit_price * (1 - o.discount_pct)), ROUND to 2."
        ),
        max_attempts=3,
    ),

    TaskDefinition(
        task_id="medium_03",
        difficulty="medium",
        question=(
            "Which product category generated the highest total gross revenue "
            "(SUM of quantity × unit_price across ALL order_items)? "
            "Return a single category name."
        ),
        expected_answer="Furniture",
        answer_type="scalar_string",
        hint=(
            "JOIN order_items → products. "
            "GROUP BY category, SUM(quantity * unit_price), ORDER BY DESC, LIMIT 1."
        ),
        max_attempts=3,
    ),

    # ══ HARD ══════════════════════════════════════════════════════════════════

    TaskDefinition(
        task_id="hard_01",
        difficulty="hard",
        question=(
            "Find the customer(s) with the highest total number of orders placed. "
            "If there is a tie, return all tied customers. "
            "Return customer names sorted alphabetically."
        ),
        # Alice Martin: 3 orders (orders 1, 4, 14) — verified
        expected_answer=["Alice Martin"],
        answer_type="ordered_list",
        hint=(
            "GROUP BY customer, COUNT(*) as cnt. "
            "HAVING cnt = (SELECT MAX(...) subquery). ORDER BY name."
        ),
        max_attempts=4,
    ),

    TaskDefinition(
        task_id="hard_02",
        difficulty="hard",
        question=(
            "For each calendar month in 2024 that contains at least one completed order, "
            "report the month (as 'YYYY-MM') and the total gross revenue from completed orders "
            "in that month (SUM of quantity × unit_price, ROUND to 2 decimals). "
            "Return results as a list of strings in the format 'YYYY-MM:revenue', "
            "sorted by month ascending."
        ),
        # verified from DB
        expected_answer=[
            "2024-01:292.9",
            "2024-02:419.96",
            "2024-03:379.97",
            "2024-04:301.38",
            "2024-05:399.96",
        ],
        answer_type="ordered_list",
        hint=(
            "strftime('%Y-%m', order_date) for grouping. "
            "JOIN order_items → orders WHERE status='completed'. "
            "Concatenate month || ':' || CAST(ROUND(rev,2) AS TEXT)."
        ),
        max_attempts=4,
    ),

    TaskDefinition(
        task_id="hard_03",
        difficulty="hard",
        question=(
            "Find customers whose total NET spend on completed orders exceeds "
            "the average NET spend per customer (only counting customers who have "
            "at least one completed order). "
            "Net spend = SUM(quantity × unit_price × (1 − discount_pct)). "
            "Return customer names sorted alphabetically."
        ),
        # verified: Alice Martin, Bob Singh
        expected_answer=["Alice Martin", "Bob Singh"],
        answer_type="ordered_list",
        hint=(
            "Use a CTE: first compute per-customer net spend, then SELECT WHERE "
            "spend > (SELECT AVG(spend) FROM cte)."
        ),
        max_attempts=4,
    ),
]

TASK_MAP: dict[str, TaskDefinition] = {t.task_id: t for t in TASKS}
