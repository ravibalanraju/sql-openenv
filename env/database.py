"""
Retail SQLite database used by the SQL BI OpenEnv environment.

Schema  (all tables created fresh on connect)
──────
customers     – id, name, city, state, email, signup_date
products      – id, name, category, unit_price, stock_qty
orders        – id, customer_id, order_date, status, discount_pct
order_items   – id, order_id, product_id, quantity, unit_price
returns       – id, order_id, product_id, return_date, reason
"""
from __future__ import annotations

import sqlite3
from typing import Optional

# ── Schema ─────────────────────────────────────────────────────────────────────

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS customers (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    city        TEXT    NOT NULL,
    state       TEXT    NOT NULL,
    email       TEXT    UNIQUE NOT NULL,
    signup_date TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS products (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    category    TEXT    NOT NULL,
    unit_price  REAL    NOT NULL,
    stock_qty   INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date  TEXT    NOT NULL,
    status      TEXT    NOT NULL,
    discount_pct REAL   NOT NULL DEFAULT 0.0
);
CREATE TABLE IF NOT EXISTS order_items (
    id          INTEGER PRIMARY KEY,
    order_id    INTEGER NOT NULL REFERENCES orders(id),
    product_id  INTEGER NOT NULL REFERENCES products(id),
    quantity    INTEGER NOT NULL,
    unit_price  REAL    NOT NULL
);
CREATE TABLE IF NOT EXISTS returns (
    id          INTEGER PRIMARY KEY,
    order_id    INTEGER NOT NULL REFERENCES orders(id),
    product_id  INTEGER NOT NULL REFERENCES products(id),
    return_date TEXT    NOT NULL,
    reason      TEXT    NOT NULL
);
"""

SCHEMA_INFO = """
Database: retail (SQLite 3)

customers   (id INT PK, name TEXT, city TEXT, state TEXT, email TEXT, signup_date TEXT 'YYYY-MM-DD')
products    (id INT PK, name TEXT, category TEXT, unit_price REAL, stock_qty INT)
orders      (id INT PK, customer_id INT FK→customers, order_date TEXT 'YYYY-MM-DD',
             status TEXT ['completed','pending','cancelled'], discount_pct REAL 0.0–1.0)
order_items (id INT PK, order_id INT FK→orders, product_id INT FK→products,
             quantity INT, unit_price REAL)
returns     (id INT PK, order_id INT FK→orders, product_id INT FK→products,
             return_date TEXT, reason TEXT)

Useful expressions
  Line revenue  : quantity * unit_price
  After discount: quantity * unit_price * (1 - o.discount_pct)
  Month bucket  : strftime('%Y-%m', order_date)
  Year bucket   : strftime('%Y', order_date)
""".strip()

# ── Seed data ──────────────────────────────────────────────────────────────────

_CUSTOMERS = [
    (1,  "Alice Martin",   "New York",    "NY", "alice@example.com",   "2022-03-15"),
    (2,  "Bob Singh",      "Los Angeles", "CA", "bob@example.com",     "2022-07-22"),
    (3,  "Carol White",    "New York",    "NY", "carol@example.com",   "2023-01-10"),
    (4,  "David Kim",      "Chicago",     "IL", "david@example.com",   "2023-04-05"),
    (5,  "Eva Zhao",       "Houston",     "TX", "eva@example.com",     "2023-06-18"),
    (6,  "Frank Lopez",    "Phoenix",     "AZ", "frank@example.com",   "2023-08-30"),
    (7,  "Grace Patel",    "New York",    "NY", "grace@example.com",   "2023-11-12"),
    (8,  "Henry Brown",    "San Diego",   "CA", "henry@example.com",   "2024-01-03"),
    (9,  "Iris Nguyen",    "Dallas",      "TX", "iris@example.com",    "2024-02-14"),
    (10, "Jack Robinson",  "New York",    "NY", "jack@example.com",    "2024-03-20"),
]

_PRODUCTS = [
    (1,  "Wireless Mouse",      "Electronics",  29.99, 120),
    (2,  "Mechanical Keyboard", "Electronics",  89.99,  45),
    (3,  "USB-C Hub",           "Electronics",  49.99,  80),
    (4,  "Notebook A5",         "Stationery",    8.99, 500),
    (5,  "Ballpoint Pens 10pk", "Stationery",    5.49, 800),
    (6,  "Desk Lamp",           "Furniture",    34.99,  60),
    (7,  "Ergonomic Chair",     "Furniture",   299.99,  15),
    (8,  "Monitor Stand",       "Furniture",    59.99,  40),
    (9,  "Webcam HD",           "Electronics",  79.99,  55),
    (10, "Mouse Pad XL",        "Accessories",  19.99, 200),
]

# status, discount_pct added
_ORDERS = [
    (1,  1, "2024-01-05", "completed", 0.00),
    (2,  2, "2024-01-12", "completed", 0.10),
    (3,  3, "2024-01-20", "completed", 0.00),
    (4,  1, "2024-02-03", "completed", 0.00),
    (5,  4, "2024-02-15", "pending",   0.05),
    (6,  5, "2024-02-28", "completed", 0.00),
    (7,  6, "2024-03-10", "cancelled", 0.00),
    (8,  7, "2024-03-18", "completed", 0.15),
    (9,  2, "2024-03-25", "completed", 0.10),
    (10, 8, "2024-04-02", "completed", 0.00),
    (11, 9, "2024-04-10", "completed", 0.00),
    (12, 3, "2024-04-22", "completed", 0.00),
    (13,10, "2024-05-01", "completed", 0.00),
    (14, 1, "2024-05-14", "completed", 0.20),
    (15, 5, "2024-05-30", "pending",   0.00),
]

_ORDER_ITEMS = [
    (1,  1,  1, 1,  29.99),
    (2,  1,  4, 3,   8.99),
    (3,  2,  2, 1,  89.99),
    (4,  2,  3, 2,  49.99),
    (5,  3,  6, 1,  34.99),
    (6,  3,  5, 2,   5.49),
    (7,  4,  7, 1, 299.99),
    (8,  5,  9, 1,  79.99),
    (9,  5, 10, 2,  19.99),
    (10, 6,  1, 2,  29.99),
    (11, 6,  8, 1,  59.99),
    (12, 7,  2, 1,  89.99),
    (13, 8,  3, 1,  49.99),
    (14, 8,  1, 1,  29.99),
    (15, 9,  7, 1, 299.99),
    (16,10,  4, 5,   8.99),
    (17,10,  5, 3,   5.49),
    (18,11,  9, 1,  79.99),
    (19,11,  2, 1,  89.99),
    (20,12,  6, 2,  34.99),
    (21,13,  1, 1,  29.99),
    (22,13,  3, 1,  49.99),
    (23,14,  7, 1, 299.99),
    (24,14, 10, 1,  19.99),
    (25,15,  8, 1,  59.99),
]

_RETURNS = [
    (1,  2,  2, "2024-01-20", "defective"),
    (2,  8,  3, "2024-03-28", "wrong item"),
    (3,  9,  7, "2024-04-05", "changed mind"),
]

# ── Connection builder ─────────────────────────────────────────────────────────

def build_connection(db_path: str = ":memory:") -> sqlite3.Connection:
    """Create, migrate, and seed a SQLite connection."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(SCHEMA_DDL)
    _seed(conn)
    conn.commit()
    return conn


def _seed(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executemany("INSERT OR IGNORE INTO customers VALUES (?,?,?,?,?,?)",  _CUSTOMERS)
    cur.executemany("INSERT OR IGNORE INTO products  VALUES (?,?,?,?,?)",    _PRODUCTS)
    cur.executemany("INSERT OR IGNORE INTO orders    VALUES (?,?,?,?,?)",    _ORDERS)
    cur.executemany("INSERT OR IGNORE INTO order_items VALUES (?,?,?,?,?)",  _ORDER_ITEMS)
    cur.executemany("INSERT OR IGNORE INTO returns   VALUES (?,?,?,?,?)",    _RETURNS)


# ── Safe query execution ───────────────────────────────────────────────────────

_FORBIDDEN_KEYWORDS = ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
                        "CREATE", "TRUNCATE", "REPLACE", "ATTACH")


def safe_execute(conn: sqlite3.Connection, query: str, limit: int = 100):
    """
    Execute a read-only query.

    Returns (rows: list[dict] | None, columns: list[str] | None, error: str | None)

    Blocks any non-SELECT / non-WITH statement.
    """
    stripped = query.strip()
    upper = stripped.upper()

    # Must start with SELECT or WITH (CTE)
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return None, None, (
            "Only SELECT or WITH…SELECT statements are allowed. "
            f"Your query starts with: '{stripped[:30]}'"
        )

    # Block forbidden keywords anywhere in the query (simple heuristic)
    for kw in _FORBIDDEN_KEYWORDS:
        if f" {kw} " in f" {upper} ":
            return None, None, f"Forbidden keyword detected: {kw}"

    try:
        cur = conn.execute(stripped)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = [dict(r) for r in cur.fetchmany(limit)]
        return rows, cols, None
    except sqlite3.Error as exc:
        return None, None, str(exc)
