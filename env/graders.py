"""
Graders for the SQL BI OpenEnv.

Each grader returns (correctness: float, reason: str) where correctness ∈ [0.0, 1.0].

Partial credit is awarded wherever it is semantically meaningful:
  - scalar_float : 1.0 exact | 0.5 close | 0.0 far
  - ordered_list : 1.0 exact | 0.5 right-items wrong-order | proportional partial
  - scalar_int   : 1.0 exact | 0.0 otherwise (count is binary)
  - scalar_string: 1.0 case-insensitive match | 0.0 otherwise

The Reward model also tracks:
  - syntax_bonus  : +0.1 if the query ran without a SQL error (no correctness credit, but useful signal)
  - attempt_penalty: −0.1 per attempt after the first (capped so reward >= 0)
"""
from __future__ import annotations
from typing import Any, List, Optional, Tuple

from env.models import Reward


# ── Helpers ────────────────────────────────────────────────────────────────────

def _first_value(rows: list[dict]) -> Optional[Any]:
    if not rows:
        return None
    vals = list(rows[0].values())
    return vals[0] if vals else None


def _col0_list(rows: list[dict]) -> list:
    if not rows:
        return []
    key = list(rows[0].keys())[0]
    return [r[key] for r in rows]


# ── Type graders ───────────────────────────────────────────────────────────────

def _grade_scalar_int(rows, expected: int) -> Tuple[float, str]:
    val = _first_value(rows)
    if val is None:
        return 0.0, "Query returned no rows."
    try:
        got = int(val)
    except (TypeError, ValueError):
        return 0.0, f"Expected integer, got: {val!r}"
    if got == expected:
        return 1.0, f"Correct! Got {got}."
    return 0.0, f"Expected {expected}, got {got}."


def _grade_scalar_float(rows, expected: float, tol: float = 0.02) -> Tuple[float, str]:
    val = _first_value(rows)
    if val is None:
        return 0.0, "Query returned no rows."
    try:
        got = float(val)
    except (TypeError, ValueError):
        return 0.0, f"Expected float, got: {val!r}"
    diff = abs(got - expected)
    if diff <= tol:
        return 1.0, f"Correct! Got {got:.4f}."
    if diff <= tol * 5:
        return 0.5, f"Close — expected {expected}, got {got:.4f} (Δ={diff:.4f})."
    return 0.0, f"Expected {expected}, got {got:.4f} (Δ={diff:.4f})."


def _grade_scalar_string(rows, expected: str) -> Tuple[float, str]:
    val = _first_value(rows)
    if val is None:
        return 0.0, "Query returned no rows."
    got = str(val).strip()
    if got.lower() == expected.lower():
        return 1.0, f"Correct! Got '{got}'."
    return 0.0, f"Expected '{expected}', got '{got}'."


def _grade_ordered_list(rows, expected: list) -> Tuple[float, str]:
    got = [str(v).strip() for v in _col0_list(rows)]
    exp = [str(v).strip() for v in expected]

    if got == exp:
        return 1.0, "Exact ordered match."

    got_set, exp_set = set(got), set(exp)
    if got_set == exp_set:
        return 0.5, f"Right items, wrong order. Got: {got}"

    # Partial: positional hits + set overlap
    positional = sum(1 for a, b in zip(got, exp) if a == b)
    overlap    = len(got_set & exp_set)
    score = round(max(positional, overlap) / max(len(exp), 1), 2)
    score = min(score, 0.9)  # never award full credit for partial

    if score > 0:
        return score, f"Partial ({overlap}/{len(exp)} items match). Expected: {exp}. Got: {got}"
    return 0.0, f"No matching items. Expected: {exp}. Got: {got}"


# ── Main dispatcher ────────────────────────────────────────────────────────────

def compute_reward(
    rows: Optional[list],
    error: Optional[str],
    expected: Any,
    answer_type: str,
    attempt: int,          # 0-indexed
    max_attempts: int,
    attempt_penalty: float = 0.10,
) -> Reward:
    """
    Compute a structured Reward for one step.

    Signals provided over the trajectory
    ─────────────────────────────────────
    1. syntax_bonus   : +0.1 if the SQL ran without error (even if answer wrong)
                        Teaches the agent to write syntactically valid SQL first.
    2. correctness    : grader score 0.0–1.0
    3. attempt_penalty: -0.10 per attempt after the first
                        Encourages getting the right answer early.

    final value = min(1.0, max(0.0, correctness + syntax_bonus - attempt_penalty))
    but syntax_bonus only contributes if correctness < 1.0 (no double-dipping).
    """
    # Step 1: syntax signal
    syntax_bonus = 0.0
    if error is None and rows is not None:
        syntax_bonus = 0.05  # ran clean

    # Step 2: correctness
    if error:
        correctness, reason = 0.0, f"SQL error: {error}"
    elif rows is None:
        correctness, reason = 0.0, "No result returned."
    else:
        if answer_type == "scalar_int":
            correctness, reason = _grade_scalar_int(rows, int(expected))
        elif answer_type == "scalar_float":
            correctness, reason = _grade_scalar_float(rows, float(expected))
        elif answer_type == "scalar_string":
            correctness, reason = _grade_scalar_string(rows, str(expected))
        elif answer_type == "ordered_list":
            correctness, reason = _grade_ordered_list(rows, list(expected))
        else:
            correctness, reason = 0.0, f"Unknown answer_type: {answer_type}"

    # No syntax bonus if already fully correct
    if correctness >= 1.0:
        syntax_bonus = 0.0

    # Step 3: attempt penalty
    penalty = round(attempt_penalty * attempt, 4)

    # Step 4: combine
    raw = correctness + syntax_bonus - penalty
    value = round(max(0.0, min(1.0, raw)), 4)

    return Reward(
        value=value,
        correctness=correctness,
        attempt_penalty=penalty,
        syntax_bonus=syntax_bonus,
        reason=reason,
    )
