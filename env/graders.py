from __future__ import annotations
from typing import Any, Optional, Tuple
from env.models import Reward


def _first_value(rows: list) -> Optional[Any]:
    if not rows:
        return None
    vals = list(rows[0].values())
    return vals[0] if vals else None


def _col0_list(rows: list) -> list:
    if not rows:
        return []
    key = list(rows[0].keys())[0]
    return [r[key] for r in rows]


def _grade_scalar_int(rows, expected: int) -> Tuple[float, str]:
    val = _first_value(rows)
    if val is None:
        return 0.05, "Query returned no rows."
    try:
        got = int(val)
    except (TypeError, ValueError):
        return 0.05, f"Expected integer, got: {val!r}"
    if got == expected:
        return 0.95, f"Correct! Got {got}."
    return 0.05, f"Expected {expected}, got {got}."


def _grade_scalar_float(rows, expected: float, tol: float = 0.02) -> Tuple[float, str]:
    val = _first_value(rows)
    if val is None:
        return 0.05, "Query returned no rows."
    try:
        got = float(val)
    except (TypeError, ValueError):
        return 0.05, f"Expected float, got: {val!r}"
    diff = abs(got - expected)
    if diff <= tol:
        return 0.95, f"Correct! Got {got:.4f}."
    if diff <= tol * 5:
        return 0.5, f"Close — expected {expected}, got {got:.4f}."
    return 0.05, f"Expected {expected}, got {got:.4f}."


def _grade_scalar_string(rows, expected: str) -> Tuple[float, str]:
    val = _first_value(rows)
    if val is None:
        return 0.05, "Query returned no rows."
    got = str(val).strip()
    if got.lower() == expected.lower():
        return 0.95, f"Correct! Got '{got}'."
    return 0.05, f"Expected '{expected}', got '{got}'."


def _grade_ordered_list(rows, expected: list) -> Tuple[float, str]:
    got = [str(v).strip() for v in _col0_list(rows)]
    exp = [str(v).strip() for v in expected]

    if got == exp:
        return 0.95, "Exact ordered match."

    got_set, exp_set = set(got), set(exp)
    if got_set == exp_set:
        return 0.5, f"Right items, wrong order. Got: {got}"

    overlap = len(got_set & exp_set)
    positional = sum(1 for a, b in zip(got, exp) if a == b)
    score = round(max(positional, overlap) / max(len(exp), 1), 2)
    score = min(max(score, 0.05), 0.85)

    return score, f"Partial ({overlap}/{len(exp)} items). Expected: {exp}. Got: {got}"


def compute_reward(
    rows: Optional[list],
    error: Optional[str],
    expected: Any,
    answer_type: str,
    attempt: int,
    max_attempts: int,
    attempt_penalty: float = 0.10,
) -> Reward:
    syntax_bonus = 0.05 if (error is None and rows is not None) else 0.0

    if error:
        correctness, reason = 0.05, f"SQL error: {error}"
    elif rows is None:
        correctness, reason = 0.05, "No result returned."
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
            correctness, reason = 0.05, f"Unknown answer_type: {answer_type}"

    if correctness >= 0.95:
        syntax_bonus = 0.0

    penalty = round(attempt_penalty * attempt, 4)
    raw = correctness + syntax_bonus - penalty

    # ✅ FIXED: correct parenthesis — returns float not tuple
    value = round(max(0.05, min(0.95, raw)), 4)

    return Reward(
        value=value,
        correctness=correctness,
        attempt_penalty=penalty,
        syntax_bonus=syntax_bonus,
        reason=reason,
    )