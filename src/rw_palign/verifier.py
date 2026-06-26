from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Callable


Verifier = Callable[[str, str], bool]


_FINAL_ANSWER_PATTERNS = (
    re.compile(r"(?:final answer|answer)\s*(?:is|:)\s*(.+)", re.IGNORECASE),
)


def extract_answer(text: str) -> str | None:
    """Return the final answer candidate, preferring the last boxed answer."""
    boxed_answers = _extract_boxed_answers(text)
    if boxed_answers:
        return _clean_answer(boxed_answers[-1])

    for line in reversed([part.strip() for part in text.splitlines() if part.strip()]):
        for pattern in _FINAL_ANSWER_PATTERNS:
            match = pattern.search(line)
            if match:
                return _clean_answer(match.group(1))
    return None


def verify_answer(
    prediction: str,
    golden: str,
    verifier: Verifier | None = None,
) -> tuple[bool, str | None, str | None]:
    """Return correctness, parsed answer, and optional per-sample error."""
    parsed_answer = extract_answer(prediction)
    if parsed_answer is None:
        return False, None, "could not extract final answer"

    try:
        if verifier is not None:
            return bool(verifier(parsed_answer, str(golden))), parsed_answer, None
        return _default_verify(parsed_answer, str(golden)), parsed_answer, None
    except Exception as exc:  # noqa: BLE001 - verifier failures must not crash a run.
        return False, parsed_answer, str(exc)


def calculate_length_penalty(
    continuation: str,
    penalty_per_token: float = 0.0,
    token_count: int | None = None,
) -> float:
    if penalty_per_token == 0:
        return 0.0
    if penalty_per_token < 0:
        raise ValueError("penalty_per_token must be non-negative")
    length = token_count if token_count is not None else len(continuation.split())
    return length * penalty_per_token


def calculate_reward(
    correct: bool,
    continuation: str,
    penalty_per_token: float = 0.0,
    token_count: int | None = None,
) -> float:
    binary_reward = 1.0 if correct else 0.0
    return binary_reward - calculate_length_penalty(
        continuation,
        penalty_per_token=penalty_per_token,
        token_count=token_count,
    )


def _default_verify(prediction: str, golden: str) -> bool:
    try:
        return _math_verify(prediction, golden)
    except ImportError:
        return _local_verify(prediction, golden)


def _math_verify(prediction: str, golden: str) -> bool:
    from math_verify import parse, verify

    parsed_prediction = parse(prediction)
    parsed_golden = parse("$" + golden + "$")
    return bool(verify(parsed_golden, parsed_prediction))


def _local_verify(prediction: str, golden: str) -> bool:
    normalized_prediction = _normalize_for_compare(prediction)
    normalized_golden = _normalize_for_compare(golden)
    if normalized_prediction == normalized_golden:
        return True

    prediction_number = _to_decimal(normalized_prediction)
    golden_number = _to_decimal(normalized_golden)
    return prediction_number is not None and prediction_number == golden_number


def _extract_boxed_answers(text: str) -> list[str]:
    answers: list[str] = []
    start = 0
    while True:
        marker = text.find(r"\boxed", start)
        if marker == -1:
            return answers

        brace_start = text.find("{", marker + len(r"\boxed"))
        if brace_start == -1:
            start = marker + len(r"\boxed")
            continue

        brace_end = _find_matching_brace(text, brace_start)
        if brace_end is None:
            start = brace_start + 1
            continue

        answers.append(text[brace_start + 1 : brace_end])
        start = brace_end + 1


def _find_matching_brace(text: str, opening_index: int) -> int | None:
    depth = 0
    for index in range(opening_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _clean_answer(answer: str) -> str | None:
    cleaned = answer.strip()
    cleaned = cleaned.strip(" \t\r\n.$")
    cleaned = re.sub(r"^(?:therefore|thus|so)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" \t\r\n.$")
    return cleaned or None


def _normalize_for_compare(value: str) -> str:
    normalized = value.strip()
    normalized = normalized.strip("$")
    normalized = re.sub(r"\\left|\\right", "", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _to_decimal(value: str) -> Decimal | None:
    try:
        return Decimal(value)
    except InvalidOperation:
        return None
