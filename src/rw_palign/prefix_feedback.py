from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PrefixActionDecision:
    action: str
    reason: str


def decide_prefix_action(
    pass_rate: float,
    tau_low: float = 0.2,
    allow_shorten: bool = False,
    feedback_round: int = 0,
    max_feedback_rounds: int = 1,
) -> str:
    """Return keep, extend, shorten_review, or skip for a prefix group."""
    return explain_prefix_action(
        pass_rate=pass_rate,
        tau_low=tau_low,
        allow_shorten=allow_shorten,
        feedback_round=feedback_round,
        max_feedback_rounds=max_feedback_rounds,
    ).action


def explain_prefix_action(
    pass_rate: float,
    tau_low: float = 0.2,
    allow_shorten: bool = False,
    feedback_round: int = 0,
    max_feedback_rounds: int = 1,
) -> PrefixActionDecision:
    _validate_inputs(pass_rate, tau_low, feedback_round, max_feedback_rounds)

    if pass_rate < tau_low:
        if feedback_round >= max_feedback_rounds:
            return PrefixActionDecision(
                action="skip",
                reason="pass_rate below tau_low but max feedback rounds reached",
            )
        return PrefixActionDecision(
            action="extend",
            reason="pass_rate below tau_low",
        )

    if pass_rate == 1.0 and allow_shorten:
        return PrefixActionDecision(
            action="shorten_review",
            reason="all samples correct and shortening review enabled",
        )

    return PrefixActionDecision(
        action="keep",
        reason="pass_rate within configured bounds",
    )


def _validate_inputs(
    pass_rate: float,
    tau_low: float,
    feedback_round: int,
    max_feedback_rounds: int,
) -> None:
    if not math.isfinite(pass_rate) or pass_rate < 0.0 or pass_rate > 1.0:
        raise ValueError("pass_rate must be between 0 and 1")
    if not math.isfinite(tau_low) or tau_low < 0.0 or tau_low > 1.0:
        raise ValueError("tau_low must be between 0 and 1")
    if feedback_round < 0:
        raise ValueError("feedback_round must be non-negative")
    if max_feedback_rounds < 0:
        raise ValueError("max_feedback_rounds must be non-negative")
