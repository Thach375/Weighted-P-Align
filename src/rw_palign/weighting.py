from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from .schemas import GroupStats, RewardedSample, WeightedSFTExample


DEFAULT_EPSILON = 1e-6
DEFAULT_CLIP = 3.0


def compute_group_stats(rewards: list[float], epsilon: float = DEFAULT_EPSILON) -> dict[str, object]:
    """Return mean, population std, pass rate, advantages, and group status."""
    normalized_rewards = _normalize_rewards(rewards)
    if not normalized_rewards:
        raise ValueError("rewards must contain at least one value")
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be a positive finite value")

    count = len(normalized_rewards)
    reward_mean = sum(normalized_rewards) / count
    reward_std = math.sqrt(sum((reward - reward_mean) ** 2 for reward in normalized_rewards) / count)
    advantages = _compute_advantages(normalized_rewards, reward_mean, reward_std, epsilon)
    positive_count = sum(1 for reward in normalized_rewards if reward > 0)

    return {
        "k": count,
        "pass_rate": positive_count / count,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "advantages": advantages,
        "status": _classify_status(count, positive_count),
    }


def compute_sft_weights(
    rewards: list[float],
    advantages: list[float],
    mode: str,
    clip: float = DEFAULT_CLIP,
) -> list[float]:
    """Return normalized per-group SFT weights for L1 filtering or L2 weighting."""
    normalized_rewards = _normalize_rewards(rewards)
    normalized_advantages = _normalize_rewards(advantages)
    if len(normalized_rewards) != len(normalized_advantages):
        raise ValueError("rewards and advantages must have the same length")
    if not normalized_rewards:
        raise ValueError("rewards must contain at least one value")
    if clip <= 0 or not math.isfinite(clip):
        raise ValueError("clip must be a positive finite value")

    normalized_mode = mode.upper()
    if normalized_mode not in {"L1", "L2"}:
        raise ValueError(f"unsupported weighting mode: {mode}")

    selected_count = sum(1 for reward in normalized_rewards if reward > 0)
    if selected_count == 0:
        return [0.0] * len(normalized_rewards)
    if len(normalized_rewards) == 1:
        return [1.0]

    if normalized_mode == "L1":
        reward_mean = sum(normalized_rewards) / len(normalized_rewards)
        raw_weights = [1.0 if reward >= reward_mean and reward > 0 else 0.0 for reward in normalized_rewards]
    else:
        raw_weights = [min(max(advantage, 0.0), clip) if reward > 0 else 0.0 for reward, advantage in zip(normalized_rewards, normalized_advantages)]
        if sum(raw_weights) == 0.0 and selected_count == len(normalized_rewards):
            raw_weights = [1.0] * len(normalized_rewards)

    return _normalize_weights(raw_weights)


def build_weighted_sft_examples(
    samples: Iterable[RewardedSample],
    groups: Iterable[GroupStats] | None = None,
    mode: str = "L2",
    clip: float = DEFAULT_CLIP,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[list[WeightedSFTExample], dict[str, object]]:
    """Build eligible weighted SFT examples and export metrics."""
    group_lookup = {group.group_id: group for group in groups or []}
    samples_by_group: dict[str, list[RewardedSample]] = defaultdict(list)
    for sample in samples:
        samples_by_group[sample.group_id].append(sample)

    examples: list[WeightedSFTExample] = []
    group_metrics: dict[str, int] = {
        "total_groups": 0,
        "selected_groups": 0,
        "skipped_groups": 0,
        "all_wrong_groups": 0,
        "singleton_groups": 0,
    }

    for group_id in sorted(samples_by_group):
        group_samples = sorted(samples_by_group[group_id], key=lambda item: item.sample_index)
        rewards = [sample.reward for sample in group_samples]
        stats = compute_group_stats(rewards, epsilon=epsilon)
        weights = compute_sft_weights(rewards, stats["advantages"], mode=mode, clip=clip)
        status = group_lookup.get(group_id).status if group_id in group_lookup else str(stats["status"])
        selected_before = len(examples)

        group_metrics["total_groups"] += 1
        if status == "all_wrong":
            group_metrics["all_wrong_groups"] += 1
        if status == "singleton":
            group_metrics["singleton_groups"] += 1

        for sample, weight, advantage in zip(group_samples, weights, stats["advantages"]):
            if weight <= 0.0:
                continue
            examples.append(
                WeightedSFTExample(
                    id=f"{sample.record_id}/{sample.sample_index}",
                    prompt=_required_extra(sample, "prompt"),
                    question=_required_extra(sample, "question"),
                    prefix=_required_extra(sample, "prefix"),
                    continuation=_required_extra(sample, "continuation"),
                    text=_join_supervision_text(
                        _required_extra(sample, "prefix"),
                        _required_extra(sample, "continuation"),
                    ),
                    weight=weight,
                    normalized_weight=weight,
                    reward=sample.reward,
                    advantage=float(advantage),
                    metadata={
                        "group_id": sample.group_id,
                        "record_id": sample.record_id,
                        "sample_index": sample.sample_index,
                        "weighting_mode": mode.upper(),
                        "group_status": status,
                        "reward_mean": stats["reward_mean"],
                        "reward_std": stats["reward_std"],
                        "parsed_answer": sample.parsed_answer,
                    },
                )
            )

        if len(examples) > selected_before:
            group_metrics["selected_groups"] += 1
        else:
            group_metrics["skipped_groups"] += 1

    metrics: dict[str, object] = {
        "input_samples": sum(len(group_samples) for group_samples in samples_by_group.values()),
        "output_examples": len(examples),
        "weighting_mode": mode.upper(),
        **group_metrics,
    }
    return examples, metrics


def _compute_advantages(
    rewards: list[float],
    reward_mean: float,
    reward_std: float,
    epsilon: float,
) -> list[float]:
    if reward_std == 0.0:
        return [0.0] * len(rewards)
    return [(reward - reward_mean) / (reward_std + epsilon) for reward in rewards]


def _classify_status(count: int, positive_count: int) -> str:
    if count == 1:
        return "singleton"
    if positive_count == 0:
        return "all_wrong"
    if positive_count == count:
        return "all_correct"
    return "mixed"


def _normalize_weights(weights: list[float]) -> list[float]:
    total = sum(weights)
    if total == 0.0:
        return [0.0] * len(weights)
    return [weight / total for weight in weights]


def _normalize_rewards(values: Iterable[float]) -> list[float]:
    normalized = [float(value) for value in values]
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError("values must be finite")
    return normalized


def _required_extra(sample: RewardedSample, field_name: str) -> str:
    value = sample.extra.get(field_name)
    if value is None or value == "":
        raise ValueError(
            f"rewarded sample {sample.record_id}/{sample.sample_index} is missing {field_name}"
        )
    return str(value)


def _join_supervision_text(prefix: str, continuation: str) -> str:
    if not prefix:
        return continuation
    if not continuation:
        return prefix
    return f"{prefix.rstrip()}\n{continuation.lstrip()}"
