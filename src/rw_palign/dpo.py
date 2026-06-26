from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .schemas import DPOPair, GroupStats, RewardedSample


def select_dpo_pair(
    samples: list[RewardedSample],
    min_reward_gap: float = 0.0,
    pair_index: int = 0,
) -> DPOPair | None:
    """Return one best-vs-worst preference pair when the reward gap is clear."""
    if not samples:
        return None

    ordered = sorted(samples, key=lambda sample: sample.sample_index)
    chosen = max(ordered, key=lambda sample: (sample.reward, -sample.sample_index))
    rejected = min(ordered, key=lambda sample: (sample.reward, sample.sample_index))
    reward_gap = chosen.reward - rejected.reward
    if reward_gap <= min_reward_gap:
        return None

    return DPOPair(
        id=f"{chosen.group_id}/{pair_index}",
        prompt=_required_extra(chosen, "prompt"),
        question=_required_extra(chosen, "question"),
        prefix=_required_extra(chosen, "prefix"),
        chosen=_required_extra(chosen, "continuation"),
        rejected=_required_extra(rejected, "continuation"),
        chosen_reward=chosen.reward,
        rejected_reward=rejected.reward,
        reward_gap=reward_gap,
        metadata={
            "group_id": chosen.group_id,
            "record_id": chosen.record_id,
            "chosen_sample_index": chosen.sample_index,
            "rejected_sample_index": rejected.sample_index,
            "chosen_parsed_answer": chosen.parsed_answer,
            "rejected_parsed_answer": rejected.parsed_answer,
        },
    )


def build_dpo_pairs(
    samples: Iterable[RewardedSample],
    groups: Iterable[GroupStats],
    min_reward_gap: float = 0.0,
) -> tuple[list[DPOPair], dict[str, object]]:
    group_lookup = {group.group_id: group for group in groups}
    samples_by_group: dict[str, list[RewardedSample]] = defaultdict(list)
    for sample in samples:
        samples_by_group[sample.group_id].append(sample)

    pairs: list[DPOPair] = []
    metrics = {
        "total_groups": len(samples_by_group),
        "pair_count": 0,
        "skipped_non_mixed": 0,
        "skipped_gap": 0,
        "min_reward_gap": min_reward_gap,
    }

    for group_id in sorted(samples_by_group):
        group = group_lookup.get(group_id)
        if group is None or group.status != "mixed":
            metrics["skipped_non_mixed"] += 1
            continue

        pair = select_dpo_pair(
            samples_by_group[group_id],
            min_reward_gap=min_reward_gap,
            pair_index=len(pairs),
        )
        if pair is None:
            metrics["skipped_gap"] += 1
            continue
        pair.metadata["group_status"] = group.status
        pairs.append(pair)

    metrics["pair_count"] = len(pairs)
    return pairs, metrics


def _required_extra(sample: RewardedSample, field_name: str) -> str:
    value = sample.extra.get(field_name)
    if value is None or value == "":
        raise ValueError(
            f"rewarded sample {sample.record_id}/{sample.sample_index} is missing {field_name}"
        )
    return str(value)
