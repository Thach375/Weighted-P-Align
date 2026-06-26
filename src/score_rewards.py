from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from rw_palign.io import (
    ensure_output_path_available,
    load_continuation_samples,
    write_jsonl,
)
from rw_palign.schemas import GroupStats, RewardedSample
from rw_palign.verifier import calculate_length_penalty, calculate_reward, local_verify_answer, verify_answer
from rw_palign.weighting import DEFAULT_EPSILON, compute_group_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score continuation rewards and group stats.")
    parser.add_argument("--input", required=True, help="ContinuationSample JSONL path.")
    parser.add_argument("--output", required=True, help="RewardedSample JSONL output path.")
    parser.add_argument("--group_stats", required=True, help="GroupStats JSONL output path.")
    parser.add_argument(
        "--verifier",
        default="math_verify",
        choices=("math_verify", "local"),
        help="Verifier mode. math_verify falls back to local exact/numeric matching if unavailable.",
    )
    parser.add_argument("--length_penalty", type=float, default=0.0, help="Penalty per whitespace token.")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON, help="Stats epsilon.")
    parser.add_argument("--strict", action="store_true", help="Fail on malformed input rows.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    group_stats_path = Path(args.group_stats)

    ensure_output_path_available(output_path, overwrite=args.overwrite)
    ensure_output_path_available(group_stats_path, overwrite=args.overwrite)

    result = load_continuation_samples(args.input, strict=args.strict)
    verifier = local_verify_answer if args.verifier == "local" else None
    rewarded_samples = [
        _score_sample(sample, verifier=verifier, verifier_name=args.verifier, length_penalty=args.length_penalty)
        for sample in result.records
    ]
    group_stats = _build_group_stats(rewarded_samples, epsilon=args.epsilon)

    write_jsonl(output_path, rewarded_samples)
    write_jsonl(group_stats_path, group_stats)
    return 0


def _score_sample(sample, verifier, verifier_name: str, length_penalty: float) -> RewardedSample:
    correct, parsed_answer, error = verify_answer(
        sample.continuation,
        str(sample.answer),
        verifier=verifier,
    )
    length_tokens = len(sample.continuation.split())
    penalty = calculate_length_penalty(
        sample.continuation,
        penalty_per_token=length_penalty,
        token_count=length_tokens,
    )
    return RewardedSample(
        group_id=sample.group_id,
        record_id=sample.record_id,
        sample_index=sample.sample_index,
        correct=correct,
        reward=calculate_reward(
            correct,
            sample.continuation,
            penalty_per_token=length_penalty,
            token_count=length_tokens,
        ),
        parsed_answer=parsed_answer,
        length_tokens=length_tokens,
        length_penalty=penalty,
        verifier=verifier_name,
        verifier_error=error,
        extra={
            "question": sample.question,
            "answer": sample.answer,
            "prefix": sample.prefix,
            "prompt": sample.prompt,
            "continuation": sample.continuation,
            "generation_config": sample.generation_config,
        },
    )


def _build_group_stats(samples: list[RewardedSample], epsilon: float) -> list[GroupStats]:
    samples_by_group: dict[str, list[RewardedSample]] = defaultdict(list)
    for sample in samples:
        samples_by_group[sample.group_id].append(sample)

    rows: list[GroupStats] = []
    for group_id in sorted(samples_by_group):
        group_samples = sorted(samples_by_group[group_id], key=lambda item: item.sample_index)
        stats = compute_group_stats([sample.reward for sample in group_samples], epsilon=epsilon)
        rows.append(
            GroupStats(
                group_id=group_id,
                record_id=group_samples[0].record_id,
                k=int(stats["k"]),
                pass_rate=float(stats["pass_rate"]),
                reward_mean=float(stats["reward_mean"]),
                reward_std=float(stats["reward_std"]),
                status=str(stats["status"]),
                extra={"advantages": stats["advantages"]},
            )
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
