from __future__ import annotations

import argparse
from pathlib import Path

from rw_palign.io import (
    ensure_output_path_available,
    load_group_stats,
    load_rewarded_samples,
    write_json,
    write_jsonl,
)
from rw_palign.weighting import DEFAULT_CLIP, DEFAULT_EPSILON, build_weighted_sft_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reward-weighted SFT JSONL examples.")
    parser.add_argument("--samples", required=True, help="RewardedSample JSONL path.")
    parser.add_argument("--groups", required=True, help="GroupStats JSONL path.")
    parser.add_argument("--output", required=True, help="Weighted SFT JSONL output path.")
    parser.add_argument("--metrics", help="Metrics JSON output path.")
    parser.add_argument("--mode", default="L2", choices=("L1", "L2"), help="Weighting mode.")
    parser.add_argument("--clip", type=float, default=DEFAULT_CLIP, help="L2 advantage clip.")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON, help="Stats epsilon.")
    parser.add_argument("--strict", action="store_true", help="Fail on malformed input rows.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    metrics_path = Path(args.metrics) if args.metrics else output_path.with_suffix(".metrics.json")

    ensure_output_path_available(output_path, overwrite=args.overwrite)
    ensure_output_path_available(metrics_path, overwrite=args.overwrite)

    sample_result = load_rewarded_samples(args.samples, strict=args.strict)
    group_result = load_group_stats(args.groups, strict=args.strict)
    examples, metrics = build_weighted_sft_examples(
        sample_result.records,
        group_result.records,
        mode=args.mode,
        clip=args.clip,
        epsilon=args.epsilon,
    )
    metrics["sample_input_errors"] = len(sample_result.errors)
    metrics["group_input_errors"] = len(group_result.errors)

    write_jsonl(output_path, examples)
    write_json(metrics_path, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
