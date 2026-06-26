from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from .io import read_jsonl


def aggregate_run_metrics(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    continuations = _read_jsonl_if_exists(run_path / "continuations.jsonl")
    rewarded = _read_jsonl_if_exists(run_path / "rewarded_samples.jsonl")
    groups = _read_jsonl_if_exists(run_path / "group_stats.jsonl")
    weighted = _read_jsonl_if_exists(run_path / "weighted_sft.jsonl")
    dpo_pairs = _read_jsonl_if_exists(run_path / "dpo_pairs.jsonl")

    record_ids = _record_ids(continuations) or _record_ids(rewarded) or _record_ids(groups)
    selected_group_ids = {
        row.get("metadata", {}).get("group_id")
        for row in weighted
        if row.get("metadata", {}).get("group_id")
    }
    group_status_counts = Counter(str(row.get("status", "unknown")) for row in groups)
    avg_pass_rate = mean(float(row.get("pass_rate", 0.0)) for row in groups) if groups else 0.0
    avg_continuation_length = _average_continuation_length(rewarded, continuations)

    return {
        "input_records": len(record_ids),
        "generated_samples": len(continuations),
        "verified_samples": len(rewarded),
        "weighted_sft_examples": len(weighted),
        "skipped_records": max(0, len(record_ids) - len(selected_group_ids)),
        "group_status_counts": dict(sorted(group_status_counts.items())),
        "average_pass_rate": avg_pass_rate,
        "average_selected_weight_count_per_group": _average_selected_weight_count(weighted),
        "average_continuation_length": avg_continuation_length,
        "dpo_pair_count": len(dpo_pairs),
    }


def load_run_config(run_dir: str | Path) -> dict[str, Any]:
    config_path = Path(run_dir) / "config.resolved.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def render_report(metrics: dict[str, Any], config: dict[str, Any] | None = None) -> str:
    config = config or {}
    lines = [
        "# Reward-Weighted P-ALIGN Run Report",
        "",
        "## Metrics",
        "",
        f"- Input records: {metrics.get('input_records', 0)}",
        f"- Generated samples: {metrics.get('generated_samples', 0)}",
        f"- Verified samples: {metrics.get('verified_samples', 0)}",
        f"- Weighted SFT examples: {metrics.get('weighted_sft_examples', 0)}",
        f"- Skipped records: {metrics.get('skipped_records', 0)}",
        f"- Average pass rate: {metrics.get('average_pass_rate', 0.0):.6f}",
        "- Average selected weight count per group: "
        f"{metrics.get('average_selected_weight_count_per_group', 0.0):.6f}",
        f"- Average continuation length: {metrics.get('average_continuation_length', 0.0):.6f}",
        f"- DPO pair count: {metrics.get('dpo_pair_count', 0)}",
        "",
        "## Group Status Counts",
        "",
    ]
    status_counts = metrics.get("group_status_counts", {})
    if status_counts:
        for status, count in sorted(status_counts.items()):
            lines.append(f"- {status}: {count}")
    else:
        lines.append("- none: 0")

    lines.extend(["", "## Config", ""])
    if config:
        for key, value in sorted(config.items()):
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- No config.resolved.json found.")
    lines.append("")
    return "\n".join(lines)


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return list(read_jsonl(path))


def _record_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["record_id"]) for row in rows if row.get("record_id") is not None}


def _average_selected_weight_count(weighted_rows: list[dict[str, Any]]) -> float:
    counts: Counter[str] = Counter()
    for row in weighted_rows:
        group_id = row.get("metadata", {}).get("group_id")
        if group_id:
            counts[str(group_id)] += 1
    return mean(counts.values()) if counts else 0.0


def _average_continuation_length(
    rewarded_rows: list[dict[str, Any]],
    continuation_rows: list[dict[str, Any]],
) -> float:
    lengths = [
        int(row["length_tokens"])
        for row in rewarded_rows
        if isinstance(row.get("length_tokens"), int)
    ]
    if lengths:
        return mean(lengths)

    fallback_lengths = [
        len(str(row.get("continuation", "")).split())
        for row in continuation_rows
        if row.get("continuation")
    ]
    return mean(fallback_lengths) if fallback_lengths else 0.0
