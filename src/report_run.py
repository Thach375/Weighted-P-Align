from __future__ import annotations

import argparse
from pathlib import Path

from rw_palign.io import ensure_output_path_available, write_json
from rw_palign.reporting import aggregate_run_metrics, load_run_config, render_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Reward-Weighted P-ALIGN run metrics and report.")
    parser.add_argument("--run_dir", required=True, help="Run directory containing JSONL artifacts.")
    parser.add_argument("--output", required=True, help="Markdown report output path.")
    parser.add_argument("--metrics_output", help="Metrics JSON output path. Defaults to run_dir/metrics.json.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    report_path = Path(args.output)
    metrics_path = Path(args.metrics_output) if args.metrics_output else run_dir / "metrics.json"

    ensure_output_path_available(report_path, overwrite=args.overwrite)
    ensure_output_path_available(metrics_path, overwrite=args.overwrite)

    metrics = aggregate_run_metrics(run_dir)
    config = load_run_config(run_dir)
    write_json(metrics_path, metrics)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(metrics, config), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
