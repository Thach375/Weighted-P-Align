import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rw_palign.io import write_jsonl  # noqa: E402
from rw_palign.reporting import aggregate_run_metrics, render_report  # noqa: E402


class RunReportingTest(unittest.TestCase):
    def test_metrics_match_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_fixture_run(run_dir)

            metrics = aggregate_run_metrics(run_dir)

            self.assertEqual(metrics["input_records"], 2)
            self.assertEqual(metrics["generated_samples"], 4)
            self.assertEqual(metrics["verified_samples"], 4)
            self.assertEqual(metrics["weighted_sft_examples"], 1)
            self.assertEqual(metrics["dpo_pair_count"], 1)
            self.assertEqual(metrics["skipped_records"], 1)

    def test_group_status_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_fixture_run(run_dir)

            metrics = aggregate_run_metrics(run_dir)

            self.assertEqual(metrics["group_status_counts"]["mixed"], 1)
            self.assertEqual(metrics["group_status_counts"]["all_wrong"], 1)

    def test_report_contains_key_metrics(self):
        metrics = {
            "input_records": 2,
            "generated_samples": 4,
            "verified_samples": 4,
            "weighted_sft_examples": 1,
            "skipped_records": 1,
            "average_pass_rate": 0.25,
            "average_selected_weight_count_per_group": 1.0,
            "average_continuation_length": 5.0,
            "dpo_pair_count": 1,
            "group_status_counts": {"mixed": 1},
        }

        report = render_report(metrics, {"model": "fixture"})

        self.assertIn("Average pass rate", report)
        self.assertIn("DPO pair count", report)
        self.assertIn("model: fixture", report)


def _write_fixture_run(run_dir):
    write_jsonl(
        run_dir / "continuations.jsonl",
        [
            {"record_id": "rec-mixed", "sample_index": 0, "continuation": "good answer"},
            {"record_id": "rec-mixed", "sample_index": 1, "continuation": "bad answer"},
            {"record_id": "rec-wrong", "sample_index": 0, "continuation": "bad"},
            {"record_id": "rec-wrong", "sample_index": 1, "continuation": "also bad"},
        ],
    )
    write_jsonl(
        run_dir / "rewarded_samples.jsonl",
        [
            {"record_id": "rec-mixed", "group_id": "rec-mixed:v1", "sample_index": 0, "length_tokens": 2},
            {"record_id": "rec-mixed", "group_id": "rec-mixed:v1", "sample_index": 1, "length_tokens": 2},
            {"record_id": "rec-wrong", "group_id": "rec-wrong:v1", "sample_index": 0, "length_tokens": 1},
            {"record_id": "rec-wrong", "group_id": "rec-wrong:v1", "sample_index": 1, "length_tokens": 2},
        ],
    )
    write_jsonl(
        run_dir / "group_stats.jsonl",
        [
            {
                "record_id": "rec-mixed",
                "group_id": "rec-mixed:v1",
                "status": "mixed",
                "pass_rate": 0.5,
            },
            {
                "record_id": "rec-wrong",
                "group_id": "rec-wrong:v1",
                "status": "all_wrong",
                "pass_rate": 0.0,
            },
        ],
    )
    write_jsonl(
        run_dir / "weighted_sft.jsonl",
        [{"metadata": {"group_id": "rec-mixed:v1"}}],
    )
    write_jsonl(
        run_dir / "dpo_pairs.jsonl",
        [{"metadata": {"group_id": "rec-mixed:v1"}}],
    )


if __name__ == "__main__":
    unittest.main()
