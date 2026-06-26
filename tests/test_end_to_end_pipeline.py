import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rw_palign.io import read_jsonl, write_jsonl  # noqa: E402


FIXTURES = ROOT / "tests" / "fixtures"


class FixturePipelineTest(unittest.TestCase):
    def test_fixture_scoring_to_weighted_sft_and_dpo(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            continuations = run_dir / "continuations.jsonl"
            rewarded = run_dir / "rewarded_samples.jsonl"
            group_stats = run_dir / "group_stats.jsonl"
            weighted_sft = run_dir / "weighted_sft.jsonl"
            dpo_pairs = run_dir / "dpo_pairs.jsonl"

            selected_rows = [
                row
                for row in read_jsonl(FIXTURES / "continuations.jsonl")
                if row["record_id"] in {"rec-mixed", "rec-all-wrong"}
            ]
            write_jsonl(continuations, selected_rows)

            subprocess.run(
                [
                    sys.executable,
                    "src/score_rewards.py",
                    "--input",
                    str(continuations),
                    "--output",
                    str(rewarded),
                    "--group_stats",
                    str(group_stats),
                    "--verifier",
                    "local",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "src/build_weighted_sft.py",
                    "--samples",
                    str(rewarded),
                    "--groups",
                    str(group_stats),
                    "--output",
                    str(weighted_sft),
                    "--mode",
                    "L2",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "src/build_dpo_pairs.py",
                    "--samples",
                    str(rewarded),
                    "--groups",
                    str(group_stats),
                    "--output",
                    str(dpo_pairs),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            groups = {row["group_id"]: row for row in read_jsonl(group_stats)}
            weighted_rows = list(read_jsonl(weighted_sft))
            dpo_rows = list(read_jsonl(dpo_pairs))

            self.assertEqual(groups["rec-mixed:v1"]["status"], "mixed")
            self.assertEqual(groups["rec-all-wrong:v1"]["status"], "all_wrong")
            self.assertEqual(len(weighted_rows), 1)
            self.assertEqual(weighted_rows[0]["metadata"]["group_id"], "rec-mixed:v1")
            self.assertAlmostEqual(weighted_rows[0]["normalized_weight"], 1.0)
            self.assertEqual(len(dpo_rows), 1)
            self.assertEqual(dpo_rows[0]["metadata"]["group_id"], "rec-mixed:v1")


if __name__ == "__main__":
    unittest.main()
