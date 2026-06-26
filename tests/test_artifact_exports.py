import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rw_palign.schemas import GroupStats, RewardedSample, WeightedSFTExample  # noqa: E402
from rw_palign.weighting import build_weighted_sft_examples  # noqa: E402


def rewarded_sample(group_id, record_id, sample_index, reward, continuation):
    return RewardedSample(
        group_id=group_id,
        record_id=record_id,
        sample_index=sample_index,
        correct=reward > 0,
        reward=reward,
        parsed_answer=str(int(reward)),
        extra={
            "question": f"Question for {record_id}",
            "answer": "4",
            "prefix": f"Prefix for {record_id}",
            "prompt": f"Prompt for {record_id}",
            "continuation": continuation,
        },
    )


class WeightedSFTExportTest(unittest.TestCase):
    def test_weighted_sft_schema(self):
        samples = [
            rewarded_sample("rec-mixed:v1", "rec-mixed", 0, 1.0, "good"),
            rewarded_sample("rec-mixed:v1", "rec-mixed", 1, 0.0, "bad"),
        ]
        groups = [
            GroupStats(
                group_id="rec-mixed:v1",
                record_id="rec-mixed",
                k=2,
                pass_rate=0.5,
                reward_mean=0.5,
                reward_std=0.5,
                status="mixed",
            )
        ]

        examples, _ = build_weighted_sft_examples(samples, groups, mode="L2")
        raw = examples[0].to_dict()

        self.assertIsInstance(examples[0], WeightedSFTExample)
        self.assertIn("prompt", raw)
        self.assertIn("prefix", raw)
        self.assertIn("continuation", raw)
        self.assertIn("text", raw)
        self.assertIn("normalized_weight", raw)
        self.assertIn("metadata", raw)

    def test_weight_sum_by_group(self):
        samples = [
            rewarded_sample("rec-all-correct:v1", "rec-all-correct", 0, 1.0, "first"),
            rewarded_sample("rec-all-correct:v1", "rec-all-correct", 1, 1.0, "second"),
        ]

        examples, _ = build_weighted_sft_examples(samples, mode="L2")

        self.assertEqual(len(examples), 2)
        self.assertAlmostEqual(sum(example.normalized_weight for example in examples), 1.0)

    def test_trace_metadata_present(self):
        samples = [
            rewarded_sample("rec-mixed:v1", "rec-mixed", 0, 1.0, "good"),
            rewarded_sample("rec-mixed:v1", "rec-mixed", 1, 0.0, "bad"),
        ]

        examples, _ = build_weighted_sft_examples(samples, mode="L1")

        self.assertEqual(examples[0].metadata["group_id"], "rec-mixed:v1")
        self.assertEqual(examples[0].metadata["record_id"], "rec-mixed")
        self.assertEqual(examples[0].metadata["sample_index"], 0)
        self.assertEqual(examples[0].metadata["weighting_mode"], "L1")

    def test_skipped_group_reported(self):
        samples = [
            rewarded_sample("rec-all-wrong:v1", "rec-all-wrong", 0, 0.0, "bad"),
            rewarded_sample("rec-all-wrong:v1", "rec-all-wrong", 1, 0.0, "also bad"),
        ]

        examples, metrics = build_weighted_sft_examples(samples, mode="L2")

        self.assertEqual(examples, [])
        self.assertEqual(metrics["skipped_groups"], 1)
        self.assertEqual(metrics["all_wrong_groups"], 1)


if __name__ == "__main__":
    unittest.main()
