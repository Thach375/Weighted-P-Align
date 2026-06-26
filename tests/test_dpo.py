import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rw_palign.dpo import build_dpo_pairs, select_dpo_pair  # noqa: E402
from rw_palign.schemas import GroupStats, RewardedSample  # noqa: E402


def rewarded_sample(group_id, record_id, sample_index, reward, continuation):
    return RewardedSample(
        group_id=group_id,
        record_id=record_id,
        sample_index=sample_index,
        correct=reward > 0,
        reward=reward,
        parsed_answer=str(reward),
        extra={
            "question": f"Question for {record_id}",
            "prefix": f"Prefix for {record_id}",
            "prompt": f"Prompt for {record_id}",
            "continuation": continuation,
        },
    )


class DPOPairSelectionTest(unittest.TestCase):
    def test_mixed_group_pair_selection(self):
        samples = [
            rewarded_sample("rec:v1", "rec", 0, 1.0, "chosen"),
            rewarded_sample("rec:v1", "rec", 1, 0.0, "rejected"),
            rewarded_sample("rec:v1", "rec", 2, 0.0, "also rejected"),
        ]

        pair = select_dpo_pair(samples)

        self.assertIsNotNone(pair)
        self.assertEqual(pair.chosen, "chosen")
        self.assertEqual(pair.rejected_reward, 0.0)
        self.assertGreater(pair.chosen_reward, pair.rejected_reward)

    def test_min_gap_enforced(self):
        samples = [
            rewarded_sample("rec:v1", "rec", 0, 1.0, "chosen"),
            rewarded_sample("rec:v1", "rec", 1, 0.5, "rejected"),
        ]

        self.assertIsNone(select_dpo_pair(samples, min_reward_gap=0.5))

    def test_all_correct_no_pair(self):
        samples = [
            rewarded_sample("rec:v1", "rec", 0, 1.0, "first"),
            rewarded_sample("rec:v1", "rec", 1, 1.0, "second"),
        ]
        groups = [
            GroupStats(
                group_id="rec:v1",
                record_id="rec",
                k=2,
                pass_rate=1.0,
                reward_mean=1.0,
                reward_std=0.0,
                status="all_correct",
            )
        ]

        pairs, metrics = build_dpo_pairs(samples, groups)

        self.assertEqual(pairs, [])
        self.assertEqual(metrics["skipped_non_mixed"], 1)

    def test_dpo_pair_schema(self):
        samples = [
            rewarded_sample("rec:v1", "rec", 0, 1.0, "chosen"),
            rewarded_sample("rec:v1", "rec", 1, 0.0, "rejected"),
        ]

        pair = select_dpo_pair(samples)
        raw = pair.to_dict()

        self.assertIn("prompt", raw)
        self.assertIn("prefix", raw)
        self.assertIn("chosen", raw)
        self.assertIn("rejected", raw)
        self.assertIn("chosen_reward", raw)
        self.assertEqual(raw["metadata"]["chosen_sample_index"], 0)
        self.assertEqual(raw["metadata"]["rejected_sample_index"], 1)


if __name__ == "__main__":
    unittest.main()
