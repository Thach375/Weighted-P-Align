import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rw_palign.weighting import compute_group_stats, compute_sft_weights  # noqa: E402


class GroupStatsTest(unittest.TestCase):
    def test_mixed_group_stats(self):
        stats = compute_group_stats([1.0, 0.0, 0.0, 0.0])

        self.assertEqual(stats["k"], 4)
        self.assertEqual(stats["status"], "mixed")
        self.assertEqual(stats["pass_rate"], 0.25)
        self.assertEqual(stats["reward_mean"], 0.25)
        self.assertAlmostEqual(stats["reward_std"], 0.4330127019)
        self.assertEqual(len(stats["advantages"]), 4)
        self.assertTrue(all(math.isfinite(value) for value in stats["advantages"]))
        self.assertGreater(stats["advantages"][0], 0)
        self.assertLess(stats["advantages"][1], 0)

    def test_all_correct_zero_std(self):
        stats = compute_group_stats([1.0, 1.0, 1.0, 1.0])

        self.assertEqual(stats["status"], "all_correct")
        self.assertEqual(stats["pass_rate"], 1.0)
        self.assertEqual(stats["reward_std"], 0.0)
        self.assertEqual(stats["advantages"], [0.0, 0.0, 0.0, 0.0])
        self.assertTrue(all(math.isfinite(value) for value in stats["advantages"]))

    def test_all_wrong_zero_std(self):
        stats = compute_group_stats([0.0, 0.0, 0.0, 0.0])

        self.assertEqual(stats["status"], "all_wrong")
        self.assertEqual(stats["pass_rate"], 0.0)
        self.assertEqual(stats["reward_std"], 0.0)
        self.assertEqual(stats["advantages"], [0.0, 0.0, 0.0, 0.0])

    def test_singleton_status(self):
        stats = compute_group_stats([1.0])

        self.assertEqual(stats["status"], "singleton")
        self.assertEqual(stats["k"], 1)
        self.assertEqual(stats["pass_rate"], 1.0)
        self.assertEqual(stats["reward_mean"], 1.0)
        self.assertEqual(stats["reward_std"], 0.0)
        self.assertEqual(stats["advantages"], [0.0])

    def test_empty_rewards_rejected(self):
        with self.assertRaises(ValueError):
            compute_group_stats([])


class SFTWeightingTest(unittest.TestCase):
    def test_l1_keeps_above_mean(self):
        stats = compute_group_stats([1.0, 0.0, 0.0, 0.0])

        weights = compute_sft_weights([1.0, 0.0, 0.0, 0.0], stats["advantages"], mode="L1")

        self.assertEqual(weights, [1.0, 0.0, 0.0, 0.0])

    def test_l1_keeps_equal_all_correct_weights(self):
        stats = compute_group_stats([1.0, 1.0, 1.0, 1.0])

        weights = compute_sft_weights([1.0, 1.0, 1.0, 1.0], stats["advantages"], mode="L1")

        self.assertEqual(weights, [0.25, 0.25, 0.25, 0.25])
        self.assertAlmostEqual(sum(weights), 1.0)

    def test_l2_clips_and_normalizes(self):
        weights = compute_sft_weights(
            rewards=[3.0, 2.0, 1.0],
            advantages=[10.0, 1.0, -1.0],
            mode="L2",
            clip=3.0,
        )

        self.assertEqual(weights, [0.75, 0.25, 0.0])
        self.assertAlmostEqual(sum(weights), 1.0)

    def test_equal_all_correct_weights(self):
        stats = compute_group_stats([1.0, 1.0, 1.0, 1.0])

        weights = compute_sft_weights([1.0, 1.0, 1.0, 1.0], stats["advantages"], mode="L2")

        self.assertEqual(weights, [0.25, 0.25, 0.25, 0.25])
        self.assertAlmostEqual(sum(weights), 1.0)

    def test_all_wrong_skipped(self):
        stats = compute_group_stats([0.0, 0.0, 0.0, 0.0])

        weights = compute_sft_weights([0.0, 0.0, 0.0, 0.0], stats["advantages"], mode="L2")

        self.assertEqual(weights, [0.0, 0.0, 0.0, 0.0])

    def test_k1_baseline_behavior(self):
        correct_stats = compute_group_stats([1.0])
        wrong_stats = compute_group_stats([0.0])

        self.assertEqual(compute_sft_weights([1.0], correct_stats["advantages"], mode="L2"), [1.0])
        self.assertEqual(compute_sft_weights([0.0], wrong_stats["advantages"], mode="L2"), [0.0])

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            compute_sft_weights([1.0], [0.0], mode="unknown")


if __name__ == "__main__":
    unittest.main()
