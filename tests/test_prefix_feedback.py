import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rw_palign.prefix_feedback import decide_prefix_action, explain_prefix_action  # noqa: E402


class PrefixFeedbackTest(unittest.TestCase):
    def test_low_pass_rate_extends(self):
        self.assertEqual(decide_prefix_action(0.0, tau_low=0.2), "extend")

    def test_mixed_group_kept(self):
        self.assertEqual(decide_prefix_action(0.5, tau_low=0.2), "keep")

    def test_all_correct_shorten_disabled(self):
        self.assertEqual(decide_prefix_action(1.0, allow_shorten=False), "keep")

    def test_all_correct_shorten_enabled(self):
        decision = explain_prefix_action(1.0, allow_shorten=True)

        self.assertEqual(decision.action, "shorten_review")
        self.assertIn("shortening", decision.reason)

    def test_feedback_round_cap(self):
        self.assertEqual(
            decide_prefix_action(
                0.0,
                tau_low=0.2,
                feedback_round=1,
                max_feedback_rounds=1,
            ),
            "skip",
        )

    def test_invalid_pass_rate_rejected(self):
        with self.assertRaises(ValueError):
            decide_prefix_action(1.5)


if __name__ == "__main__":
    unittest.main()
