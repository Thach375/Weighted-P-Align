import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rw_palign.verifier import (  # noqa: E402
    calculate_length_penalty,
    calculate_reward,
    extract_answer,
    verify_answer,
)


class AnswerExtractionTest(unittest.TestCase):
    def test_extracts_final_boxed_answer(self):
        text = "A first attempt gives \\boxed{3}. Rechecking gives \\boxed{4}."

        self.assertEqual(extract_answer(text), "4")

    def test_extracts_nested_boxed_answer(self):
        text = "The simplified expression is \\boxed{\\frac{1}{2}}."

        self.assertEqual(extract_answer(text), "\\frac{1}{2}")

    def test_unboxed_fallback_uses_final_answer_phrase(self):
        text = "Compute directly. Therefore, the final answer is 142."

        self.assertEqual(extract_answer(text), "142")

    def test_unparseable_answer_returns_none(self):
        self.assertIsNone(extract_answer("The calculation is unfinished."))


class AnswerVerificationTest(unittest.TestCase):
    def test_numeric_equivalence(self):
        correct, parsed_answer, error = verify_answer("Therefore \\boxed{142}.", "142.0")

        self.assertTrue(correct)
        self.assertEqual(parsed_answer, "142")
        self.assertIsNone(error)

    def test_exact_text_equivalence_after_normalization(self):
        correct, parsed_answer, error = verify_answer("The final answer is \\boxed{x + 1}.", "x+1")

        self.assertTrue(correct)
        self.assertEqual(parsed_answer, "x + 1")
        self.assertIsNone(error)

    def test_missing_answer_records_error(self):
        correct, parsed_answer, error = verify_answer("No conclusion here.", "4")

        self.assertFalse(correct)
        self.assertIsNone(parsed_answer)
        self.assertIn("could not extract", error)

    def test_verifier_exception_records_error(self):
        def failing_verifier(prediction, golden):
            raise RuntimeError("verifier unavailable")

        correct, parsed_answer, error = verify_answer(
            "The final answer is \\boxed{4}.",
            "4",
            verifier=failing_verifier,
        )

        self.assertFalse(correct)
        self.assertEqual(parsed_answer, "4")
        self.assertIn("verifier unavailable", error)


class RewardShapingTest(unittest.TestCase):
    def test_zero_length_penalty_by_default(self):
        self.assertEqual(calculate_length_penalty("one two three"), 0.0)
        self.assertEqual(calculate_reward(True, "one two three"), 1.0)
        self.assertEqual(calculate_reward(False, "one two three"), 0.0)

    def test_nonzero_length_penalty_is_subtracted_from_binary_reward(self):
        continuation = "one two three four"

        self.assertEqual(calculate_length_penalty(continuation, penalty_per_token=0.05), 0.2)
        self.assertAlmostEqual(calculate_reward(True, continuation, penalty_per_token=0.05), 0.8)
        self.assertAlmostEqual(calculate_reward(False, continuation, penalty_per_token=0.05), -0.2)


if __name__ == "__main__":
    unittest.main()
