import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LegacyCLIHelpTest(unittest.TestCase):
    def test_legacy_cli_help_works_without_model_dependencies(self):
        for script in (
            "src/binary_search.py",
            "src/prefix-alignment.py",
            "src/test.py",
            "src/evaluation.py",
        ):
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, script, "--help"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )

                self.assertEqual(result.returncode, 0, msg=result.stderr)
                self.assertIn("usage:", result.stdout)


class InferenceScriptTest(unittest.TestCase):
    def test_build_prompt_uses_problem_text(self):
        inference = load_module("src/test.py", "legacy_inference")

        prompt = inference.build_prompt("What is 2 + 2?")

        self.assertIn("What is 2 + 2?", prompt)
        self.assertIn("\\boxed{}", prompt)

    def test_split_list_batches_by_size(self):
        inference = load_module("src/test.py", "legacy_inference_split")

        self.assertEqual(inference.split_list([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])


class EvaluationScriptTest(unittest.TestCase):
    def test_evaluate_jsonl_writes_metrics_without_oat_dependency(self):
        evaluation = load_module("src/evaluation.py", "legacy_evaluation")

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.jsonl"
            output_path = Path(tmp) / "output.jsonl"
            rows = [
                {"answer": "4", "output": ["The final answer is \\boxed{4}."]},
                {"answer": "3", "output": ["The final answer is \\boxed{4}."]},
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            metrics = evaluation.evaluate_jsonl(
                str(input_path),
                str(output_path),
                use_oat=False,
            )

            self.assertEqual(metrics["total"], 2)
            self.assertEqual(metrics["pass_count"], 1)
            self.assertEqual(metrics["pass_at_1"], 0.5)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
