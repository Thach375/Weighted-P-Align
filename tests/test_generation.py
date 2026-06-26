import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rw_palign.generation import (  # noqa: E402
    build_generation_config,
    generate_continuation_samples,
    mock_generator,
)
from rw_palign.io import load_prefix_records, read_jsonl, write_jsonl  # noqa: E402
from rw_palign.prompts import build_continuation_prompt  # noqa: E402


FIXTURES = ROOT / "tests" / "fixtures"


class PromptBuilderTest(unittest.TestCase):
    def test_prompt_includes_question_prefix_and_boxed_instruction(self):
        prompt = build_continuation_prompt("What is 2 + 2?", "Two pairs combine.")

        self.assertIn("Please continue from the draft", prompt)
        self.assertIn("step by step", prompt)
        self.assertIn("\\boxed{}", prompt)
        self.assertIn("Question: What is 2 + 2?", prompt)
        self.assertIn("Prefix: Two pairs combine.", prompt)


class GenerationCoreTest(unittest.TestCase):
    def test_mock_generation_outputs_k_samples(self):
        records = load_prefix_records(FIXTURES / "prefix_records.jsonl").records[:2]
        config = build_generation_config(
            model="fixture-model",
            temperature=0.8,
            top_p=0.95,
            max_tokens=128,
            batch_size=4,
            seed=42,
            backend="mock",
        )

        samples = generate_continuation_samples(records, mock_generator, config, k=2)

        self.assertEqual(len(samples), 4)
        self.assertEqual({sample.generation_config["model"] for sample in samples}, {"fixture-model"})
        self.assertTrue(all(sample.prompt.startswith("Please continue") for sample in samples))

    def test_resume_skips_existing_samples(self):
        records = load_prefix_records(FIXTURES / "prefix_records.jsonl").records[:1]
        config = build_generation_config(
            model="fixture-model",
            temperature=0.8,
            top_p=0.95,
            max_tokens=128,
            batch_size=4,
            seed=42,
            backend="mock",
        )

        samples = generate_continuation_samples(
            records,
            mock_generator,
            config,
            k=2,
            resume_keys={(records[0].id, 0)},
        )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].sample_index, 1)

    def test_invalid_k_rejected(self):
        records = load_prefix_records(FIXTURES / "prefix_records.jsonl").records[:1]
        config = build_generation_config(
            model="fixture-model",
            temperature=0.8,
            top_p=0.95,
            max_tokens=128,
            batch_size=4,
            seed=42,
            backend="mock",
        )

        with self.assertRaises(ValueError):
            generate_continuation_samples(records, mock_generator, config, k=0)


class GenerationCLITest(unittest.TestCase):
    def test_generation_config_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run" / "continuations.jsonl"
            command = [
                sys.executable,
                "src/generate_continuations.py",
                "--input",
                str(FIXTURES / "prefix_records.jsonl"),
                "--output",
                str(output),
                "--model",
                "fixture-model",
                "--k",
                "2",
                "--max_tokens",
                "128",
                "--batch_size",
                "4",
                "--backend",
                "mock",
                "--limit",
                "2",
            ]

            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            rows = list(read_jsonl(output))
            config = list(read_jsonl(output))[0]["generation_config"]
            resolved_config = output.parent / "config.resolved.json"

            self.assertEqual(len(rows), 4)
            self.assertEqual(config["model"], "fixture-model")
            self.assertEqual(config["temperature"], 0.8)
            self.assertEqual(config["top_p"], 0.95)
            self.assertEqual(config["max_tokens"], 128)
            self.assertEqual(config["seed"], 42)
            self.assertTrue(resolved_config.exists())

    def test_no_overwrite_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "continuations.jsonl"
            write_jsonl(output, [{"already": "here"}])
            command = [
                sys.executable,
                "src/generate_continuations.py",
                "--input",
                str(FIXTURES / "prefix_records.jsonl"),
                "--output",
                str(output),
                "--model",
                "fixture-model",
                "--backend",
                "mock",
            ]

            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)

            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
