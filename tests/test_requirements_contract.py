import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RequirementsContractTest(unittest.TestCase):
    def test_requirements_uses_pip_style_specifiers(self):
        bad_conda_lines = []
        for line_number, line in enumerate((ROOT / "requirements.txt").read_text().splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if re.match(r"^[A-Za-z0-9_.-]+=[^=]", stripped):
                bad_conda_lines.append((line_number, stripped))

        self.assertEqual(bad_conda_lines, [])

    def test_runtime_dependencies_are_declared(self):
        requirement_text = (ROOT / "requirements.txt").read_text()

        for package in (
            "torch",
            "transformers",
            "vllm",
            "jsonlines",
            "tqdm",
            "math-verify",
            "sympy",
            "latex2sympy2-extended",
        ):
            with self.subTest(package=package):
                self.assertIn(package, requirement_text)


if __name__ == "__main__":
    unittest.main()
