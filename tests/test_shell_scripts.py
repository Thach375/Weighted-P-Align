import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ShellScriptContractTest(unittest.TestCase):
    def test_python_entry_points_exist(self):
        for script_path in (ROOT / "scripts").glob("*.sh"):
            text = script_path.read_text(encoding="utf-8")
            for match in re.finditer(r"python\s+(src/[^\s\\]+\.py)", text):
                target = ROOT / match.group(1)
                with self.subTest(script=script_path.name, target=match.group(1)):
                    self.assertTrue(target.exists())

    def test_cuda_assignments_have_no_space_after_equals(self):
        for script_path in (ROOT / "scripts").glob("*.sh"):
            text = script_path.read_text(encoding="utf-8")
            with self.subTest(script=script_path.name):
                self.assertNotRegex(text, r"CUDA_VISIBLE_DEVICES=\s+")

    def test_train_script_uses_external_training_cli(self):
        text = (ROOT / "scripts" / "train.sh").read_text(encoding="utf-8")

        self.assertNotIn("src/train.py", text)
        self.assertIn("llamafactory-cli train", text)


if __name__ == "__main__":
    unittest.main()
