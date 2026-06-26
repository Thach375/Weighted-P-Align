import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CLIHelpTest(unittest.TestCase):
    def test_cli_help_commands_exit_zero(self):
        for script in (
            "src/generate_continuations.py",
            "src/score_rewards.py",
            "src/build_weighted_sft.py",
            "src/build_dpo_pairs.py",
            "src/report_run.py",
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


if __name__ == "__main__":
    unittest.main()
