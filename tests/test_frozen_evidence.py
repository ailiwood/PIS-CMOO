"""tests/test_frozen_evidence.py — verify the frozen-evidence verification
script can run end-to-end against the distributed artifacts.

Run with:
    python -m unittest tests.test_frozen_evidence
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "scripts" / "verify_frozen_results.py"


class TestFrozenEvidenceScript(unittest.TestCase):
    def test_verify_frozen_results_passes(self):
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.run(
            [sys.executable, str(VERIFY)],
            cwd=str(ROOT),
            capture_output=True, text=True, encoding="utf-8",
            env=env,
            timeout=120,
        )
        self.assertEqual(
            proc.returncode, 0,
            msg=f"verify_frozen_results.py exited {proc.returncode}.\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
        )
        self.assertIsNotNone(proc.stdout, "captured stdout was None")
        self.assertIn("ALL FROZEN-ARTIFACT CHECKS PASS", proc.stdout)


if __name__ == "__main__":
    unittest.main()