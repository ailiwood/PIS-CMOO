"""tests/test_imports.py — smoke test for the package import surface.

Run with:
    python -m unittest tests.test_imports

Pass criteria: every public module is importable in isolation, and the
package's __version__ is set.

Note: requires the `src/` directory to be on sys.path. When running tests
from the project root, run with:
    PYTHONPATH=src python -m unittest tests.test_imports
"""
import importlib
import os
import sys
import unittest

# Ensure src/ is on sys.path for direct invocation.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


class TestPackageImports(unittest.TestCase):
    def test_package_imports(self):
        import pis_cmoo
        self.assertTrue(hasattr(pis_cmoo, "__version__"))
        self.assertEqual(pis_cmoo.__version__, "0.1.0-paper")

    def test_subpackage_imports(self):
        for mod in [
            "pis_cmoo.prediction",
            "pis_cmoo.uncertainty",
            "pis_cmoo.interpretation",
            "pis_cmoo.optimization",
            "pis_cmoo.validation",
            "pis_cmoo.utils",
        ]:
            with self.subTest(mod=mod):
                importlib.import_module(mod)


if __name__ == "__main__":
    unittest.main()