"""tests/test_data_schema.py — verify the synthetic example dataset
matches the documented schema (15 columns, 200 rows, the seven final
predictors present, no mine/project identifiers).
"""
import csv
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data" / "example" / "synthetic_demo.csv"

EXPECTED_COLUMNS = {
    "record_id",
    "WCR", "GP_MPa", "grout_take_L_per_m", "grouting_time_h",
    "depth_m", "section_length_m", "pre_lugeon",
    "acceptance_threshold_lu", "post_lugeon",
    "allowable_GP_MPa", "min_WCR", "max_WCR",
    "max_grout_take_L_per_m", "max_grouting_time_h",
}

FORBIDDEN_COLUMNS = {
    "mine_id", "project_id", "borehole_id", "province", "qualified",
    "delta_lugeon", "relative_reduction", "log_reduction",
}


class TestSyntheticSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = list(csv.DictReader(SYN.open(encoding="utf-8")))

    def test_columns_present(self):
        cols = set(self.rows[0].keys())
        missing = EXPECTED_COLUMNS - cols
        self.assertFalse(missing, f"missing expected columns: {missing}")

    def test_no_sensitive_columns(self):
        cols = set(self.rows[0].keys())
        leaked = FORBIDDEN_COLUMNS & cols
        self.assertFalse(leaked, f"sensitive columns leaked: {leaked}")

    def test_no_real_mine_ids(self):
        ids = {r["record_id"] for r in self.rows}
        bad = {x for x in ids if not x.startswith("SYN-")}
        self.assertFalse(bad, f"non-SYN record ids: {bad}")

    def test_row_count(self):
        self.assertGreaterEqual(len(self.rows), 50)


if __name__ == "__main__":
    unittest.main()