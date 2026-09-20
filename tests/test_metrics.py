"""tests/test_metrics.py — unit tests for the headline metric formulas.

These tests do NOT touch any model or frozen artifact. They verify the
closed-form definitions of MCC, balanced accuracy, sensitivity, specificity
against sklearn reference values.

Run with:
    python -m unittest tests.test_metrics
"""
import unittest

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
)


def _sensitivity(tp, fn):
    return tp / (tp + fn) if (tp + fn) else float("nan")


def _specificity(tn, fp):
    return tn / (tn + fp) if (tn + fp) else float("nan")


def _balanced_accuracy(tp, fn, tn, fp):
    return (_sensitivity(tp, fn) + _specificity(tn, fp)) / 2


def _mcc(tp, fp, tn, fn):
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return num / den if den else float("nan")


class TestSC_M01_Metrics(unittest.TestCase):
    """Verify SC-M01 (n=269) classification metrics against the
    reconciliation-derived authoritative values."""

    def test_tabdpt_cm_sums_to_269(self):
        tp, fp, tn, fn = 62, 12, 165, 30
        self.assertEqual(tp + fp + tn + fn, 269)

    def test_tabdpt_mcc_formula_matches_sklearn(self):
        tp, fp, tn, fn = 62, 12, 165, 30
        y_true = np.array([1] * (tp + fn) + [0] * (tn + fp))
        y_pred = np.array([1] * tp + [0] * fn + [1] * fp + [0] * tn)
        sklearn_mcc = float(matthews_corrcoef(y_true, y_pred))
        formula_mcc = _mcc(tp, fp, tn, fn)
        self.assertAlmostEqual(sklearn_mcc, formula_mcc, places=10)
        # Reconciliation value (10-decimal): 0.6438777834
        self.assertAlmostEqual(formula_mcc, 0.6438777834, places=10)

    def test_tabdpt_balanced_accuracy(self):
        tp, fp, tn, fn = 62, 12, 165, 30
        sens = _sensitivity(tp, fn)
        spec = _specificity(tn, fp)
        self.assertAlmostEqual(sens, 0.6739130435, places=10)
        self.assertAlmostEqual(spec, 0.9322033898, places=10)
        bacc = _balanced_accuracy(tp, fn, tn, fp)
        self.assertAlmostEqual(bacc, 0.8030582167, places=10)
        sklearn_bacc = float(balanced_accuracy_score(
            np.array([1] * (tp + fn) + [0] * (tn + fp)),
            np.array([1] * tp + [0] * fn + [1] * fp + [0] * tn),
        ))
        self.assertAlmostEqual(sklearn_bacc, bacc, places=10)

    def test_catboost_cm_sums_to_269(self):
        tp, fp, tn, fn = 55, 9, 168, 37
        self.assertEqual(tp + fp + tn + fn, 269)

    def test_catboost_mcc_formula_matches_sklearn(self):
        tp, fp, tn, fn = 55, 9, 168, 37
        y_true = np.array([1] * (tp + fn) + [0] * (tn + fp))
        y_pred = np.array([1] * tp + [0] * fn + [1] * fp + [0] * tn)
        sklearn_mcc = float(matthews_corrcoef(y_true, y_pred))
        formula_mcc = _mcc(tp, fp, tn, fn)
        self.assertAlmostEqual(sklearn_mcc, formula_mcc, places=10)
        self.assertAlmostEqual(formula_mcc, 0.6093740948, places=10)


class TestCMsAreInternallyConsistent(unittest.TestCase):
    """Both cohorts (275 script-style and 269 prediction-evaluable) should
    yield CMs whose MCC formula reproduces the sklearn reference value."""

    def test_tabdpt_n_275(self):
        tp, fp, tn, fn = 62, 12, 171, 30
        self.assertEqual(tp + fp + tn + fn, 275)
        self.assertAlmostEqual(_mcc(tp, fp, tn, fn), 0.6472192478, places=10)

    def test_catboost_n_275(self):
        tp, fp, tn, fn = 55, 9, 174, 37
        self.assertEqual(tp + fp + tn + fn, 275)
        self.assertAlmostEqual(_mcc(tp, fp, tn, fn), 0.6126046238, places=10)


if __name__ == "__main__":
    unittest.main()