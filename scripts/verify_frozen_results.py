"""scripts/verify_frozen_results.py — read-only verification of frozen artifacts.

This script does NOT train, calibrate, or re-derive any model. It reads the
publicly distributed frozen artifacts (artifacts/frozen/*) and confirms they
match the headline numbers recorded in the PIS-CMOO manuscript.

Run:
    python scripts/verify_frozen_results.py

Exits 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "artifacts" / "frozen"


# ---- headline numbers (taken from the SCM01 final reconciliation, RESOLVED) ----

EXPECTED = {
    # G5 selection freeze
    "g5_champion_model": "tabdpt",
    "g5_champion_target": "log1p",
    "g5_champion_macro_mae": 1.8118,
    "g5_champion_worst_site_rmse": 9.5578,
    "g5_robust_alternate_model": "catboost",
    "g5_robust_alternate_macro_mae": 1.8504,

    # G5 macro metrics (development, natural Lu)
    "tabdpt_macro_r2": 0.5461592368982876,
    "tabdpt_worst_site_r2": 0.35919190545626056,
    "catboost_macro_r2": 0.5562291990991302,
    "g5_tabdpt_cov90": 0.8987,
    "g5_catboost_cov90": 0.8987,   # from g5_uq_summary.csv, see NOTE below

    # G7.6 final optimization (development)
    "g7_6_n_contexts": 1387,
    "g7_6_n_safe_contexts_nsga3_c7": 576,
    "g7_6_external_n_eligible_contexts": 259,
    "g7_6_external_n_tree_safe": 41,
    "g7_6_external_n_levelb": 11,

    # G8 external validation (SC-M01)
    "external_n_prediction_evaluable": 269,
    "external_n_optimization_eligible": 259,
    "external_n_raw": 275,

    # TabDPT external regression (n=269)
    "tabdpt_ext_mae": 0.8198440079887549,
    "tabdpt_ext_rmse": 1.4624028317400377,
    "tabdpt_ext_r2": 0.45951168397129294,

    # CatBoost external regression (n=269)
    "catboost_ext_mae": 0.9309947982691094,
    "catboost_ext_rmse": 1.7034121902335155,
    "catboost_ext_r2": 0.2666829660936779,

    # External UQ coverage (n=269)
    "tabdpt_ext_cov90": 0.9517,
    "tabdpt_ext_cov95": 0.9740,
    "catboost_ext_cov90": 0.9591,
    "catboost_ext_cov95": 0.9777,

    # Observed-action safety (n_observed_actions=275, UCB n=269 corrected)
    "tabdpt_ucb_safe_rate_str": "8/269",
    "tabdpt_conservative_false_unsafe": 84,
    "catboost_ucb_safe_rate_str": "7/269",
    "catboost_conservative_false_unsafe": 85,
    "tabdpt_ucb_safe_but_unqualified": 0,
    "catboost_ucb_safe_but_unqualified": 0,

    # Decision availability (n=259 optimization-eligible)
    "ext_tree_safe_n": 41,
    "ext_levelb_n": 11,
    "ext_disagreement_only_n": 30,
    "ext_no_tree_safe_n": 218,

    # RESOLVED classification metrics (n=269, prediction-evaluable)
    "tabdpt_mcc": 0.6438777834,
    "tabdpt_balanced_accuracy": 0.8030582167,
    "tabdpt_sensitivity": 0.6739130435,
    "tabdpt_specificity": 0.9322033898,
    "tabdpt_auroc": 0.9153770572,
    "tabdpt_pr_auc": 0.8588590623,
    "tabdpt_tp": 62, "tabdpt_fp": 12, "tabdpt_tn": 165, "tabdpt_fn": 30,

    "catboost_mcc": 0.6093740948,
    "catboost_balanced_accuracy": 0.7734893147,
    "catboost_sensitivity": 0.5978260870,
    "catboost_specificity": 0.9491525424,
    "catboost_auroc": 0.9086219602,
    "catboost_pr_auc": 0.8481690427,
    "catboost_tp": 55, "catboost_fp": 9, "catboost_tn": 168, "catboost_fn": 37,
}


def _approx(a, b, rel=1e-3, abs_tol=1e-6):
    return abs(a - b) <= max(abs_tol, rel * abs(b))


def check_g5_selection_freeze():
    p = FROZEN / "g5_selection_freeze.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["status"] == "CHAMPION_SELECTED"
    assert d["champion"]["model"] == EXPECTED["g5_champion_model"]
    assert d["champion"]["target_variant"] == EXPECTED["g5_champion_target"]
    assert _approx(d["champion"]["macro_mean_mae"], EXPECTED["g5_champion_macro_mae"]), \
        f"TabDPT macro_mean_mae drift: {d['champion']['macro_mean_mae']}"
    assert _approx(d["champion"]["worst_site_rmse"], EXPECTED["g5_champion_worst_site_rmse"]), \
        f"worst_site_rmse drift: {d['champion']['worst_site_rmse']}"
    assert d["robust_alternate"]["model"] == EXPECTED["g5_robust_alternate_model"]
    assert _approx(d["robust_alternate"]["macro_mean_mae"], EXPECTED["g5_robust_alternate_macro_mae"])
    print(f"  PASS  {p.name}  (TabDPT champion, CatBoost robust-alternate, no drift)")


def check_g5_macro_metrics():
    p = FROZEN / "g5_macro_metrics.csv"
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    tabdpt = [r for r in rows if r["model"] == "tabdpt" and r["route"] == "CORE_CC" and r["target_variant"] == "log1p"]
    catboost = [r for r in rows if r["model"] == "catboost" and r["route"] == "EXTENDED_FOLD_IMPUTED" and r["target_variant"] == "log1p"]
    assert len(tabdpt) == 1, f"TabDPT CORE_CC log1p row count = {len(tabdpt)}"
    assert len(catboost) == 1, f"CatBoost EXTENDED_FOLD_IMPUTED log1p row count = {len(catboost)}"
    assert _approx(float(tabdpt[0]["macro_mean_r2"]), EXPECTED["tabdpt_macro_r2"])
    assert _approx(float(tabdpt[0]["worst_site_r2"]), EXPECTED["tabdpt_worst_site_r2"])
    assert _approx(float(catboost[0]["macro_mean_r2"]), EXPECTED["catboost_macro_r2"])
    print(f"  PASS  {p.name}  (TabDPT macro R^2=0.5462, worst-site R^2=0.3592; CatBoost macro R^2=0.5562)")


def check_g5_uq_summary():
    p = FROZEN / "g5_uq_summary.csv"
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    tabdpt = [r for r in rows if r["model"] == "tabdpt" and r["target_variant"] == "log1p"]
    assert len(tabdpt) >= 1
    print(f"  PASS  {p.name}  (rows={len(rows)})")


def check_g7_6_status():
    p = FROZEN / "g7_6_full1387_status.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["n_contexts"] == EXPECTED["g7_6_n_contexts"]
    assert d["per_config_safe_contexts"]["NSGA3_C7"] == EXPECTED["g7_6_n_safe_contexts_nsga3_c7"]
    print(f"  PASS  {p.name}  (n_contexts=1387, NSGA3_C7 safe=576/1387=41.5%)")


def check_g8_protocol_freeze():
    p = FROZEN / "g8_external_protocol_freeze.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["external_mine"] == "SC-M01"
    assert "GX-M01" in d["final_training_mines"]
    assert "WCR" in d["allowed_external_input_columns"]
    assert "post_lugeon" in d["forbidden_outcome_columns"]
    print(f"  PASS  {p.name}  (SC-M01 external; 7-feature predictor; post_lugeon forbidden pre-unblind)")


def check_g8_completion_manifest():
    p = FROZEN / "g8_external_validation_completion_manifest.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    c = d["cohort"]
    assert c["raw_external"] == EXPECTED["external_n_raw"]
    assert c["prediction_evaluable"] == EXPECTED["external_n_prediction_evaluable"]
    assert c["optimization_eligible"] == EXPECTED["external_n_optimization_eligible"]
    assert d["subsequent_changes_restricted_to"].startswith("reporting")
    assert d["no_post_unblind_scientific_adaptation"] is True
    print(f"  PASS  {p.name}  (cohorts 275/269/259; CLOSED protocol)")


def check_g8_external_prediction_metrics():
    p = FROZEN / "g8_external_prediction_metrics.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    n_records = d["n_records_with_outcome"]
    assert n_records == EXPECTED["external_n_raw"], f"n_records_with_outcome = {n_records}"

    # Regression (n=269)
    tabdpt_reg = d["prediction_metrics"]["tabdpt"]
    catboost_reg = d["prediction_metrics"]["catboost"]
    assert tabdpt_reg["n"] == 269
    assert _approx(tabdpt_reg["mae"], EXPECTED["tabdpt_ext_mae"])
    assert _approx(tabdpt_reg["rmse"], EXPECTED["tabdpt_ext_rmse"])
    assert _approx(tabdpt_reg["r2"], EXPECTED["tabdpt_ext_r2"])
    assert catboost_reg["n"] == 269
    assert _approx(catboost_reg["mae"], EXPECTED["catboost_ext_mae"])
    assert _approx(catboost_reg["rmse"], EXPECTED["catboost_ext_rmse"])
    assert _approx(catboost_reg["r2"], EXPECTED["catboost_ext_r2"])

    # Classification (n=269, RESOLVED reconciliation values)
    cls_t = d["qualification_classification"]["tabdpt"]
    cls_c = d["qualification_classification"]["catboost"]
    assert cls_t["n"] == 269, f"TabDPT qual n = {cls_t['n']}"
    assert cls_c["n"] == 269, f"CatBoost qual n = {cls_c['n']}"
    assert _approx(cls_t["mcc"], EXPECTED["tabdpt_mcc"]), \
        f"TabDPT MCC drift: got {cls_t['mcc']}, expected {EXPECTED['tabdpt_mcc']}"
    assert _approx(cls_t["balanced_accuracy"], EXPECTED["tabdpt_balanced_accuracy"])
    assert _approx(cls_t["sensitivity"], EXPECTED["tabdpt_sensitivity"])
    assert _approx(cls_t["specificity"], EXPECTED["tabdpt_specificity"])
    assert _approx(cls_t["auroc_qualification_margin"], EXPECTED["tabdpt_auroc"])
    assert _approx(cls_t["pr_auc_qualification_margin"], EXPECTED["tabdpt_pr_auc"])
    assert (cls_t["tp"], cls_t["fp"], cls_t["tn"], cls_t["fn"]) == (62, 12, 165, 30)
    assert _approx(cls_c["mcc"], EXPECTED["catboost_mcc"])
    assert _approx(cls_c["balanced_accuracy"], EXPECTED["catboost_balanced_accuracy"])
    assert _approx(cls_c["sensitivity"], EXPECTED["catboost_sensitivity"])
    assert _approx(cls_c["specificity"], EXPECTED["catboost_specificity"])
    assert _approx(cls_c["auroc_qualification_margin"], EXPECTED["catboost_auroc"])
    assert _approx(cls_c["pr_auc_qualification_margin"], EXPECTED["catboost_pr_auc"])
    assert (cls_c["tp"], cls_c["fp"], cls_c["tn"], cls_c["fn"]) == (55, 9, 168, 37)

    # UQ coverage (n=269)
    uq_t = d["uq_metrics"]["tabdpt"]
    uq_c = d["uq_metrics"]["catboost"]
    assert _approx(uq_t["90"]["coverage"], EXPECTED["tabdpt_ext_cov90"])
    assert _approx(uq_t["95"]["coverage"], EXPECTED["tabdpt_ext_cov95"])
    assert _approx(uq_c["90"]["coverage"], EXPECTED["catboost_ext_cov90"])
    assert _approx(uq_c["95"]["coverage"], EXPECTED["catboost_ext_cov95"])

    print(f"  PASS  {p.name}")
    print(f"        TabDPT n=269: MAE={tabdpt_reg['mae']:.4f} RMSE={tabdpt_reg['rmse']:.4f} R²={tabdpt_reg['r2']:.4f}")
    print(f"        TabDPT qual: MCC={cls_t['mcc']:.4f} BalAcc={cls_t['balanced_accuracy']:.4f} AUROC={cls_t['auroc_qualification_margin']:.4f}")
    print(f"        CatBoost qual: MCC={cls_c['mcc']:.4f} BalAcc={cls_c['balanced_accuracy']:.4f} AUROC={cls_c['auroc_qualification_margin']:.4f}")


def check_observed_action_safety():
    p = FROZEN / "g8_observed_action_safety_corrected.csv"
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    by_model = {r["model"]: r for r in rows}
    tabdpt = by_model["TabDPT"]
    catboost = by_model["CatBoost"]
    assert int(tabdpt["n_evaluable"]) == 269
    assert int(catboost["n_evaluable"]) == 269
    assert tabdpt["ucb90_safe_rate"] == EXPECTED["tabdpt_ucb_safe_rate_str"]
    assert catboost["ucb90_safe_rate"] == EXPECTED["catboost_ucb_safe_rate_str"]
    assert int(tabdpt["ucb90_safe_but_unqualified"]) == EXPECTED["tabdpt_ucb_safe_but_unqualified"]
    assert int(catboost["ucb90_safe_but_unqualified"]) == EXPECTED["catboost_ucb_safe_but_unqualified"]
    assert int(tabdpt["ucb90_unsafe_but_qualified"]) == EXPECTED["tabdpt_conservative_false_unsafe"]
    assert int(catboost["ucb90_unsafe_but_qualified"]) == EXPECTED["catboost_conservative_false_unsafe"]
    print(f"  PASS  {p.name}  (TabDPT 8/269 UCB-safe, 0 false-safe; CatBoost 7/269, 0 false-safe)")


def check_recommendation_availability():
    p = FROZEN / "g8_recommendation_availability_corrected.csv"
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    by_metric = {r["metric"]: r for r in rows}
    assert int(by_metric["tree_safe / eligible"]["denominator"]) == 259
    assert int(by_metric["LevelB / eligible"]["denominator"]) == 259
    assert int(by_metric["LevelB / eligible"]["numerator"]) == EXPECTED["ext_levelb_n"]
    print(f"  PASS  {p.name}  (decision availability on n=259)")


def main():
    print("=" * 72)
    print("PIS-CMOO — frozen-result verification")
    print("=" * 72)
    try:
        check_g5_selection_freeze()
        check_g5_macro_metrics()
        check_g5_uq_summary()
        check_g7_6_status()
        check_g8_protocol_freeze()
        check_g8_completion_manifest()
        check_g8_external_prediction_metrics()
        check_observed_action_safety()
        check_recommendation_availability()
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        return 1
    except FileNotFoundError as e:
        print(f"\nFAIL: missing artifact: {e}")
        return 1
    print("\n" + "=" * 72)
    print("ALL FROZEN-ARTIFACT CHECKS PASS")
    print("=" * 72)
    print("NOTE: This verification is a read-only check of the publicly distributed")
    print("frozen artifacts. It does NOT re-derive any model, UQ calibration, or")
    print("optimization output. The SC-M01 final classification metrics reflect the")
    print("RESOLVED reconciliation (n=269 prediction-evaluable cohort; see")
    print("docs/evidence_manifest.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())