"""scripts/211_g8_external_evaluation.py — G8 §13 external prediction/UQ evaluation +
§14 observed-action safety audit.

Post-unblinding. Evaluates the frozen TabDPT (primary) and CatBoost (secondary) deployment
predictions at historical observed decisions against the one-time unblinded SC-M01 outcome.
No scientific protocol modification.

Outputs (08_validation/g8/):
  - g8_external_prediction_metrics.json
  - g8_external_uq_metrics.json
  - g8_observed_action_safety.csv
"""
from __future__ import annotations
import sys, json, datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, matthews_corrcoef,
                             balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C

V8 = C.VALIDATION_DIR / "g8"
OUT = V8


def _reg_metrics(actual, pred):
    m = {}
    m["n"] = int(len(actual))
    m["mae"] = float(mean_absolute_error(actual, pred))
    m["rmse"] = float(np.sqrt(mean_squared_error(actual, pred)))
    m["r2"] = float(r2_score(actual, pred))
    m["median_ae"] = float(np.median(np.abs(actual - pred)))
    return m


def _class_metrics(actual_qualified, pred_safe):
    # pred_safe = predicted qualified (point pred <= threshold)
    n = len(actual_qualified)
    tp = int((pred_safe & actual_qualified).sum())
    fp = int((pred_safe & ~actual_qualified).sum())
    tn = int((~pred_safe & ~actual_qualified).sum())
    fn = int((~pred_safe & actual_qualified).sum())
    mcc = matthews_corrcoef(actual_qualified, pred_safe) if len(set(actual_qualified)) > 1 else float("nan")
    bacc = balanced_accuracy_score(actual_qualified, pred_safe)
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    return {"n": n, "mcc": float(mcc), "balanced_accuracy": float(bacc),
            "sensitivity": float(sens), "specificity": float(spec),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def main():
    pred = pd.read_csv(V8 / "preoutcome" / "g8_external_historical_action_predictions.csv")
    out = pd.read_parquet(V8 / "outcome" / "g8_sc_m01_outcome.parquet")
    m = pred.merge(out[["record_id", "post_lugeon", "qualified"]], on="record_id", how="left")
    m = m[m["post_lugeon"].notna()].reset_index(drop=True)
    actual = m["post_lugeon"].astype(float).values
    qualified = m["qualified"].astype(bool).values
    thr = m["acceptance_threshold_lu"].astype(float).values

    pred_metrics = {}
    class_metrics = {}
    uq_metrics = {}
    for mdl, tag in [("tabdpt", "TabDPT (primary)"), ("catboost", "CatBoost (secondary)")]:
        p = m[f"pred_{mdl}_post"].astype(float).values
        ok = ~np.isnan(p)
        pred_metrics[mdl] = _reg_metrics(actual[ok], p[ok])

        # qualification ranking: margin = thr - pred_post (higher => predicted qualified)
        margin = (thr - p)
        mok = ok & np.isfinite(margin)
        if len(set(qualified[mok])) > 1:
            auroc = float(roc_auc_score(qualified[mok], margin[mok]))
            pr_auc = float(average_precision_score(qualified[mok], margin[mok]))
        else:
            auroc = pr_auc = float("nan")
        point_safe = p <= thr
        cm = _class_metrics(qualified, point_safe)
        cm["auroc_qualification_margin"] = auroc
        cm["pr_auc_qualification_margin"] = pr_auc
        class_metrics[mdl] = cm

        uq = {}
        for a, qkey in [("90", "q90"), ("95", "q95")]:
            q = json.loads((V8 / f"g8_{mdl}_conformal_freeze.json").read_text(encoding="utf-8"))[qkey]
            pl = m[f"pred_{mdl}_log"].astype(float).values
            ucb = np.expm1(pl + q)
            cov = float(np.mean(ucb[ok] >= actual[ok]))
            width = ucb[ok] - p[ok]
            under = int((ucb[ok] < actual[ok]).sum())
            uq[a] = {
                "coverage": round(cov, 4),
                "mean_width": round(float(np.mean(width)), 4),
                "median_width": round(float(np.median(width)), 4),
                "undercoverage_count": under,
                "upper_bound_violation_count": under,
            }
        uq_metrics[mdl] = uq

    # §14 observed-action safety audit
    safety_rows = []
    for mdl, tag in [("tabdpt", "TabDPT"), ("catboost", "CatBoost")]:
        p = m[f"pred_{mdl}_post"].astype(float).values
        pl = m[f"pred_{mdl}_log"].astype(float).values
        point_safe = p <= thr
        ucb90_safe = np.expm1(pl + json.loads((V8 / f"g8_{mdl}_conformal_freeze.json")
                              .read_text(encoding="utf-8"))["q90"]) <= thr
        n = len(actual)
        false_safe_point = int((point_safe & ~qualified).sum())
        false_safe_ucb = int((ucb90_safe & ~qualified).sum())
        false_unsafe_ucb = int((~ucb90_safe & qualified).sum())
        safety_rows.append({
            "model": mdl, "n_observed_actions": n,
            "point_safe_but_unqualified": false_safe_point,
            "ucb90_safe_but_unqualified": false_safe_ucb,
            "ucb90_unsafe_but_qualified_conservative": false_unsafe_ucb,
            "ucb90_safe_rate": round(float(ucb90_safe.mean()), 4),
        })
    safety_df = pd.DataFrame(safety_rows)
    safety_df.to_csv(OUT / "g8_observed_action_safety.csv", index=False, encoding="utf-8",
                     lineterminator="\n")

    result = {
        "stage": "G8_EXTERNAL_EVALUATION",
        "n_records_with_outcome": int(len(m)),
        "prediction_metrics": pred_metrics,
        "qualification_classification": class_metrics,
        "uq_metrics": uq_metrics,
        "observed_action_safety": safety_rows,
        "no_protocol_modification": True,
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (OUT / "g8_external_prediction_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
