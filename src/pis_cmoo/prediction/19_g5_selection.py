"""
scripts/19_g5_selection.py — apply the PRE-FROZEN selection protocol to choose champion
and robust alternate. No SC-M01. No post-hoc composite score.

Selection (per g5_selection_protocol.yaml):
  eligibility gate -> UQ reliability floor -> minimize raw-Lu macro MAE -> tie-breaks.
Writes: 04_prediction/g5_selection_freeze.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import _common as C

FAMILY = {
    "ridge": "linear", "elastic_net": "linear",
    "random_forest": "tree_ensemble", "extra_trees": "tree_ensemble",
    "hist_gradient_boosting": "gradient_boosting", "xgboost": "gradient_boosting",
    "lightgbm": "gradient_boosting", "catboost": "gradient_boosting",
    "tabpfn_v2": "tfm", "tabicl": "tfm", "tabdpt": "tfm",
}


def main() -> int:
    C.ensure_dirs()
    macro = pd.read_csv(C.PRED_DIR / "g5_macro_metrics.csv")
    uq = pd.read_csv(C.PRED_DIR / "g5_uq_summary.csv")
    import json
    failures = json.loads((C.PRED_DIR / "g5_failures.json").read_text(encoding="utf-8"))

    # merge UQ into macro
    m = macro.merge(uq, on=["route", "model", "target_variant"], how="left")
    # eligibility: 6 folds, no failures
    failed_keys = {(f["route"], f["model"], f["target_variant"]) for f in failures["failures"]}
    m["eligible"] = (m["n_folds"] == 6) & (~m.apply(lambda r: (r["route"], r["model"], r["target_variant"]) in failed_keys, axis=1))
    m["family"] = m["model"].map(FAMILY)

    # UQ reliability floor
    m["uq_floor_ok"] = (
        (m["coverage_90_macro"] >= 0.85) &
        (m["coverage_95_macro"] >= 0.90) &
        (m["coverage_95_worst_site"] >= 0.80)
    )
    # coverage deficit (95% macro vs 0.95)
    m["coverage_95_deficit"] = (0.95 - m["coverage_95_macro"]).abs()

    qualifying = m[m["eligible"] & m["uq_floor_ok"]].reset_index(drop=True)

    all_eligible = m[m["eligible"]].sort_values("macro_mean_mae").reset_index(drop=True)

    result = {
        "generated_at_utc": C.now_utc(),
        "n_configs": int(len(m)),
        "n_eligible": int(m["eligible"].sum()),
        "n_uq_floor_pass": int((m["eligible"] & m["uq_floor_ok"]).sum()),
    }

    if len(qualifying) == 0:
        result["status"] = "CALIBRATION_FAILURE"
        result["champion"] = None
        result["alternate"] = None
        result["note"] = ("No configuration satisfies the UQ reliability floor under "
                          "limited-mine conditions; no champion declared. Request GPT decision.")
        C.write_json(C.PRED_DIR / "g5_selection_freeze.json", result)
        print("STOP_CALIBRATION_FAILURE: no config passes UQ floor.")
        if len(all_eligible):
            r = all_eligible.iloc[0]
            print(f"  best-eligible (reference only): {r['model']} {r['target_variant']} macro MAE={r['macro_mean_mae']:.4f}")
        return 2

    # Frozen selection: minimize macro MAE; among configs within 2% of the min, pick
    # lower worst-site RMSE; then narrower 95% interval + smaller coverage deficit.
    best_mae = float(qualifying["macro_mean_mae"].min())
    contenders = qualifying[qualifying["macro_mean_mae"] <= best_mae * 1.02].copy()
    contenders = contenders.sort_values(
        ["worst_site_rmse", "width_95_mean", "coverage_95_deficit"]).reset_index(drop=True)
    champ = contenders.iloc[0]

    # robust alternate: best qualifying config from a DIFFERENT family (by macro MAE)
    alt = qualifying[qualifying["family"] != champ["family"]].sort_values("macro_mean_mae")
    if len(alt) == 0:
        alt = qualifying[qualifying["model"] != champ["model"]]
    if len(alt) == 0:
        alt = qualifying.iloc[1:2]
    alternate = alt.iloc[0]

    result["status"] = "CHAMPION_SELECTED"
    result["champion"] = {
        "route": champ["route"], "model": champ["model"], "target_variant": champ["target_variant"],
        "family": champ["family"], "macro_mean_mae": round(float(champ["macro_mean_mae"]), 4),
        "macro_mean_rmse": round(float(champ["macro_mean_rmse"]), 4),
        "worst_site_rmse": round(float(champ["worst_site_rmse"]), 4),
        "coverage_90_macro": round(float(champ["coverage_90_macro"]), 4),
        "coverage_95_macro": round(float(champ["coverage_95_macro"]), 4),
        "coverage_95_worst_site": round(float(champ["coverage_95_worst_site"]), 4),
    }
    result["robust_alternate"] = {
        "route": alternate["route"], "model": alternate["model"], "target_variant": alternate["target_variant"],
        "family": alternate["family"], "macro_mean_mae": round(float(alternate["macro_mean_mae"]), 4),
        "macro_mean_rmse": round(float(alternate["macro_mean_rmse"]), 4),
        "worst_site_rmse": round(float(alternate["worst_site_rmse"]), 4),
    }
    result["selection_protocol"] = {
        "rule1": "minimize raw-Lu macro MAE among eligible+UQ-floor-passing",
        "tiebreak": "lower worst-site RMSE -> narrower 95% interval -> smaller coverage deficit -> simpler",
        "excluded_sole_deciders": ["R2", "pooled", "pilot"],
    }
    C.write_json(C.PRED_DIR / "g5_selection_freeze.json", result)

    print(f"Configs: {result['n_configs']}; eligible {result['n_eligible']}; UQ-floor-pass {result['n_uq_floor_pass']}")
    print(f"Champion: {champ['model']} / {champ['target_variant']} / {champ['route']}  macro MAE={champ['macro_mean_mae']:.4f}")
    print(f"Alternate: {alternate['model']} / {alternate['target_variant']} / {alternate['route']}  macro MAE={alternate['macro_mean_mae']:.4f}")
    print("\nTop qualifying (by macro MAE):")
    for _, r in qualifying.head(6).iterrows():
        print(f"  {r['model']:24s} {r['target_variant']:5s} {r['route']:22s} "
              f"MAE={r['macro_mean_mae']:.4f} worstRMSE={r['worst_site_rmse']:.4f} "
              f"cov95={r['coverage_95_macro']:.3f} worst95={r['coverage_95_worst_site']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
