"""
scripts/17_g5_uq.py — grouped cross-fitted residual conformal UQ for top candidate configs.

For each top-8 config (by macro MAE), refit (with the frozen best hyperparameters) inside
each outer-training set via GroupKFold(borehole_id, 3) to obtain grouped out-of-fold
calibration residuals in the MODELING space, then build 90%/95% two-sided + one-sided
upper intervals. Test-mine outcome is NEVER used for calibration.

Writes:
  - 04_prediction/g5_uq_summary.csv
  - 04_prediction/g5_uq_intervals.csv   (prediction-level for top configs)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import _common as C

CORE_FEATURES = C.DECISION_COLS + ["depth_m", "section_length_m", "pre_lugeon"]
EXT_NUMERIC = ["groundwater_pressure_MPa"]
EXT_CATEGORICAL = ["lithology", "karst_class", "fracture_class", "hydrogeological_regime"]
TARGET = "post_lugeon"
TOP_N = 8


def make_classic(name):
    if name == "ridge":
        return Ridge()
    if name == "elastic_net":
        return ElasticNet(max_iter=5000)
    if name == "random_forest":
        return RandomForestRegressor(random_state=C.SEED_MASTER, n_jobs=1)
    if name == "extra_trees":
        return ExtraTreesRegressor(random_state=C.SEED_MASTER, n_jobs=1)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(random_state=C.SEED_MASTER)
    if name == "xgboost":
        return XGBRegressor(random_state=C.SEED_MASTER, n_jobs=1, verbosity=0)
    if name == "lightgbm":
        return LGBMRegressor(random_state=C.SEED_MASTER, n_jobs=1, verbose=-1)
    if name == "catboost":
        return CatBoostRegressor(random_seed=C.SEED_MASTER, thread_count=1, verbose=0)
    raise ValueError(name)


def make_tfm(name):
    if name == "tabpfn_v2":
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor(device="cpu")
    if name == "tabicl":
        from tabicl import TabICLRegressor
        return TabICLRegressor(device="cpu")
    if name == "tabdpt":
        from tabdpt import TabDPTRegressor
        return TabDPTRegressor(device="cpu")
    raise ValueError(name)


def build_pipeline(route, model_name, best_params):
    is_linear = model_name in ("ridge", "elastic_net")
    if model_name in ("tabpfn_v2", "tabicl", "tabdpt"):
        return make_tfm(model_name)
    model = make_classic(model_name)
    if route == "CORE_CC":
        steps = [("scale", StandardScaler())] if is_linear else []
        pre = ColumnTransformer([("num", Pipeline(steps) if steps else "passthrough", CORE_FEATURES)])
    else:
        num_steps = [("imp", SimpleImputer(strategy="median")),
                     ("mi", SimpleImputer(strategy="constant", fill_value=0, add_indicator=True))]
        if is_linear:
            num_steps.append(("scale", StandardScaler()))
        cat_steps = [("imp", SimpleImputer(strategy="most_frequent")),
                     ("mi", SimpleImputer(strategy="constant", fill_value="__missing__", add_indicator=True)),
                     ("ohe", OneHotEncoder(handle_unknown="ignore"))]
        pre = ColumnTransformer([("num", Pipeline(num_steps), CORE_FEATURES + EXT_NUMERIC),
                                 ("cat", Pipeline(cat_steps), EXT_CATEGORICAL)])
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.set_params(**best_params)
    return pipe


def main() -> int:
    C.ensure_dirs()
    if not (C.PRED_DIR / "g5_macro_metrics.csv").exists():
        print("STOP_NO_MACRO — run 16_g5_analyze.py first", file=sys.stderr)
        return 2

    macro = pd.read_csv(C.PRED_DIR / "g5_macro_metrics.csv")
    best_params_df = pd.read_csv(C.PRED_DIR / "g5_best_params.csv")
    top = macro.sort_values("macro_mean_mae").head(TOP_N)
    candidates = list(top[["route", "model", "target_variant"]].itertuples(index=False, name=None))

    df = C.read_frozen()
    dev = df[df["mine_id"].isin(C.DEVELOPMENT_MINES)].copy()
    dev = dev[dev[C.CORE_REQUIRED].notna().all(axis=1)].reset_index(drop=True)

    summary_rows = []
    interval_rows = []

    for route, model_name, target in candidates:
        feat_cols = CORE_FEATURES if route == "CORE_CC" else CORE_FEATURES + EXT_NUMERIC + EXT_CATEGORICAL
        is_tfm = model_name in ("tabpfn_v2", "tabicl", "tabdpt")
        # get best params for this config (per fold)
        bp_sub = best_params_df[(best_params_df["route"] == route) & (best_params_df["model"] == model_name)
                                & (best_params_df["target_variant"] == target)]

        cov = {a: [] for a in [0.10, 0.05]}
        widths = {a: [] for a in [0.10, 0.05]}
        per_mine = []
        for fold_mine in C.DEVELOPMENT_MINES:
            test = dev[dev["mine_id"] == fold_mine]
            train = dev[dev["mine_id"] != fold_mine].reset_index(drop=True)
            X_test = test[feat_cols]
            y_test_raw = test[TARGET].astype(float).values
            X_train = train[feat_cols]
            y_train_raw = train[TARGET].astype(float).values
            y_train = y_train_raw if target == "raw" else np.log1p(y_train_raw)
            groups = train["borehole_id"].values

            # cross-fitted calibration residuals in modeling space
            gkf = GroupKFold(n_splits=3)
            calib_resid = []
            for tr, va in gkf.split(X_train, y_train, groups):
                if is_tfm:
                    m = make_tfm(model_name)
                    m.fit(X_train.iloc[tr].astype("float32").values, y_train[tr].astype("float32"))
                    pred = m.predict(X_train.iloc[va].astype("float32").values).astype(float)
                else:
                    params = json.loads(bp_sub.loc[bp_sub["outer_fold"] == fold_mine, "best_params"].iloc[0]) \
                        if len(bp_sub) else {}
                    m = build_pipeline(route, model_name, params)
                    m.fit(X_train.iloc[tr], y_train[tr])
                    pred = m.predict(X_train.iloc[va]).astype(float)
                calib_resid.append(y_train[va] - pred)
            calib_resid = np.concatenate(calib_resid)

            # test prediction (recompute for interval alignment)
            if is_tfm:
                m = make_tfm(model_name)
                m.fit(X_train.astype("float32").values, y_train.astype("float32"))
                y_pred_model = m.predict(X_test.astype("float32").values).astype(float)
            else:
                params = json.loads(bp_sub.loc[bp_sub["outer_fold"] == fold_mine, "best_params"].iloc[0]) if len(bp_sub) else {}
                m = build_pipeline(route, model_name, params)
                m.fit(X_train, y_train)
                y_pred_model = m.predict(X_test).astype(float)

            for alpha in [0.10, 0.05]:
                q2 = float(np.quantile(np.abs(calib_resid), 1 - alpha))
                q1 = float(np.quantile(calib_resid, 1 - alpha))
                lo_model = y_pred_model - q2
                hi_model = y_pred_model + q2
                upper_model = y_pred_model + q1
                if target == "raw":
                    lo_lu = np.clip(lo_model, 0, None)
                    hi_lu = hi_model
                    upper_lu = upper_model
                else:
                    lo_lu = np.clip(np.expm1(lo_model), 0, None)
                    hi_lu = np.expm1(hi_model)
                    upper_lu = np.expm1(upper_model)
                covered = (y_test_raw >= lo_lu) & (y_test_raw <= hi_lu)
                cov[alpha].append(float(covered.mean()))
                widths[alpha].append(float(np.mean(hi_lu - lo_lu)))
                if alpha == 0.05:
                    for i in range(len(test)):
                        interval_rows.append({
                            "route": route, "model": model_name, "target_variant": target,
                            "outer_fold": fold_mine, "record_id": test["record_id"].iloc[i],
                            "y_true": float(y_test_raw[i]),
                            "lo_95": float(lo_lu[i]), "hi_95": float(hi_lu[i]),
                            "upper_95": float(upper_lu[i]),
                            "covered_95": int(covered[i]),
                        })
            per_mine.append({"outer_fold": fold_mine, "coverage_90": cov[0.10][-1],
                             "coverage_95": cov[0.05][-1]})

        # worst-site coverage = min over folds
        worst90 = min(pm["coverage_90"] for pm in per_mine)
        worst95 = min(pm["coverage_95"] for pm in per_mine)
        summary_rows.append({
            "route": route, "model": model_name, "target_variant": target,
            "coverage_90_macro": round(float(np.mean(cov[0.10])), 4),
            "coverage_95_macro": round(float(np.mean(cov[0.05])), 4),
            "coverage_95_worst_site": round(float(worst95), 4),
            "coverage_90_worst_site": round(float(worst90), 4),
            "width_90_mean": round(float(np.mean(widths[0.10])), 4),
            "width_95_mean": round(float(np.mean(widths[0.05])), 4),
            "width_95_median": round(float(np.median(widths[0.05])), 4),
        })
        print(f"  {route:22s} {model_name:24s} {target:5s} "
              f"cov90={summary_rows[-1]['coverage_90_macro']:.3f} cov95={summary_rows[-1]['coverage_95_macro']:.3f} "
              f"worst95={summary_rows[-1]['coverage_95_worst_site']:.3f}")

    pd.DataFrame(summary_rows).to_csv(C.PRED_DIR / "g5_uq_summary.csv", index=False, lineterminator="\n")
    pd.DataFrame(interval_rows).to_csv(C.PRED_DIR / "g5_uq_intervals.csv", index=False, lineterminator="\n")
    print(f"\nUQ done: {len(summary_rows)} candidate configs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
