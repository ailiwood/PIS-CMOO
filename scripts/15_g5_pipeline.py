"""
scripts/15_g5_pipeline.py — formal 6-fold nested Leave-One-Mine-Out on 6 development mines.

Models: 8 classic + TabPFN v2 + TabICL + TabDPT. Targets: raw + log1p.
Routes: CORE_CC (11 models) + EXTENDED_FOLD_IMPUTED (8 classic; TFM deferred, documented).

Inner tuning: GroupKFold(borehole_id, 3) with bounded search (grid for linear, random 20
for trees/boosting) per g5_search_spaces.yaml. SC-M01 is never read for training/tuning.

Outputs:
  - 04_prediction/g5_oof.csv            (prediction-level, unique per record+route+model+target+seed)
  - 04_prediction/g5_metrics.csv        (per fold × config)
  - 04_prediction/g5_runtime.csv
  - 04_prediction/g5_best_params.csv
  - 04_prediction/g5_failures.json
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import _common as C

warnings.filterwarnings("ignore")

CORE_FEATURES = ["WCR", "GP_MPa", "grout_take_L_per_m", "grouting_time_h",
                 "depth_m", "section_length_m", "pre_lugeon"]
EXT_NUMERIC = ["groundwater_pressure_MPa"]
EXT_CATEGORICAL = ["lithology", "karst_class", "fracture_class", "hydrogeological_regime"]
TARGET = "post_lugeon"
SEED = C.SEED_MASTER

OOF_COLS = ["record_id", "mine_id", "project_id", "borehole_id", "outer_fold",
            "route", "model", "target_variant", "seed", "y_true", "y_pred_clipped",
            "y_pred_unclipped", "residual", "negative_pred", "acceptance_threshold_lu",
            "qualified_true", "qualified_pred_regression"]


def load_search_spaces() -> dict:
    with (C.CONFIG_DIR / "g5_search_spaces.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_classic(name: str):
    if name == "ridge":
        return Ridge()
    if name == "elastic_net":
        return ElasticNet(max_iter=5000)
    if name == "random_forest":
        return RandomForestRegressor(random_state=SEED, n_jobs=1)
    if name == "extra_trees":
        return ExtraTreesRegressor(random_state=SEED, n_jobs=1)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(random_state=SEED)
    if name == "xgboost":
        return XGBRegressor(random_state=SEED, n_jobs=1, verbosity=0)
    if name == "lightgbm":
        return LGBMRegressor(random_state=SEED, n_jobs=1, verbose=-1)
    if name == "catboost":
        return CatBoostRegressor(random_seed=SEED, thread_count=1, verbose=0)
    raise ValueError(name)


def make_tfm(name: str):
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


def is_linear(name: str) -> bool:
    return name in ("ridge", "elastic_net")


def build_preprocessor(route: str, scale: bool):
    if route == "CORE_CC":
        steps = []
        if scale:
            steps.append(("scale", StandardScaler()))
        return ColumnTransformer([("num", Pipeline(steps) if steps else "passthrough", CORE_FEATURES)])
    num_cols = CORE_FEATURES + EXT_NUMERIC
    num_steps = [("imp", SimpleImputer(strategy="median")),
                 ("mi", SimpleImputer(strategy="constant", fill_value=0, add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [("imp", SimpleImputer(strategy="most_frequent")),
                 ("mi", SimpleImputer(strategy="constant", fill_value="__missing__", add_indicator=True)),
                 ("ohe", OneHotEncoder(handle_unknown="ignore"))]
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("cat", Pipeline(cat_steps), EXT_CATEGORICAL),
    ])


def tune_classic(pipe, name, params, groups, X, y):
    """Return refit pipeline with best hyperparameters (bounded search)."""
    gkf = GroupKFold(n_splits=3)
    if is_linear(name):
        searcher = GridSearchCV(pipe, params, cv=gkf, scoring="neg_mean_absolute_error",
                                n_jobs=-1, error_score="raise")
    else:
        searcher = RandomizedSearchCV(pipe, params, n_iter=20, cv=gkf,
                                      scoring="neg_mean_absolute_error", n_jobs=-1,
                                      random_state=SEED, error_score="raise")
    searcher.fit(X, y, groups=groups)
    return searcher.best_estimator_, searcher.best_params_


def run_config(route, model_name, target, dev, folds, search_spaces):
    """Run one (route, model, target) config across all 6 folds. Returns (oof_rows, metric_rows, runtime_rows, best_params_rows)."""
    feat_cols = CORE_FEATURES if route == "CORE_CC" else CORE_FEATURES + EXT_NUMERIC + EXT_CATEGORICAL
    is_tfm = model_name in ("tabpfn_v2", "tabicl", "tabdpt")

    oof_rows, metric_rows, runtime_rows, best_rows = [], [], [], []
    for fold_mine in folds:
        test = dev[dev["mine_id"] == fold_mine]
        train = dev[dev["mine_id"] != fold_mine].reset_index(drop=True)
        if len(test) == 0 or len(train) == 0:
            continue
        X_train = train[feat_cols]
        X_test = test[feat_cols]
        y_train_raw = train[TARGET].astype(float).values
        y_test_raw = test[TARGET].astype(float).values
        y_train = y_train_raw if target == "raw" else np.log1p(y_train_raw)

        groups = train["borehole_id"].values
        t0 = time.time()
        try:
            if is_tfm:
                model = make_tfm(model_name)
                model.fit(X_train.astype("float32").values, y_train.astype("float32"))
                best_params = {}
                y_pred_model = model.predict(X_test.astype("float32").values).astype(float)
            else:
                model = make_classic(model_name)
                pipe = Pipeline([("pre", build_preprocessor(route, is_linear(model_name))),
                                 ("model", model)])
                params = search_spaces[model_name]["params"]
                pipe, best_params = tune_classic(pipe, model_name, params, groups, X_train, y_train)
                y_pred_model = pipe.predict(X_test).astype(float)
            ok = True
            err = ""
        except Exception as exc:
            ok = False
            err = f"{type(exc).__name__}: {str(exc)[:200]}"
            y_pred_model = np.full(len(test), np.nan)
            best_params = {}
        elapsed = time.time() - t0

        if not ok:
            runtime_rows.append({"route": route, "model": model_name, "target_variant": target,
                                 "outer_fold": fold_mine, "seconds": round(elapsed, 2),
                                 "status": "ERROR", "error": err})
            continue

        # back-transform to Lu, clip negatives (primary)
        y_pred_unclipped = y_pred_model if target == "raw" else np.expm1(y_pred_model)
        y_pred_clipped = np.clip(y_pred_unclipped, 0, None)
        neg = y_pred_unclipped < 0
        resid = y_test_raw - y_pred_clipped
        thr = test["acceptance_threshold_lu"].astype(float).values

        m = {
            "mae": float(mean_absolute_error(y_test_raw, y_pred_clipped)),
            "rmse": float(np.sqrt(mean_squared_error(y_test_raw, y_pred_clipped))),
            "r2": float(r2_score(y_test_raw, y_pred_clipped)),
            "median_ae": float(np.median(np.abs(y_test_raw - y_pred_clipped))),
        }
        metric_rows.append({"route": route, "model": model_name, "target_variant": target,
                            "outer_fold": fold_mine, "test_n": len(test), **m})
        runtime_rows.append({"route": route, "model": model_name, "target_variant": target,
                             "outer_fold": fold_mine, "seconds": round(elapsed, 2), "status": "OK",
                             "negative_pred_rate": round(float(neg.mean()), 6)})
        best_rows.append({"route": route, "model": model_name, "target_variant": target,
                          "outer_fold": fold_mine, "best_params": json.dumps(best_params, default=str)})

        for i in range(len(test)):
            oof_rows.append({
                "record_id": test["record_id"].iloc[i],
                "mine_id": test["mine_id"].iloc[i],
                "project_id": test["project_id"].iloc[i],
                "borehole_id": test["borehole_id"].iloc[i],
                "outer_fold": fold_mine,
                "route": route, "model": model_name, "target_variant": target, "seed": SEED,
                "y_true": float(y_test_raw[i]),
                "y_pred_clipped": float(y_pred_clipped[i]),
                "y_pred_unclipped": float(y_pred_unclipped[i]),
                "residual": float(resid[i]),
                "negative_pred": int(neg[i]),
                "acceptance_threshold_lu": float(thr[i]),
                "qualified_true": int(test["qualified"].iloc[i]),
                "qualified_pred_regression": int(y_pred_clipped[i] <= thr[i]),
            })
    return oof_rows, metric_rows, runtime_rows, best_rows


def main() -> int:
    C.ensure_dirs()
    C.PRED_DIR.mkdir(parents=True, exist_ok=True)
    frz_sha = C.sha256(C.FROZEN_CSV)
    if frz_sha != C.EXPECTED_FROZEN_SHA256:
        print(f"STOP_FROZEN_HASH_MISMATCH", file=sys.stderr)
        return 4

    df = C.read_frozen()
    dev = df[df["mine_id"].isin(C.DEVELOPMENT_MINES)].copy()
    dev = dev[dev[C.CORE_REQUIRED].notna().all(axis=1)].reset_index(drop=True)
    folds = C.DEVELOPMENT_MINES  # all 6

    ss = load_search_spaces()
    classic_models = list(ss["search_spaces"].keys())
    tfm_models = ["tabpfn_v2", "tabicl", "tabdpt"]
    routes_models = {
        "CORE_CC": classic_models + tfm_models,
        "EXTENDED_FOLD_IMPUTED": classic_models,  # TFM deferred on EXTENDED (documented)
    }

    oof_all, met_all, run_all, best_all = [], [], [], []
    n_configs = 0
    for route, models in routes_models.items():
        for target in ["raw", "log1p"]:
            for model_name in models:
                n_configs += 1
                o, m, r, b = run_config(route, model_name, target, dev, folds, ss["search_spaces"])
                oof_all += o; met_all += m; run_all += r; best_all += b
                n_ok = sum(1 for x in r if x["status"] == "OK")
                print(f"[{n_configs:>2}] {route:22s} {model_name:24s} {target:5s} folds_ok={n_ok}/6")

    oof_df = pd.DataFrame(oof_all, columns=OOF_COLS)
    oof_df.to_csv(C.PRED_DIR / "g5_oof.csv", index=False, lineterminator="\n")
    pd.DataFrame(met_all).to_csv(C.PRED_DIR / "g5_metrics.csv", index=False, lineterminator="\n")
    pd.DataFrame(run_all).to_csv(C.PRED_DIR / "g5_runtime.csv", index=False, lineterminator="\n")
    pd.DataFrame(best_all).to_csv(C.PRED_DIR / "g5_best_params.csv", index=False, lineterminator="\n")

    failures = [r for r in run_all if r["status"] == "ERROR"]
    C.write_json(C.PRED_DIR / "g5_failures.json", {
        "generated_at_utc": C.now_utc(), "n_configs": n_configs,
        "n_failures": len(failures), "failures": failures,
    })

    print(f"\nG5 pipeline done: {n_configs} configs, OOF rows={len(oof_df)}, fold-runs_ok={len(run_all)-len(failures)}/{len(run_all)}")
    print(f"  OOF sha256 = {C.sha256(C.PRED_DIR / 'g5_oof.csv')}")
    print(f"  failures = {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
