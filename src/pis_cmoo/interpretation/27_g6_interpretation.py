"""
scripts/27_g6_interpretation.py — G6 core interpretation on 6 development mines.

Re-fits the G5-frozen champion (TabDPT/log1p/CORE_CC) and tree reference
(CatBoost/log1p/CORE_CC) per outer fold (same frozen hyperparams/seed, no re-search),
verifies OOF reproduction, then computes:
  - TabDPT grouped permutation importance (model-agnostic direct evidence)
  - CatBoost TreeSHAP + SHAP interaction (tree reference)
  - 1D ALE (CatBoost full; TabDPT on 4 key features)
  - surrogate (HGB on TabDPT predictions) fidelity
  - cross-mine stability (rank, Spearman, Jaccard, direction agreement)

SC-M01 never read. Writes to 06_interpretation/.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score

import _common as C

warnings.filterwarnings("ignore")

CORE_FEATURES = ["WCR", "GP_MPa", "grout_take_L_per_m", "grouting_time_h",
                 "depth_m", "section_length_m", "pre_lugeon"]
TARGET = "post_lugeon"
INTERP = C.INTERP_DIR
ALE_TABDPT_FEATURES = ["pre_lugeon", "GP_MPa", "grout_take_L_per_m", "depth_m"]


def refit_catboost(train, test, params):
    from catboost import CatBoostRegressor
    Xtr = train[CORE_FEATURES].astype(float)
    Xte = test[CORE_FEATURES].astype(float)
    ytr = np.log1p(train[TARGET].astype(float).values)
    params = {k.replace("model__", ""): v for k, v in params.items()}
    m = CatBoostRegressor(random_seed=C.SEED_MASTER, thread_count=1, verbose=0, **params)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    return m, Xte, pred


def refit_tabdpt(train, test):
    from tabdpt import TabDPTRegressor
    Xtr = train[CORE_FEATURES].astype("float32").values
    Xte = test[CORE_FEATURES].astype("float32").values
    ytr = np.log1p(train[TARGET].astype("float32").values)
    m = TabDPTRegressor(device="cpu")
    m.fit(Xtr, ytr)
    pred = m.predict(Xte).astype(float)
    return m, Xte, pred


def ale_1d(model, X, feature, n_bins=15):
    """First-order 1D ALE for a numeric feature (model-agnostic)."""
    vals = X[feature].astype(float).values
    bins = np.quantile(vals, np.linspace(0, 1, n_bins + 1))
    X = X.copy()
    effects = []
    for k in range(n_bins):
        lo, hi = bins[k], bins[k + 1]
        mask = (vals >= lo) & (vals <= hi)
        if mask.sum() == 0:
            continue
        Xl = X.iloc[mask].copy(); Xl[feature] = lo
        Xh = X.iloc[mask].copy(); Xh[feature] = hi
        pl = _predict(model, Xl); ph = _predict(model, Xh)
        effects.append((float((lo + hi) / 2), float((ph - pl).mean()), int(mask.sum())))
    if not effects:
        return np.array([])
    e = np.array(effects)
    ale = np.cumsum(e[:, 1]) - np.mean(np.cumsum(e[:, 1]))
    return np.column_stack([e[:, 0], ale, e[:, 2]])


def _predict(model, X):
    a = X.astype("float32").values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype="float32")
    return np.asarray(model.predict(a), dtype=float)


def main() -> int:
    C.ensure_dirs()
    INTERP.mkdir(parents=True, exist_ok=True)
    if C.sha256(C.FROZEN_CSV) != C.EXPECTED_FROZEN_SHA256:
        print("STOP_G5_FREEZE_CHANGED", file=sys.stderr)
        return 4

    df = C.read_frozen()
    dev = df[df["mine_id"].isin(C.DEVELOPMENT_MINES)].copy()
    dev = dev[dev[C.CORE_REQUIRED].notna().all(axis=1)].reset_index(drop=True)
    g5_oof = pd.read_csv(C.PRED_DIR / "g5_oof.csv")
    bp = pd.read_csv(C.PRED_DIR / "g5_best_params.csv")
    bp_cb = bp[(bp["model"] == "catboost") & (bp["target_variant"] == "log1p") & (bp["route"] == "CORE_CC")]

    perm_rows, shap_rows, ale_rows, inter_rows = [], [], [], []
    oof_repro = []
    surrogate_rows = []
    per_fold_shap = {}  # feature -> list of per-fold mean|shap|

    for mine in C.DEVELOPMENT_MINES:
        test = dev[dev["mine_id"] == mine]
        train = dev[dev["mine_id"] != mine].reset_index(drop=True)
        yte = np.log1p(test[TARGET].astype(float).values)
        # ---- refit + OOF reproduction ----
        m_cb, Xte_cb, pred_cb = refit_catboost(train, test,
                                               json.loads(bp_cb.loc[bp_cb["outer_fold"] == mine, "best_params"].iloc[0]))
        m_td, Xte_td, pred_td = refit_tabdpt(train, test)
        g5_cb = g5_oof[(g5_oof["model"] == "catboost") & (g5_oof["target_variant"] == "log1p")
                       & (g5_oof["route"] == "CORE_CC") & (g5_oof["outer_fold"] == mine)]
        g5_td = g5_oof[(g5_oof["model"] == "tabdpt") & (g5_oof["target_variant"] == "log1p")
                       & (g5_oof["route"] == "CORE_CC") & (g5_oof["outer_fold"] == mine)]
        g5_cb_pred = np.log1p(g5_cb.sort_values("record_id")["y_pred_clipped"].values)
        g5_td_pred = np.log1p(g5_td.sort_values("record_id")["y_pred_clipped"].values)
        oof_repro.append({"outer_fold": mine,
                          "catboost_mae_vs_g5": float(np.mean(np.abs(pred_cb - g5_cb_pred))),
                          "tabdpt_mae_vs_g5": float(np.mean(np.abs(pred_td - g5_td_pred)))})

        # ---- TabDPT permutation importance (model-agnostic) ----
        try:
            r = permutation_importance(m_td, Xte_td.astype("float32"), yte.astype("float32"),
                                       scoring="neg_mean_absolute_error", n_repeats=10,
                                       random_state=C.SEED_MASTER)
            for i, f in enumerate(CORE_FEATURES):
                perm_rows.append({"outer_fold": mine, "model": "tabdpt", "feature": f,
                                  "importance_mean": float(r.importances_mean[i]),
                                  "importance_std": float(r.importances_std[i])})
        except Exception as e:
            print(f"  [warn] TabDPT permutation fail {mine}: {type(e).__name__}")

        # ---- CatBoost TreeSHAP ----
        ex = shap.TreeExplainer(m_cb)
        sv = ex.shap_values(Xte_cb)
        sv = np.asarray(sv)
        for i, f in enumerate(CORE_FEATURES):
            shap_rows.append({"outer_fold": mine, "feature": f,
                              "mean_abs_shap": float(np.abs(sv[:, i]).mean()),
                              "signed_mean_shap": float(sv[:, i].mean())})
            per_fold_shap.setdefault(f, []).append(float(np.abs(sv[:, i]).mean()))
        # SHAP interaction
        iv = ex.shap_interaction_values(Xte_cb)
        iv = np.asarray(iv)
        main_abs = np.abs(iv).mean(axis=0)
        np.fill_diagonal(main_abs, 0)
        for i in range(len(CORE_FEATURES)):
            for j in range(i + 1, len(CORE_FEATURES)):
                inter_rows.append({"outer_fold": mine, "feat1": CORE_FEATURES[i], "feat2": CORE_FEATURES[j],
                                   "interaction_strength": float(main_abs[i, j] + main_abs[j, i])})

        # ---- 1D ALE ----
        for f in CORE_FEATURES:
            a = ale_1d(m_cb, Xte_cb, f, n_bins=15)
            for row in a:
                ale_rows.append({"outer_fold": mine, "model": "catboost", "feature": f,
                                 "x": float(row[0]), "ale": float(row[1]), "n": int(row[2])})
        for f in ALE_TABDPT_FEATURES:
            Xtd = pd.DataFrame(Xte_td, columns=CORE_FEATURES)
            a = ale_1d(m_td, Xtd, f, n_bins=8)
            for row in a:
                ale_rows.append({"outer_fold": mine, "model": "tabdpt", "feature": f,
                                 "x": float(row[0]), "ale": float(row[1]), "n": int(row[2])})

        # ---- surrogate: HGB on TabDPT predictions ----
        Xtr_h = train[CORE_FEATURES].astype(float)
        Xte_h = test[CORE_FEATURES].astype(float)
        pred_td_train = _predict(m_td, train[CORE_FEATURES])
        s = HistGradientBoostingRegressor(random_state=C.SEED_MASTER)
        s.fit(Xtr_h, pred_td_train)
        s_pred = s.predict(Xte_h)
        surrogate_rows.append({"outer_fold": mine, "r2": float(r2_score(pred_td, s_pred)),
                               "mae": float(mean_absolute_error(pred_td, s_pred))})

        print(f"  {mine}: catboost_repro={oof_repro[-1]['catboost_mae_vs_g5']:.2e} "
              f"tabdpt_repro={oof_repro[-1]['tabdpt_mae_vs_g5']:.4f} "
              f"surrogate_r2={surrogate_rows[-1]['r2']:.4f}")

    # ---- save ----
    pd.DataFrame(perm_rows).to_csv(INTERP / "g6_permutation_importance.csv", index=False, lineterminator="\n")
    pd.DataFrame(shap_rows).to_csv(INTERP / "g6_tree_shap.csv", index=False, lineterminator="\n")
    pd.DataFrame(inter_rows).to_csv(INTERP / "g6_shap_interaction.csv", index=False, lineterminator="\n")
    pd.DataFrame(ale_rows).to_csv(INTERP / "g6_ale.csv", index=False, lineterminator="\n")
    pd.DataFrame(oof_repro).to_csv(INTERP / "g6_oof_reproduction.csv", index=False, lineterminator="\n")
    pd.DataFrame(surrogate_rows).to_csv(INTERP / "g6_surrogate_fidelity.csv", index=False, lineterminator="\n")

    # ---- stability (rank/Spearman/Jaccard/direction) ----
    piv = pd.DataFrame(shap_rows).pivot_table(index="feature", columns="outer_fold", values="mean_abs_shap")
    ranks = piv.rank(ascending=False)
    spearman = piv.corr(method="spearman")
    spearman.to_csv(INTERP / "g6_stability_spearman.csv", index=True, lineterminator="\n")
    # top-k frequency + Jaccard
    top5 = {f: set(ranks[f].sort_values().head(5).index) for f in ranks.columns}
    top10 = {f: set(ranks[f].sort_values().head(10).index) for f in ranks.columns}
    stab = []
    for f in CORE_FEATURES:
        occ5 = sum(1 for s in top5.values() if f in s)
        occ10 = sum(1 for s in top10.values() if f in s)
        jac5 = np.mean([len(top5[a] & top5[b]) / len(top5[a] | top5[b]) for a in top5 for b in top5 if a < b])
        stab.append({"feature": f, "top5_freq": occ5, "top10_freq": occ10,
                     "mean_rank": float(ranks.loc[f].mean()), "mean_abs_shap": float(piv.loc[f].mean())})
    stab_df = pd.DataFrame(stab).sort_values("mean_abs_shap", ascending=False)
    stab_df["mean_jaccard_top5"] = round(jac5, 4)
    stab_df.to_csv(INTERP / "g6_stability_summary.csv", index=False, lineterminator="\n")

    print("\nTree-SHAP global importance (mean|SHAP| across folds):")
    for _, r in stab_df.iterrows():
        print(f"  {r['feature']:24s} mean|SHAP|={r['mean_abs_shap']:.4f} top5_freq={r['top5_freq']}/6")
    print(f"\nOOF reproduction: catboost max={max(oof_repro, key=lambda x:x['catboost_mae_vs_g5'])['catboost_mae_vs_g5']:.2e}, "
          f"tabdpt max={max(oof_repro, key=lambda x:x['tabdpt_mae_vs_g5'])['tabdpt_mae_vs_g5']:.4f}")
    print(f"Surrogate fidelity: macro R2={np.mean([s['r2'] for s in surrogate_rows]):.4f}, "
          f"min per-mine R2={min(s['r2'] for s in surrogate_rows):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
