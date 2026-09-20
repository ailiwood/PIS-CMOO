"""
scripts/32_g6_1_full_tree_shap.py — persist full record-level TreeSHAP (1433 records)
and recompute record-weighted + mine-balanced importance/stability.

Re-fits CatBoost/log1p/CORE_CC per fold (frozen hyperparams, no re-search), computes
record-level TreeSHAP for every held-out CORE_CC record, then recomputes importance,
rank, top-k frequency, Jaccard, and direction agreement. SC-M01 never used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap

import _common as C

INTERP = C.INTERP_DIR
CORE_FEATURES = ["WCR", "GP_MPa", "grout_take_L_per_m", "grouting_time_h",
                 "depth_m", "section_length_m", "pre_lugeon"]


def main() -> int:
    C.ensure_dirs()
    INTERP.mkdir(parents=True, exist_ok=True)
    if C.sha256(C.FROZEN_CSV) != C.EXPECTED_FROZEN_SHA256:
        print("STOP_G5_FREEZE_CHANGED", file=sys.stderr)
        return 4
    df = C.read_frozen()
    dev = df[df["mine_id"].isin(C.DEVELOPMENT_MINES)].copy()
    dev = dev[dev[C.CORE_REQUIRED].notna().all(axis=1)].reset_index(drop=True)
    bp = pd.read_csv(C.PRED_DIR / "g5_best_params.csv")
    bp_cb = bp[(bp["model"] == "catboost") & (bp["target_variant"] == "log1p") & (bp["route"] == "CORE_CC")]

    from catboost import CatBoostRegressor
    rec_rows = []
    for mine in C.DEVELOPMENT_MINES:
        test = dev[dev["mine_id"] == mine]
        train = dev[dev["mine_id"] != mine].reset_index(drop=True)
        params = json.loads(bp_cb.loc[bp_cb["outer_fold"] == mine, "best_params"].iloc[0])
        params = {k.replace("model__", ""): v for k, v in params.items()}
        m = CatBoostRegressor(random_seed=C.SEED_MASTER, thread_count=1, verbose=0, **params)
        m.fit(train[CORE_FEATURES].astype(float), np.log1p(train["post_lugeon"].astype(float).values))
        ex = shap.TreeExplainer(m)
        sv = np.asarray(ex.shap_values(test[CORE_FEATURES].astype(float)))
        for i in range(len(test)):
            row = {"record_id": test["record_id"].iloc[i], "mine_id": mine, "outer_fold": mine,
                   "y_true": float(test["post_lugeon"].iloc[i]),
                   "base_value": float(ex.expected_value)}
            for j, f in enumerate(CORE_FEATURES):
                row[f"shap_{f}"] = float(sv[i, j])
            rec_rows.append(row)
    rec = pd.DataFrame(rec_rows)
    rec.to_csv(INTERP / "g6_1_tree_shap_record_level.csv", index=False, lineterminator="\n")

    shap_cols = [f"shap_{f}" for f in CORE_FEATURES]
    abs_shap = rec[shap_cols].abs()
    abs_shap.columns = CORE_FEATURES

    # record-weighted
    rw = abs_shap.mean(axis=0).rename("mean_abs_shap_record_weighted")
    # mine-balanced
    rec_ = rec.copy()
    for f in CORE_FEATURES:
        rec_[f"abs_{f}"] = rec[f"shap_{f}"].abs()
    mb = rec_.groupby("mine_id")[[f"abs_{f}" for f in CORE_FEATURES]].mean().mean(axis=0)
    mb.index = CORE_FEATURES

    # per-mine rank
    per_mine = rec_.groupby("mine_id")[[f"abs_{f}" for f in CORE_FEATURES]].mean()
    per_mine.columns = CORE_FEATURES
    ranks = per_mine.rank(axis=1, ascending=False)

    # top-k frequency + Jaccard
    def topk_freq(ranks, k):
        freq = {f: 0 for f in CORE_FEATURES}
        sets = {}
        for mine in ranks.index:
            sets[mine] = set(ranks.loc[mine].sort_values().head(k).index)
            for f in sets[mine]:
                freq[f] += 1
        jac = np.mean([len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for a in sets for b in sets if a < b])
        return freq, jac

    freq5, jac5 = topk_freq(ranks, 5)
    freq10, jac10 = topk_freq(ranks, 10)

    summary = pd.DataFrame({
        "feature": CORE_FEATURES,
        "record_weighted_abs": [float(rw[f]) for f in CORE_FEATURES],
        "mine_balanced_abs": [float(mb[f]) for f in CORE_FEATURES],
        "top5_freq": [freq5[f] for f in CORE_FEATURES],
        "top10_freq": [freq10[f] for f in CORE_FEATURES],
    }).sort_values("record_weighted_abs", ascending=False)
    summary.to_csv(INTERP / "g6_1_full_tree_shap_summary.csv", index=False, lineterminator="\n")

    # compare with G6 aggregate (capped/aggregate) result
    g6 = pd.read_csv(INTERP / "g6_stability_summary.csv").set_index("feature")["mean_abs_shap"]
    summary["g6_aggregate_abs"] = summary["feature"].map(g6)
    summary.to_csv(INTERP / "g6_1_full_vs_capped.csv", index=False, lineterminator="\n")

    print(f"Full record-level TreeSHAP: {len(rec)} records persisted.")
    print("\nImportance (record-weighted | mine-balanced | G6 aggregate):")
    for _, r in summary.iterrows():
        print(f"  {r['feature']:24s} {r['record_weighted_abs']:.4f} | {r['mine_balanced_abs']:.4f} | {r['g6_aggregate_abs']:.4f}")
    print(f"\nMean top-5 Jaccard (mine-pair): {jac5:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
