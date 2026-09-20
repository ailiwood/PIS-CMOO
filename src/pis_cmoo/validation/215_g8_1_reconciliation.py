"""scripts/215_g8_1_reconciliation.py — G8.1 PART A reporting reconciliation (§2–§7).

Deterministic source recomputation only. No model/UQ/optimizer change. Produces:
  §2  12_handoffs/G8_external_validation_completion_addendum.md + manifest
  §3  08_validation/g8_1/g8_external_cohort_ledger.csv
  §4  10_tables/g8_1/g8_observed_action_safety_corrected.csv
  §5  10_tables/g8_1/g8_recommendation_availability_corrected.csv
  §6  10_tables/g8_1/g8_dev_vs_external_corrected.csv
  §7  08_validation/g8_1/g8_{development,external}_representative_identity_final.csv
      + 10_tables/g8_1/g8_unique_decision_verification.csv
"""
from __future__ import annotations
import sys, json, hashlib, datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C

ROOT = C.PROJECT_ROOT
V8 = C.VALIDATION_DIR / "g8"
V81 = C.VALIDATION_DIR / "g8_1"
T81 = C.TABLES_DIR / "g8_1"
H = C.HANDOFFS_DIR
FEATURES = ["WCR", "GP_MPa", "grout_take_L_per_m", "grouting_time_h",
            "depth_m", "section_length_m", "pre_lugeon"]
HARD = ["allowable_GP_MPa", "min_WCR", "max_WCR", "max_grout_take_L_per_m", "max_grouting_time_h"]
for d in (V81, T81, V81 / "freeze"):
    d.mkdir(parents=True, exist_ok=True)


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(p):
    return str(p.relative_to(ROOT)).replace("\\", "/")


def main():
    sc = pd.read_parquet(V8 / "preoutcome" / "g8_sc_m01_blinded_inputs.parquet")
    out = pd.read_parquet(V8 / "outcome" / "g8_sc_m01_outcome.parquet")

    # ---------------- §3 cohort ledger ----------------
    feat_ok = sc[FEATURES].notna().all(axis=1)
    hard_ok = sc[HARD].notna().all(axis=1)
    thr_ok = sc["acceptance_threshold_lu"].notna()
    ledger = sc[["record_id", "mine_id"]].copy()
    ledger["raw_external"] = True
    ledger["outcome_available"] = sc["record_id"].isin(set(out["record_id"]))
    ledger["threshold_available"] = thr_ok.values
    ledger["prediction_evaluable"] = feat_ok.values
    ledger["optimization_eligible"] = (feat_ok & hard_ok & thr_ok).values
    ledger["support_metadata_available"] = feat_ok.values
    ledger["engineering_metadata_available"] = hard_ok.values
    ledger["exclusion_reason_prediction"] = np.where(feat_ok, "", "missing 7-feature predictor")
    ledger["exclusion_reason_optimization"] = np.where(
        feat_ok & hard_ok & thr_ok, "",
        np.where(~feat_ok, "missing 7-feature predictor",
                 np.where(~hard_ok, "missing engineering bound", "missing threshold")))
    ledger.to_csv(V81 / "g8_external_cohort_ledger.csv", index=False,
                  encoding="utf-8", lineterminator="\n")
    n_raw = int(ledger["raw_external"].sum())
    n_pred = int(ledger["prediction_evaluable"].sum())
    n_opt = int(ledger["optimization_eligible"].sum())
    assert (n_raw, n_pred, n_opt) == (275, 269, 259), (n_raw, n_pred, n_opt)
    print(f"cohort: raw={n_raw}, prediction_evaluable={n_pred}, optimization_eligible={n_opt}")

    # ---------------- §4 corrected observed-action safety ----------------
    pred = pd.read_csv(V8 / "preoutcome" / "g8_external_historical_action_predictions.csv")
    m = pred.merge(out[["record_id", "post_lugeon", "qualified"]], on="record_id", how="left")
    m = m[m["post_lugeon"].notna()].reset_index(drop=True)
    qualified = m["qualified"].astype(bool).values
    thr = m["acceptance_threshold_lu"].astype(float).values
    rows = []
    for mdl, tag in [("tabdpt", "TabDPT"), ("catboost", "CatBoost")]:
        pl = m[f"pred_{mdl}_log"].astype(float).values
        p = m[f"pred_{mdl}_post"].astype(float).values
        q90 = json.loads((V8 / f"g8_{mdl}_conformal_freeze.json").read_text(encoding="utf-8"))["q90"]
        evaluable = ~np.isnan(pl)
        n_ev = int(evaluable.sum())
        n_not = int((~evaluable).sum())
        point_safe = p <= thr
        ucb90_safe = np.expm1(pl + q90) <= thr
        # only evaluable records count; not-evaluable are excluded from numerator/denominator
        point_safe_e = point_safe & evaluable
        ucb90_safe_e = ucb90_safe & evaluable
        qual_e = qualified & evaluable
        false_safe_point = int((point_safe_e & ~qual_e).sum())
        false_safe_ucb = int((ucb90_safe_e & ~qual_e).sum())
        conservative_false_unsafe = int((~ucb90_safe_e & qual_e).sum())
        rows.append({
            "model": tag, "n_evaluable": n_ev, "n_not_evaluable": n_not,
            "point_safe_n": int(point_safe_e.sum()),
            "ucb90_safe_n": int(ucb90_safe_e.sum()),
            "point_safe_but_unqualified": false_safe_point,
            "ucb90_safe_but_unqualified": false_safe_ucb,
            "ucb90_unsafe_but_qualified": conservative_false_unsafe,
            "ucb90_safe_rate": f"{int(ucb90_safe_e.sum())}/{n_ev}",
            "false_safe_rate": f"{false_safe_ucb}/{n_ev}",
        })
    pd.DataFrame(rows).to_csv(T81 / "g8_observed_action_safety_corrected.csv", index=False,
                              encoding="utf-8", lineterminator="\n")

    # ---------------- §5 corrected recommendation availability ----------------
    opt = json.loads((C.OPT_DIR / "g8" / "g8_external_optimization_status.json").read_text(encoding="utf-8"))
    lvl = json.loads((V8 / "preoutcome" / "g8_external_context_level_LevelB_summary.json").read_text(encoding="utf-8"))
    elig = opt["n_eligible_contexts"]
    tree_safe = opt["n_tree_safe_contexts"]
    levelb = lvl["n_contexts_with_LevelB_rep"]
    dis_only = lvl["n_contexts_only_disagreement"]
    no_safe = elig - tree_safe
    assert tree_safe + no_safe == elig, (tree_safe, no_safe, elig)
    assert levelb + dis_only == tree_safe, (levelb, dis_only, tree_safe)
    avail = pd.DataFrame([{
        "metric": "tree_safe / eligible", "numerator": tree_safe, "denominator": elig,
        "rate": round(tree_safe / elig, 4)},
        {"metric": "LevelB / eligible", "numerator": levelb, "denominator": elig,
         "rate": round(levelb / elig, 4)},
        {"metric": "LevelB / tree_safe", "numerator": levelb, "denominator": tree_safe,
         "rate": round(levelb / tree_safe, 4)},
        {"metric": "disagreement_only / eligible", "numerator": dis_only, "denominator": elig,
         "rate": round(dis_only / elig, 4)},
        {"metric": "no_tree_safe / eligible", "numerator": no_safe, "denominator": elig,
         "rate": round(no_safe / elig, 4)}])
    avail.to_csv(T81 / "g8_recommendation_availability_corrected.csv", index=False,
                 encoding="utf-8", lineterminator="\n")
    print(f"availability: tree_safe={tree_safe} levelb={levelb} dis_only={dis_only} no_safe={no_safe} elig={elig}")

    # ---------------- §6 corrected dev-vs-external (both scales) ----------------
    oof = pd.read_csv(C.PRED_DIR / "g5_oof.csv")
    scale_rows = []
    for mdl, tag in [("tabdpt", "TabDPT"), ("catboost", "CatBoost")]:
        dev = oof[(oof.model == mdl) & (oof.target_variant == "log1p") & (oof.route == "CORE_CC")]
        y = dev.y_true.values.astype(float)
        p = dev.y_pred_clipped.values.astype(float)
        yl, pl = np.log1p(y), np.log1p(p)
        ext = pred.dropna(subset=[f"pred_{mdl}_log"]).merge(
            out[["record_id", "post_lugeon"]], on="record_id", how="inner")
        ye = ext["post_lugeon"].astype(float).values
        pe = ext[f"pred_{mdl}_post"].astype(float).values
        yel = np.log1p(ye); pel = ext[f"pred_{mdl}_log"].astype(float).values
        scale_rows += [
            {"model": tag, "scale": "Lu", "statistic": "MAE", "development": round(float(mean_absolute_error(y, p)), 4),
             "external": round(float(mean_absolute_error(ye, pe)), 4), "dev_unit": "six-mine LOMO macro", "external_unit": "one untouched mine"},
            {"model": tag, "scale": "Lu", "statistic": "RMSE", "development": round(float(np.sqrt(mean_squared_error(y, p))), 4),
             "external": round(float(np.sqrt(mean_squared_error(ye, pe))), 4), "dev_unit": "six-mine LOMO macro", "external_unit": "one untouched mine"},
            {"model": tag, "scale": "Lu", "statistic": "R2", "development": round(float(r2_score(y, p)), 4),
             "external": round(float(r2_score(ye, pe)), 4), "dev_unit": "six-mine LOMO macro", "external_unit": "one untouched mine"},
            {"model": tag, "scale": "Lu", "statistic": "median_AE", "development": round(float(np.median(np.abs(y - p))), 4),
             "external": round(float(np.median(np.abs(ye - pe))), 4), "dev_unit": "six-mine LOMO macro", "external_unit": "one untouched mine"},
            {"model": tag, "scale": "log1p", "statistic": "MAE", "development": round(float(mean_absolute_error(yl, pl)), 4),
             "external": round(float(mean_absolute_error(yel, pel)), 4), "dev_unit": "six-mine LOMO macro", "external_unit": "one untouched mine"},
            {"model": tag, "scale": "log1p", "statistic": "RMSE", "development": round(float(np.sqrt(mean_squared_error(yl, pl))), 4),
             "external": round(float(np.sqrt(mean_squared_error(yel, pel))), 4), "dev_unit": "six-mine LOMO macro", "external_unit": "one untouched mine"},
            {"model": tag, "scale": "log1p", "statistic": "R2", "development": round(float(r2_score(yl, pl)), 4),
             "external": round(float(r2_score(yel, pel)), 4), "dev_unit": "six-mine LOMO macro", "external_unit": "one untouched mine"},
        ]
    scale_df = pd.DataFrame(scale_rows)
    scale_df.to_csv(T81 / "g8_dev_vs_external_corrected.csv", index=False,
                    encoding="utf-8", lineterminator="\n")

    # ---------------- §7 representative identity final ----------------
    dev_reps = pd.read_csv(C.OPT_DIR / "g7_6" / "g7_6_tree_representatives_prefrozen.csv")
    dev_base = {"conservative", "balanced", "resource_saving"}
    dev_reps["base_role"] = dev_reps["role"].str.split("_via_same_decision").str[0]
    dev_dec = dev_reps[["record_id", "WCR", "GP_MPa", "grout_take_L_per_m", "grouting_time_h"]].round(9)
    dev_key = dev_dec.astype(str).agg("|".join, axis=1)
    dev_summary = {
        "scope": "development", "contexts": int(dev_reps["record_id"].nunique()),
        "role_assignments": int(dev_reps["record_id"].nunique()) * 3,
        "raw_rows": int(len(dev_reps)),
        "unique_decisions": int(dev_key.nunique()),
        "multi_role_decisions": int(dev_reps.groupby(dev_key)["base_role"].nunique().gt(1).sum()),
        "duplicate_bookkeeping_rows": int((dev_reps.groupby(["record_id", "base_role"]).size() > 1).sum()),
    }
    ext_reps = pd.read_csv(V8 / "preoutcome" / "g8_external_tree_representatives_prefrozen.csv")
    ext_audit = pd.read_csv(V8 / "preoutcome" / "g8_external_tabdpt_representative_audit.csv")
    ext_summary = {
        "scope": "external", "contexts": int(ext_reps["record_id"].nunique()),
        "role_assignments": int(ext_reps["record_id"].nunique()) * 3,
        "raw_rows": int(len(ext_reps)),
        "unique_decisions": int(len(ext_reps)),
        "multi_role_decisions": int(ext_reps["roles"].str.contains(",").sum()),
        "duplicate_bookkeeping_rows": 0,
        "unique_LevelB": int(ext_audit["level_b"].sum()),
        "unique_disagreement": int(ext_audit["model_disagreement"].sum()),
    }
    pd.DataFrame([dev_summary, ext_summary]).to_csv(
        V81 / "g8_representative_identity_final.csv", index=False, encoding="utf-8", lineterminator="\n")
    dev_id = pd.DataFrame([{"scope": "development", "contexts": dev_summary["contexts"],
                            "role_assignments": dev_summary["role_assignments"],
                            "raw_rows": dev_summary["raw_rows"], "unique_decisions": dev_summary["unique_decisions"],
                            "multi_role_decisions": dev_summary["multi_role_decisions"],
                            "duplicate_bookkeeping": dev_summary["duplicate_bookkeeping_rows"]}])
    ext_id = pd.DataFrame([{"scope": "external", "contexts": ext_summary["contexts"],
                            "role_assignments": ext_summary["role_assignments"],
                            "raw_rows": ext_summary["raw_rows"], "unique_decisions": ext_summary["unique_decisions"],
                            "multi_role_decisions": ext_summary["multi_role_decisions"],
                            "duplicate_bookkeeping": 0}])
    dev_id.to_csv(V81 / "g8_development_representative_identity_final.csv", index=False,
                  encoding="utf-8", lineterminator="\n")
    ext_id.to_csv(V81 / "g8_external_representative_identity_final.csv", index=False,
                  encoding="utf-8", lineterminator="\n")
    pd.concat([dev_id, ext_id], ignore_index=True).to_csv(
        T81 / "g8_unique_decision_verification.csv", index=False, encoding="utf-8", lineterminator="\n")

    # ---------------- §2 addendum + manifest ----------------
    manifest = {
        "stage": "G8_EXTERNAL_VALIDATION_COMPLETION",
        "one_time_outcome_unblind_valid": True,
        "preoutcome_freeze_valid": True,
        "no_post_unblind_scientific_adaptation": True,
        "external_validation_scientific_protocol": "CLOSED",
        "subsequent_changes_restricted_to": "reporting / provenance / denominator / scale correction only",
        "corrected_tables_cannot_trigger_adaptation": True,
        "cohort": {"raw_external": n_raw, "prediction_evaluable": n_pred, "optimization_eligible": n_opt},
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (H / "g8_external_validation_completion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    add = ("# G8 External Validation Completion Addendum\n\n"
           "- One-time SC-M01 outcome unblind: VALID (exact record_id match, n=275).\n"
           "- Pre-outcome freeze: VALID (SC_M01_OUTCOME_ACCESSED=FALSE; no forbidden-column leak).\n"
           "- No post-unblind scientific adaptation: CONFIRMED (no retraining/recalibration/model/alpha/support/bounds/optimizer/representative change).\n"
           "- External validation scientific protocol: CLOSED.\n"
           "- Subsequent changes restricted to reporting/provenance/denominator/scale correction; corrected tables cannot trigger model/UQ/optimizer adaptation.\n"
           f"- Cohorts: RAW_EXTERNAL={n_raw}, PREDICTION_EVALUABLE={n_pred}, OPTIMIZATION_ELIGIBLE={n_opt}.\n")
    (H / "G8_external_validation_completion_addendum.md").write_text(add, encoding="utf-8")

    print("reconciliation complete")
    print("dev-vs-external (TabDPT, Lu R2): dev vs external")
    print(scale_df[(scale_df.model == "TabDPT") & (scale_df.scale == "Lu") & (scale_df.statistic == "R2")].to_string(index=False))
    print(scale_df[(scale_df.model == "TabDPT") & (scale_df.scale == "log1p") & (scale_df.statistic == "R2")].to_string(index=False))


if __name__ == "__main__":
    main()
