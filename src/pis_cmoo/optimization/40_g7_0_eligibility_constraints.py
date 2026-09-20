"""
scripts/40_g7_0_eligibility_constraints.py — G7 eligibility manifest + hard-constraint freeze.

Starts from the G5 6-mine CORE_CC list (1433 dev), then excludes only records where a hard
safety constraint or decision variable is missing (phase-wise complete case). NO global row
deletion (processed stays 1766). Freezes the 5 hard safety constraints.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import _common as C

READY = C.DATA_DIR / "analysis_ready" / "g7_0"
MANI = C.MANIFEST_DIR / "g7_0"
OPT = C.OPT_DIR / "g7_0_constraints"
HARD = ["allowable_GP_MPa", "min_WCR", "max_WCR", "max_grout_take_L_per_m", "max_grouting_time_h"]
DECISION = ["WCR", "GP_MPa", "grout_take_L_per_m", "grouting_time_h"]


def main() -> int:
    for d in [READY, MANI, OPT]:
        d.mkdir(parents=True, exist_ok=True)

    frz = C.read_frozen()
    core_ok = frz[C.CORE_REQUIRED].notna().all(axis=1)
    dev = frz["mine_id"].isin(C.DEVELOPMENT_MINES)

    # G7 eligible: dev + CORE_CC + hard constraints + decision vars present
    eligible = core_ok & dev & frz[HARD + DECISION].notna().all(axis=1)

    # exclusion reason
    def reason(row):
        miss = [c for c in HARD + DECISION if pd.isna(row[c])]
        if not core_ok[row.name]:
            miss = ["core_incomplete"] + miss
        if not dev[row.name]:
            miss = ["external_mine"] + miss
        return ";".join(miss) if miss else ""

    frz = frz.copy()
    frz["eligible_g7"] = eligible.astype(int)
    frz["exclusion_reason_g7"] = frz.apply(reason, axis=1)
    frz["analysis_cohort"] = frz["mine_id"].apply(lambda m: "external" if m == C.EXTERNAL_MINE else "development")

    manifest = frz[["record_id", "mine_id", "project_id", "borehole_id", "analysis_cohort",
                    "eligible_g7", "exclusion_reason_g7"]]
    manifest.to_csv(MANI / "g7_development_eligibility_manifest_v1.csv", index=False, lineterminator="\n")

    # contexts_eligible (stage subset, dev only)
    contexts = frz[eligible][["record_id", "mine_id", "project_id", "borehole_id",
                              "depth_m", "groundwater_pressure_MPa", "pre_lugeon",
                              "WCR", "GP_MPa", "grout_take_L_per_m", "grouting_time_h",
                              "allowable_GP_MPa", "min_WCR", "max_WCR",
                              "max_grout_take_L_per_m", "max_grouting_time_h",
                              "post_lugeon", "acceptance_threshold_lu", "lithology"]].copy()
    contexts.to_csv(READY / "g7_contexts_eligible_v1.csv", index=False, lineterminator="\n")

    # per-mine exclusion stats
    stats = []
    for m in C.DEVELOPMENT_MINES:
        sub = frz[frz["mine_id"] == m]
        core_n = int(sub[core_ok[sub.index]].sum()) if False else int(sub[C.CORE_REQUIRED].notna().all(axis=1).sum())
        elig_n = int((sub["eligible_g7"] == 1).sum())
        excl_n = core_n - elig_n
        stats.append({"mine_id": m, "core_cc_n": core_n, "g7_eligible_n": elig_n,
                      "g7_excluded_n": excl_n,
                      "exclusion_rate": round(excl_n / core_n, 4) if core_n else 0.0})
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(MANI / "g7_0_exclusion_stats.csv", index=False, lineterminator="\n")

    # included vs excluded comparison (depth, pre_lugeon, lithology)
    inc = frz[eligible]
    exc = frz[core_ok & dev & ~eligible]
    comp = []
    for c in ["depth_m", "pre_lugeon"]:
        comp.append({"variable": c, "included_median": float(inc[c].median()),
                     "excluded_median": float(exc[c].median()) if len(exc) else None,
                     "excluded_n": int(len(exc))})
    comp_df = pd.DataFrame(comp)
    comp_df.to_csv(MANI / "g7_0_included_vs_excluded.csv", index=False, lineterminator="\n")

    # ---- hard-constraint freeze ----
    constraints = {
        "generated_at_utc": C.now_utc(),
        "stage": "G7_0_CONSTRAINT_FREEZE",
        "constraints": [
            {"name": "WCR_bounds", "rule": "min_WCR <= WCR <= max_WCR",
             "variables": ["WCR", "min_WCR", "max_WCR"], "type": "numeric_site_recorded"},
            {"name": "pressure_upper", "rule": "0 < GP_MPa <= allowable_GP_MPa_clean",
             "variables": ["GP_MPa", "allowable_GP_MPa"], "type": "numeric_site_recorded"},
            {"name": "grout_take_upper", "rule": "0 <= grout_take_L_per_m <= max_grout_take_L_per_m",
             "variables": ["grout_take_L_per_m", "max_grout_take_L_per_m"], "type": "numeric_site_recorded"},
            {"name": "grouting_time_upper", "rule": "0 < grouting_time_h <= max_grouting_time_h",
             "variables": ["grouting_time_h", "max_grouting_time_h"], "type": "numeric_site_recorded"},
            {"name": "conformal_safety", "rule": "one-sided conformal upper bound(post_lugeon) <= acceptance_threshold_lu",
             "variables": ["post_lugeon", "acceptance_threshold_lu"], "type": "conformal_safety"},
        ],
        "decision_variables": DECISION,
        "fixed_context": ["depth_m", "groundwater_pressure_MPa", "lithology", "karst_class",
                          "fracture_class", "hydrogeological_regime"],
        "pressure_repair_note": ("row-level allowable_GP_MPa already integrates site + hydrogeology + "
                                 "engineer judgment; treat as physics/engineering-informed repair. "
                                 "No cubic-law verification without fracture aperture + hydraulic gradient."),
        "route_R1_site_recorded": "primary; source-table site constraints",
        "route_R2_code_informed": "sensitivity; code-derived conservative bounds (only if code quantifies)",
        "route_R3_intersection": "conservative; intersection of site + code bounds",
    }
    C.write_json(OPT / "g7_0_constraint_freeze.json", constraints)

    print("G7 eligibility + constraints frozen.")
    print(f"  dev CORE_CC={int((core_ok & dev).sum())}, G7 eligible={int(eligible.sum())}, excluded={int((core_ok & dev & ~eligible).sum())}")
    for _, r in stats_df.iterrows():
        print(f"  {r['mine_id']}: core={r['core_cc_n']} eligible={r['g7_eligible_n']} excluded={r['g7_excluded_n']} rate={r['exclusion_rate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
