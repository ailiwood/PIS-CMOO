"""scripts/209_g8_preoutcome_freeze.py — G8 §11 mandatory one-way pre-outcome checkpoint.

Hashes every pre-outcome artifact and verifies no forbidden outcome column leaked into any
pre-outcome data file. Records SC_M01_OUTCOME_ACCESSED = FALSE. This file MUST exist and
pass before any post_lugeon is read.

Output: 08_validation/g8/freeze/g8_preoutcome_freeze.json
"""
from __future__ import annotations
import sys, json, hashlib, datetime
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C

ROOT = C.PROJECT_ROOT
V8 = C.VALIDATION_DIR / "g8"
G8 = C.OPT_DIR / "g8"
FORBIDDEN = ["post_lugeon", "qualified", "delta_lugeon", "relative_reduction",
             "log_reduction", "flag_post_gt_pre"]


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(p):
    return str(p.relative_to(ROOT)).replace("\\", "/")


def main():
    files = [
        V8 / "freeze" / "g8_external_protocol_freeze.json",
        V8 / "preoutcome" / "g8_sc_m01_blinded_inputs.parquet",
        V8 / "preoutcome" / "g8_sc_m01_blinded_inputs_freeze.json",
        V8 / "preoutcome" / "g8_external_domain_shift_preoutcome.csv",
        V8 / "preoutcome" / "g8_external_domain_shift_summary.json",
        V8 / "preoutcome" / "g8_development_joint_support.pkl",
        V8 / "preoutcome" / "g8_external_historical_action_predictions.csv",
        V8 / "preoutcome" / "g8_external_historical_action_predictions_freeze.json",
        G8 / "g8_external_solutions.parquet",
        G8 / "g8_external_optimization_status.json",
        V8 / "preoutcome" / "g8_external_tree_representatives_prefrozen.csv",
        V8 / "preoutcome" / "g8_external_tree_representatives_prefreeze.json",
        V8 / "preoutcome" / "g8_external_tabdpt_representative_audit.csv",
        V8 / "preoutcome" / "g8_external_tabdpt_representative_audit_summary.json",
        V8 / "preoutcome" / "g8_external_context_level_LevelB_summary.json",
        V8 / "g8_catboost_conformal_freeze.json",
        V8 / "g8_tabdpt_conformal_freeze.json",
        G8 / "deployment" / "g8_catboost_deployment_bundle.json",
        G8 / "deployment" / "g8_tabdpt_deployment_bundle.json",
        G8 / "deployment" / "cache" / "catboost_final.pkl",
        G8 / "deployment" / "cache" / "tabdpt_final.pkl",
    ]
    scripts = [C.SCRIPTS_DIR / f for f in [
        "200_g8_pre_release_seal.py", "201_g8_representative_identity_audit.py",
        "202_g8_protocol_freeze.py", "203_g8_deployment_bundles.py",
        "204_g8_blinded_input.py", "205_g8_domain_shift.py",
        "206_g8_historical_predictions.py", "207_g8_external_optimization.py",
        "208_g8_external_reps_audit.py", "209_g8_preoutcome_freeze.py",
    ]]
    files += [s for s in scripts if s.exists()]

    recs = []
    for p in files:
        recs.append({"relative_path": _rel(p), "size_bytes": p.stat().st_size, "sha256": _sha(p)})

    # forbidden-outcome leak test on the data artifacts
    leaks = []
    for p in [V8 / "preoutcome" / "g8_sc_m01_blinded_inputs.parquet",
              V8 / "preoutcome" / "g8_external_historical_action_predictions.csv",
              G8 / "g8_external_solutions.parquet",
              V8 / "preoutcome" / "g8_external_tree_representatives_prefrozen.csv"]:
        if p.suffix == ".parquet":
            cols = list(pd.read_parquet(p).columns)
        else:
            cols = list(pd.read_csv(p).columns)
        for f in FORBIDDEN:
            if f in cols:
                leaks.append(f"{_rel(p)} has {f}")

    pass_ = (len(leaks) == 0)
    freeze = {
        "stage": "G8_PREOUTCOME_FREEZE",
        "SC_M01_OUTCOME_ACCESSED": False,
        "forbidden_outcome_leak": leaks,
        "preoutcome_check_pass": bool(pass_),
        "n_artifacts": len(recs),
        "artifacts": recs,
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (V8 / "freeze" / "g8_preoutcome_freeze.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"PREOUTCOME CHECK PASS: {pass_}")
    print(f"forbidden outcome leak: {leaks}")
    print(f"SC_M01_OUTCOME_ACCESSED: {freeze['SC_M01_OUTCOME_ACCESSED']}")
    print(f"n_artifacts: {len(recs)}")
    print(f"preoutcome_freeze_sha256: {_sha(V8 / 'freeze' / 'g8_preoutcome_freeze.json')}")


if __name__ == "__main__":
    main()
