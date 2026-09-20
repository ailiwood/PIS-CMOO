"""scripts/210_g8_unblind.py — G8 §12 ONE-TIME SC-M01 outcome unblinding.

This is the ONLY script that reads SC-M01 post_lugeon. It loads the outcome columns for
SC-M01 rows only, verifies exact record_id match against the frozen blinded input, and
writes a clean outcome file for the evaluation phase. Records UTC timestamp + hashes.

After this point NO scientific protocol modification is permitted.
"""
from __future__ import annotations
import sys, json, hashlib, datetime
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C

V8 = C.VALIDATION_DIR / "g8"
OUT = V8 / "outcome"
OUT.mkdir(parents=True, exist_ok=True)


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    # blinded inputs reference
    blinded = pd.read_parquet(V8 / "preoutcome" / "g8_sc_m01_blinded_inputs.parquet")
    blind_ids = set(blinded["record_id"])

    # one-time outcome load (SC-M01 rows only)
    outcome_cols = ["record_id", "post_lugeon", "acceptance_threshold_lu",
                    "delta_lugeon", "relative_reduction", "log_reduction"]
    df = pd.read_csv(C.FROZEN_CSV, usecols=["record_id", "mine_id"] + outcome_cols[1:])
    sc = df[df["mine_id"] == "SC-M01"].reset_index(drop=True)
    sc_ids = set(sc["record_id"])

    # exact match check
    assert sc_ids == blind_ids, f"record_id mismatch: blind={len(blind_ids)} outcome={len(sc_ids)}"
    assert len(sc) == len(blinded), "row count mismatch"

    # derive qualified from frozen threshold
    sc["qualified"] = sc["post_lugeon"] <= sc["acceptance_threshold_lu"]

    sc.to_parquet(OUT / "g8_sc_m01_outcome.parquet", index=False)

    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    freeze = {
        "stage": "G8_SC_M01_OUTCOME_UNBLIND",
        "unblind_timestamp_utc": ts,
        "outcome_file": "08_validation/g8/outcome/g8_sc_m01_outcome.parquet",
        "outcome_file_sha256": _sha(OUT / "g8_sc_m01_outcome.parquet"),
        "n_records": int(len(sc)),
        "record_id_exact_match": True,
        "record_id_sha256": hashlib.sha256("\n".join(sorted(sc["record_id"])).encode("utf-8")).hexdigest(),
        "outcome_columns": ["post_lugeon", "qualified", "delta_lugeon", "relative_reduction", "log_reduction"],
        "no_post_unblind_protocol_modification": True,
        "script_sha256": _sha(Path(__file__)),
    }
    (OUT / "g8_sc_m01_outcome_unblind_freeze.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(freeze, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
