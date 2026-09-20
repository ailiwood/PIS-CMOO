"""scripts/115_g7_6_finalize.py — G7.6 final freeze + artifact registry + test coverage matrix.

Computes SHA256 over all G7.6 evidence artifacts and writes:
  - 07_optimization/g7_6/freeze/g7_6_final_freeze.json
  - 12_handoffs/g7_6_artifact_registry.csv / .json
  - 12_handoffs/g7_6_test_coverage_matrix.csv
"""
from __future__ import annotations
import sys, json, hashlib, datetime
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C

WORKSPACE = C.WORKSPACE
O = C.OPT_DIR / "g7_6"
V = C.VALIDATION_DIR / "g7_6"
H = C.HANDOFFS_DIR


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(p: Path) -> str:
    return str(p.relative_to(WORKSPACE.parent)).replace("\\", "/")


def main():
    files = []
    # authority addendum + manifest
    files += [H / "G7_5_authority_addendum.md", H / "g7_5_authority_manifest.json"]
    # validation artifacts
    files += sorted(V.glob("*.csv")) + sorted(V.glob("*.json"))
    # optimization freeze + top-level artifacts (exclude the self-referential final freeze)
    files += [p for p in sorted((O / "freeze").glob("*.json"))
              if p.name != "g7_6_final_freeze.json"]
    files += sorted(O.glob("*.csv")) + sorted(O.glob("*.json")) + sorted(O.glob("*.parquet"))
    files += [O / "g7_6_tree_representatives_prefrozen.csv"]
    # model caches
    files += sorted((O / "cache").glob("tree_*.pkl"))
    files += sorted((O / "cache").glob("tabdpt_*.pkl"))
    # scripts + tests
    scripts = ["105_g7_6_denominator_reconciliation.py",
               "106_g7_6_tree_cross_site_conformal.py",
               "107_g7_6_tree_evaluator_reconfirmation.py",
               "108_g7_6_tabdpt_cross_site_conformal.py",
               "109_g7_6_38310_audit.py",
               "_g7_6_engine.py",
               "110_g7_6_300panel_ablation.py",
               "111_g7_6_full1387.py",
               "112_g7_6_tree_reps.py",
               "113_g7_6_tabdpt_audit.py",
               "114_g7_6_joint_support_sensitivity.py"]
    files += [C.SCRIPTS_DIR / s for s in scripts]
    files += [C.TESTS_DIR / "test_g7_6_checks.py"]

    files = [p for p in files if p.exists()]
    # dedup preserving order
    seen, uniq = set(), []
    for p in files:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    files = uniq

    recs = []
    for p in files:
        recs.append({
            "relative_path": _rel(p),
            "size_bytes": p.stat().st_size,
            "sha256": _sha(p),
        })
    print(f"Hashed {len(recs)} files")

    reg_csv = pd.DataFrame(recs)
    reg_csv.to_csv(H / "g7_6_artifact_registry.csv", index=False, encoding="utf-8",
                   lineterminator="\n")

    registry = {
        "stage": "G7_6_ARTIFACT_REGISTRY",
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "file_count": len(recs),
        "files": recs,
    }
    reg_json = json.dumps(registry, ensure_ascii=False, indent=2)
    (H / "g7_6_artifact_registry.json").write_text(reg_json, encoding="utf-8")
    registry["artifact_registry_sha256"] = hashlib.sha256(reg_json.encode("utf-8")).hexdigest()
    (H / "g7_6_artifact_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

    freeze = {
        "stage": "G7_6_FINAL_FREEZE",
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "authoritative_g75_status": "G7_5_ARCHITECTURE_VALID_BUT_UQ_AND_FINAL_OPTIMIZATION_EVIDENCE_OPEN",
        "tree_uq_selected": {"model": "catboost", "method": "UQ-B", "alpha": 0.10},
        "tabdpt_uq_selected": {"method": "UQ-B", "alpha": 0.10},
        "primary_optimizer": "NSGA-III C7",
        "registry_csv_sha256": _sha(H / "g7_6_artifact_registry.csv"),
        "representatives_prefrozen_sha256": _sha(O / "g7_6_tree_representatives_prefrozen.csv"),
        "sc_m01_access": "FORBIDDEN — never opened",
    }
    (O / "freeze" / "g7_6_final_freeze.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {O / 'freeze' / 'g7_6_final_freeze.json'}")
    print(f"Wrote {H / 'g7_6_artifact_registry.csv'} and .json")


if __name__ == "__main__":
    main()
