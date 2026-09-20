# Data Availability

The PIS-CMOO framework was developed and frozen against a **real multi-site grouting field dataset**. That dataset is **NOT** distributed with this public release because it was collected under commercial confidentiality and contains operationally sensitive parameters.

This README documents:

1. the three-level data-disclosure classification used by this release,
2. what is distributed, what is restricted, and what is synthetic, and
3. how a third-party user can run the pipeline against their own data.

---

## 1. Disclosure classification

### Level A — Public (distributed with this release)

These are non-identifiable aggregate measurements / protocol declarations:

* `artifacts/frozen/g5_selection_freeze.json` — G5 model-selection freeze (no record-level data)
* `artifacts/frozen/g5_macro_metrics.csv` — development LOMO macro metrics
* `artifacts/frozen/g5_qualification_macro.csv` — development qualification metrics
* `artifacts/frozen/g5_uq_summary.csv` — development UQ coverage summary
* `artifacts/frozen/g7_6_full1387_status.json` — G7.6 NSGA-III C7 freeze
* `artifacts/frozen/g7_6_matched_convergence.csv` — convergence summary
* `artifacts/frozen/g7_6_300panel_ablation_summary.csv` — 300-panel ablation summary
* `artifacts/frozen/g8_external_protocol_freeze.json` — protocol declaration (mine names are explicitly part of the public protocol)
* `artifacts/frozen/g8_external_validation_freeze.json` — validation freeze (mine-level SHA256 of registries)
* `artifacts/frozen/g8_external_prediction_metrics.json` — final external metrics (the n=269 reconciled cohort)
* `artifacts/frozen/g8_external_prediction_uq_corrected.csv` — corrected external summary
* `artifacts/frozen/g8_observed_action_safety_corrected.csv` — corrected safety counts
* `artifacts/frozen/g8_dev_vs_external_corrected.csv` — corrected dev-vs-external table
* `artifacts/frozen/g8_recommendation_availability_corrected.csv` — corrected availability
* `artifacts/frozen/g8_unique_decision_verification.csv` — unique-decision verification
* `artifacts/frozen/g8_external_validation_completion_manifest.json` — G8.1 completion manifest

All of these files contain **aggregate or summary numbers only**, never per-record operational parameters or engineering outcomes.

### Level B — Derived public summary (not distributed here)

Internal paper-support tables that summarize derived quantities but are tied to record-level provenance are kept in the project's `paper_support_release/` and `results_for_paper0902/` directories. They are not part of this public release because:

* They are also distributed internally as **Level A** files (the corrected summary CSVs in Level A above), and
* Distributing the underlying record-level derivations would require distributing the per-record outcome / feature values (which is Level C).

### Level C — Restricted (NOT distributed)

The following categories are **NOT** included in this public release:

* The canonical source file `data/true_data08221157.xlsx` (SHA256 `31e6236cf8fc4792c07fc11c9ae78f1eb0b89dd34baa5f855707cc2026912492`).
* The frozen CSV `02_data/frozen/real_data_analysis_v1.csv` (SHA256 `f46e6794eeb1357788d892528fe1bd2ba26df54d45c17fe8ddb7625ab41481cc`).
* The post-unblind SC-M01 outcome parquet (any file containing `post_lugeon` or `qualified` per record).
* The historical-action prediction parquet/CSV (per-record engineering operating parameters).
* The per-record cohort ledger (contains `record_id`, `mine_id`, and per-record boolean cohort flags).
* The per-record tree-representatives CSV (contains engineering operating parameters `WCR, GP_MPa, grout_take_L_per_m, grouting_time_h` per record).
* The per-record development LOMO out-of-fold predictions (record-level residuals and predictions).
* Any file containing borehole IDs, project IDs, or per-record engineering thresholds from real field data.

These files are kept in the internal analysis workspace and are governed by the project's data-confidentiality policy. The framework's protocol freeze explicitly enumerates the **allowed external input columns** (`configs/external_validation.yaml` → `allowed_external_input_columns`) and the **forbidden outcome columns** (`forbidden_outcome_columns`); the same columns are stripped or hashed before any public-facing artifact is generated.

---

## 2. Synthetic example dataset

A small **synthetic demonstration dataset** is provided at `data/example/synthetic_demo.csv`. It contains 200 rows × 15 columns with:

* the seven final predictors (`WCR, GP_MPa, grout_take_L_per_m, grouting_time_h, depth_m, section_length_m, pre_lugeon`)
* a synthetic `post_lugeon` outcome (drawn from a synthetic regression, not from any real measurement)
* synthetic engineering bounds and a synthetic per-record acceptance threshold.

**Important:** the synthetic dataset is explicitly **NOT** derived from the real field dataset. It exists **only for software smoke testing** and does not reproduce the manuscript numerical results. Anyone running the pipeline against this synthetic dataset will obtain different numerical results from those reported in the manuscript — that is expected.

---

## 3. Running the pipeline against your own data

If you have access to a comparable multi-site grouting dataset, the framework can be run end-to-end by:

1. placing your data file at `data/true_data08221157.xlsx` (or modifying `_common.py:CANONICAL_SOURCE`),
2. placing the frozen CSV at `02_data/frozen/real_data_analysis_v1.csv` (or modifying `_common.py:FROZEN_CSV`),
3. running the scripts in the order documented in the **Reproducing the workflow** section of `README.md`,
4. comparing your output to the manuscript headline metrics via `scripts/verify_frozen_results.py` (only meaningful if you have the same real field dataset).

The framework's design assumes:

* a single canonical source file containing both raw and derived columns,
* a frozen CSV at the start of analysis-ready processing,
* record-specific acceptance thresholds (no universal threshold),
* per-mine engineering bounds,
* a single untouched external mine (declared in `configs/external_validation.yaml`).

Any deviation from these assumptions requires editing the relevant frozen protocol manifests and re-running the corresponding gate.

---

## 4. Reproducibility tiers

The PIS-CMOO public release supports the following reproducibility tiers:

| Tier | description | status |
|---|---|---|
| Computational reproducibility | the code in `src/` and `scripts/` is auditable and the verification script reads the distributed frozen artifacts | **Yes** (see `scripts/verify_frozen_results.py`) |
| Derived / aggregate reproducibility | the frozen aggregate artifacts in `artifacts/frozen/` are exactly what was used to produce the manuscript's headline numbers | **Yes** (sha256-verified, no drift) |
| Full raw-data reproducibility | re-running the pipeline against the real field dataset to reproduce every per-record output | **No** — the real field dataset is not distributed |
| Synthetic-data smoke reproducibility | running the pipeline against `data/example/synthetic_demo.csv` to confirm the code runs end-to-end | **Yes**, but numerical results will differ from the manuscript by design |

For the SC-M01 final classification metrics specifically, this release explicitly incorporates the **RESOLVED** reconciliation from `docs/evidence_manifest.md` (FINAL STATUS: RESOLVED) and from the internal SCM01 reconciliation report. The verification script checks those values to 10-decimal precision against the frozen metrics JSON.