# Reproducibility notes

This document describes the levels of reproducibility supported by this public release, the seed convention, the deterministic/non-deterministic split, and the limitations of reproducing the manuscript's numerical results from this repository alone.

## 1. Reproducibility tiers

| Tier | description | supported? |
|---|---|---|
| Computational | the code in `src/` and `scripts/` is auditable and the verification script reads the distributed frozen artifacts | yes |
| Aggregate | the frozen aggregate artifacts in `artifacts/frozen/` match the manuscript's headline numbers to 10-decimal precision | yes (verified by `scripts/verify_frozen_results.py`) |
| Full raw-data | re-running the pipeline against the real field dataset to reproduce every per-record output | no — real field dataset is not distributed (Level C; see `data/README.md`) |
| Synthetic-data smoke | running the pipeline against `data/example/synthetic_demo.csv` to confirm the code runs end-to-end | yes, but the numerical results will differ from the manuscript by design |

## 2. Seed convention

* **Master seed**: `20260822` (recorded in `src/pis_cmoo/utils/_common.py:SEED_MASTER`).
* **Optimizer seeds** (NSGA-III): `20260822, 20260823, 20260824, 20260825, 20260826`.
* **CatBoost random_seed**: `20260822`.

## 3. Determinism notes

* The verification script (`scripts/verify_frozen_results.py`) is fully deterministic: it reads the frozen artifacts and compares them to hard-coded expected values.
* The frozen artifacts themselves are bit-identical to the corresponding internal artifacts (sha256 matches the internal computation).
* The seven final predictors and the log1p target scale are deterministic by design.
* The conformal calibration is deterministic given a frozen residual distribution.
* The optimization under a fixed seed is deterministic in the sense that it produces the same Pareto front on identical hardware and identical BLAS / thread settings. The published `g7_6_full1387_status.json` records the n_safe_contexts across five seeds; the headline number (576 safe contexts) is the seed-mean / seed-majority result.
* The TabDPT model may carry its own internal stochasticity (model implementation). Reproducing the exact numerical results on identical hardware requires the same TabDPT package version; cross-version re-runs may exhibit small numerical deviations.

## 4. Running the shipped verification

```bash
cd PIS-CMOO
python scripts/verify_frozen_results.py
```

Expected output:

```
================================================================
PIS-CMOO — frozen-result verification
================================================================
  PASS  g5_selection_freeze.json  (TabDPT champion, CatBoost robust-alternate, no drift)
  PASS  g5_macro_metrics.csv  (TabDPT macro R²=0.5462, worst-site R²=0.3592; CatBoost macro R²=0.5562)
  ...
================================================================
ALL FROZEN-ARTIFACT CHECKS PASS
================================================================
```

## 5. Limitations

* This release cannot reproduce the per-record engineering outputs (operating-parameter recommendations, Pareto members per context, etc.) because the per-record tree-representatives CSV and historical-action predictions are Level C (not distributed).
* The TabDPT package version is documented in `configs/prediction.yaml` but the package itself is not vendored here. Reproducing the exact TabDPT numerical results requires obtaining the same TabDPT package (or its source) outside this public release.
* The synthetic example dataset is provided only for software smoke testing. Numerical results obtained by running the pipeline against the synthetic dataset are **not** the manuscript's results.
* The pipeline assumes a single canonical source xlsx and a single frozen CSV; deviations from this convention require editing the corresponding `_common.py` paths.

## 6. SC-M01 reconciliation pointer

The SC-M01 final classification metrics (MCC / balanced accuracy / AUROC / sensitivity / specificity) reflect the RESOLVED reconciliation (FINAL STATUS: RESOLVED) of the 275-vs-269 cohort ambiguity. The reconciliation is described in detail in the internal `SCM01_external_classification_metrics_FINAL_RECONCILIATION.md` and reproduced in `docs/evidence_manifest.md`. The shipped `artifacts/frozen/g8_external_prediction_metrics.json` is the post-reconciliation authoritative source; `scripts/verify_frozen_results.py` re-checks every metric to 10-decimal precision.