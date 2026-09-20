# Frozen artifacts — public release

This directory contains **non-identifiable aggregate numerical evidence** from the PIS-CMOO frozen pipeline (G5 / G7.6 / G8 / G8.1). All files are Level A in the data-disclosure scheme documented in `data/README.md`.

## Inventory

| file | stage | cohort | scale | what it proves |
|---|---|---|---|---|
| `g5_selection_freeze.json` | G5 | development (6 mines) | natural Lu + log1p | TabDPT was selected as the champion regression model; CatBoost was the robust alternate. Selection protocol recorded. |
| `g5_macro_metrics.csv` | G5 | development LOMO | natural Lu + log1p | TabDPT macro R²=0.5462, worst-site R²=0.3592; CatBoost macro R²=0.5562. |
| `g5_qualification_macro.csv` | G5 | development LOMO | thresholded | development-side qualification classification metrics. |
| `g5_uq_summary.csv` | G5 | development LOMO | 90 / 95 coverage | cross-site residual conformal UQ coverage on development. |
| `g7_6_full1387_status.json` | G7.6 | development (n=1387) | Pareto front | NSGA-III C7 freeze: n_safe_contexts=576/1387=41.5%; per-mine safe counts. |
| `g7_6_matched_convergence.csv` | G7.6 | development (n=1387) | budget-matched | budget-matched convergence ablation. |
| `g7_6_300panel_ablation_summary.csv` | G7.6 | development (300 panels) | budget ablation | 300-panel ablation summary. |
| `g8_external_protocol_freeze.json` | G8 | pre-unblind | protocol declaration | training mines (GX/GZ/YN six), forbidden outcome columns, NSGA-III C7, abstention rule, no-adaptation rule. |
| `g8_external_validation_freeze.json` | G8 | post-unblind (n=275 raw) | registry SHA256 | unblind timestamp; CSV/JSON registry SHA256. |
| `g8_external_validation_completion_manifest.json` | G8.1 | cohort freeze | n/a | cohort=275/269/259; protocol CLOSED; subsequent changes restricted to reporting/provenance/denominator/scale correction. |
| `g8_external_prediction_metrics.json` | G8 | external n=269 prediction-evaluable | natural Lu + log1p | **FINAL** external regression, qualification classification, UQ, observed-action safety. All classification metrics reflect the RESOLVED SCM01 reconciliation. |
| `g8_external_prediction_uq_corrected.csv` | G8.1 | external n=269 | natural Lu + log1p | corrected summary table. |
| `g8_observed_action_safety_corrected.csv` | G8.1 | external n=269 (UCB mask); n=275 (historical actions) | mixed | corrected observed-action safety (UCB-safe 8/269 TabDPT, 7/269 CatBoost). |
| `g8_dev_vs_external_corrected.csv` | G8.1 | development LOMO vs external | natural Lu + log1p | corrected dev-vs-external side-by-side. |
| `g8_recommendation_availability_corrected.csv` | G8.1 | external n=259 optimization-eligible | rates | tree-safe 41/259, Level-B 11/259, disagreement-only 30, no-tree-safe 218. |
| `g8_unique_decision_verification.csv` | G8.1 | dev + ext | count | unique-decision verification. |

## Headline numbers (manuscript-facing)

### Development LOMO

| metric | TabDPT | CatBoost |
|---|---|---|
| macro R² (natural Lu) | 0.546 | 0.558 |
| worst-site R² | 0.359 | — |
| 90% coverage (development) | 0.901 | 0.904 |
| tree-safe (n_safe / 1387) | — | 576 / 1387 = 41.5% |
| cross-model verified (Level-B) | — | 256 / 1387 = 18.5% |

### External SC-M01 (n=269 prediction-evaluable)

| metric | TabDPT | CatBoost |
|---|---|---|
| MAE (Lu) | 0.820 | 0.931 |
| RMSE (Lu) | 1.462 | 1.703 |
| R² (Lu) | 0.460 | 0.267 |
| 90% coverage | 0.9517 | 0.9591 |
| 95% coverage | 0.9740 | 0.9777 |
| MCC | 0.644 | 0.609 |
| Balanced accuracy | 0.803 | 0.773 |
| Sensitivity | 0.674 | 0.598 |
| Specificity | 0.932 | 0.949 |
| AUROC (qual margin) | 0.915 | 0.909 |
| PR-AUC (qual margin) | 0.859 | 0.848 |
| TP / FP / TN / FN | 62 / 12 / 165 / 30 | 55 / 9 / 168 / 37 |
| UCB-safe rate (n=269) | 8/269 ≈ 3.0% | 7/269 ≈ 2.6% |
| UCB false-safe (UCB-safe but unqualified) | 0 | 0 |
| Conservative false-unsafe (UCB-unsafe but qualified) | 84 | 85 |

### External decision availability (n=259 optimization-eligible)

| metric | value |
|---|---|
| tree-safe | 41 / 259 = 15.8% |
| Level-B (cross-model verified) | 11 / 259 = 4.3% |
| Disagreement-only | 30 |
| No-tree-safe | 218 |

## SC-M01 classification reconciliation

The SC-M01 classification metrics (MCC / balanced accuracy / AUROC / sensitivity / specificity) have a known cohort ambiguity (275 vs 269). The reconciliation is documented in the project's internal `SCM01_external_classification_metrics_FINAL_RECONCILIATION.md` (FINAL STATUS: RESOLVED) and reproduced in `docs/evidence_manifest.md`. The values distributed in `artifacts/frozen/g8_external_prediction_metrics.json` are the **post-reconciliation authoritative values** (n=269 prediction-evaluable).

The reconciliation audit confirmed:

* the original `211_g8_external_evaluation.py` script did not apply the `~np.isnan(p)` mask to the thresholded confusion matrix (the bug),
* the G8.1 reconciliation corrected this and re-derived the n=269 CM,
* the AUROC and PR-AUC were already correctly computed on n=269 in both the original and corrected files (the AUROC path correctly applied the mask),
* the manuscript TeX (MCC=0.644 for TabDPT; 0.609 for CatBoost) is consistent with the n=269 cohort.

`scripts/verify_frozen_results.py` re-checks every classification metric to 10-decimal precision against the frozen metrics JSON.