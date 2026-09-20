# Evidence Manifest

This manifest documents every **publicly distributed** artifact in this release, including its scientific role, cohort, scale, SHA256, and the original (internal) project path. The manifest is intended to be sufficient to re-verify the manuscript's headline numbers without access to the internal project workspace.

| artifact_id | file (public path) | role | cohort | scale | SHA256 | original project path |
|---|---|---|---|---|---|---|
| g5-sel-freeze | `artifacts/frozen/g5_selection_freeze.json` | G5 model-selection freeze | development LOMO | n/a | computed at release | `010_true_data_analysis/04_prediction/g5_selection_freeze.json` |
| g5-macro | `artifacts/frozen/g5_macro_metrics.csv` | development LOMO macro metrics | development LOMO | natural Lu + log1p | computed at release | `010_true_data_analysis/04_prediction/g5_macro_metrics.csv` |
| g5-qual | `artifacts/frozen/g5_qualification_macro.csv` | development qualification metrics | development LOMO | thresholded | computed at release | `010_true_data_analysis/04_prediction/g5_qualification_macro.csv` |
| g5-uq | `artifacts/frozen/g5_uq_summary.csv` | development UQ coverage summary | development LOMO | 90/95 coverage | computed at release | `010_true_data_analysis/04_prediction/g5_uq_summary.csv` |
| g76-status | `artifacts/frozen/g7_6_full1387_status.json` | G7.6 NSGA-III C7 freeze | development n=1387 | Pareto | computed at release | `010_true_data_analysis/07_optimization/g7_6/g7_6_full1387_status.json` |
| g76-conv | `artifacts/frozen/g7_6_matched_convergence.csv` | G7.6 budget-matched convergence | development n=1387 | budget | computed at release | `010_true_data_analysis/07_optimization/g7_6/g7_6_matched_convergence.csv` |
| g76-ablation | `artifacts/frozen/g7_6_300panel_ablation_summary.csv` | G7.6 300-panel ablation | development (300 panels) | budget ablation | computed at release | `010_true_data_analysis/07_optimization/g7_6/g7_6_300panel_ablation_summary.csv` |
| g8-protocol | `artifacts/frozen/g8_external_protocol_freeze.json` | G8 protocol declaration (pre-unblind) | n/a | protocol | computed at release | `010_true_data_analysis/08_validation/g8/freeze/g8_external_protocol_freeze.json` |
| g8-validate | `artifacts/frozen/g8_external_validation_freeze.json` | G8 validation freeze (post-unblind) | n=275 raw | registry SHA256 | computed at release | `010_true_data_analysis/08_validation/g8/freeze/g8_external_validation_freeze.json` |
| g8-complete | `artifacts/frozen/g8_external_validation_completion_manifest.json` | G8.1 completion manifest | 275/269/259 | n/a | computed at release | `010_true_data_analysis/12_handoffs/g8_external_validation_completion_manifest.json` |
| g8-pred-metrics | `artifacts/frozen/g8_external_prediction_metrics.json` | FINAL external metrics (post-reconciliation) | external n=269 prediction-evaluable | natural Lu + log1p | computed at release | `010_true_data_analysis/08_validation/g8/g8_external_prediction_metrics.json` |
| g8-pred-uq-corr | `artifacts/frozen/g8_external_prediction_uq_corrected.csv` | G8.1 corrected summary table | external n=269 | natural Lu + log1p | computed at release | `010_true_data_analysis/10_tables/g8_1/g8_external_prediction_uq_corrected.csv` |
| g8-safety-corr | `artifacts/frozen/g8_observed_action_safety_corrected.csv` | G8.1 corrected safety counts | external (n_evaluable=269; n_observed_actions=275) | mixed | computed at release | `010_true_data_analysis/10_tables/g8_1/g8_observed_action_safety_corrected.csv` |
| g8-dev-ext-corr | `artifacts/frozen/g8_dev_vs_external_corrected.csv` | G8.1 corrected dev-vs-external | dev LOMO vs external n=269 | natural Lu + log1p | computed at release | `010_true_data_analysis/10_tables/g8_1/g8_dev_vs_external_corrected.csv` |
| g8-avail-corr | `artifacts/frozen/g8_recommendation_availability_corrected.csv` | G8.1 corrected availability | external n=259 optimization-eligible | rates | computed at release | `010_true_data_analysis/10_tables/g8_1/g8_recommendation_availability_corrected.csv` |
| g8-unique-dec | `artifacts/frozen/g8_unique_decision_verification.csv` | G8.1 unique-decision verification | dev + ext | count | computed at release | `010_true_data_analysis/10_tables/g8_1/g8_unique_decision_verification.csv` |

(All SHA256 values above are computed and recorded in `artifacts/frozen/SHA256_MANIFEST.csv` at release time.)

## SC-M01 final classification reconciliation — RESOLVED

A complete record-level deterministic recomputation has been performed against the frozen artifacts and recorded in the internal `SCM01_external_classification_metrics_FINAL_RECONCILIATION.md` (FINAL STATUS: RESOLVED). The reconciled authoritative values for **n = 269 prediction-evaluable** are:

| metric | TabDPT (n=269) | CatBoost (n=269) |
|---|---|---|
| Sensitivity | 0.6739 | 0.5978 |
| Specificity | 0.9322 | 0.9492 |
| Balanced Accuracy | 0.8031 | 0.7735 |
| MCC | **0.6439** | **0.6094** |
| AUROC (qual margin) | 0.9154 | 0.9086 |
| PR-AUC (qual margin) | 0.8589 | 0.8482 |
| TP / FP / TN / FN | 62 / 12 / 165 / 30 | 55 / 9 / 168 / 37 |

Manuscript-rounded values (3-decimal): TabDPT MCC=0.644, balanced_accuracy=0.803, sensitivity=0.674, specificity=0.932, AUROC=0.915, PR-AUC=0.859; CatBoost MCC=0.609, balanced_accuracy=0.773, sensitivity=0.598, specificity=0.949, AUROC=0.909, PR-AUC=0.848.

## Data origin

* `data/true_data08221157.xlsx` (SHA256 `31e6236cf8fc4792c07fc11c9ae78f1eb0b89dd34baa5f855707cc2026912492`) and `02_data/frozen/real_data_analysis_v1.csv` (SHA256 `f46e6794eeb1357788d892528fe1bd2ba26df54d45c17fe8ddb7625ab41481cc`) are **Level C** and are not distributed. See `data/README.md`.

## Random seeds

* Master seed: `20260822` (recorded in `src/pis_cmoo/utils/_common.py`).
* Optimizer seeds: `20260822, 20260823, 20260824, 20260825, 20260826`.