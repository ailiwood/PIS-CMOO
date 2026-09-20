# Method overview

> Brief description of the PIS-CMOO scientific workflow. See the associated manuscript for the full methodology.

## 1. Inputs

Each grouting section (a "record") is described by:

* seven **final predictors** — `WCR, GP_MPa, grout_take_L_per_m, grouting_time_h, depth_m, section_length_m, pre_lugeon` (numerical);
* a per-record `acceptance_threshold_lu` (the contractually specified leak-acceptance threshold in Lu);
* per-record engineering bounds: `min_WCR, max_WCR, allowable_GP_MPa, max_grout_take_L_per_m, max_grouting_time_h`;
* optional context: `mine_id, project_id, borehole_id, mine_type, lithology, karst_class, fracture_class, hydrogeological_regime, project_stage, groundwater_pressure_MPa` (context only; not added to the predictor set).

The prediction target is `log1p(post_lugeon)`. The qualification label is `post_lugeon <= acceptance_threshold_lu`.

## 2. Prediction (TabDPT primary, CatBoost secondary)

* TabDPT: tabular-feature regressor trained on the CORE_CC route with six-fold leave-one-mine-out across the six development mines (`GX-M01, GX-M02, GZ-M01, GZ-M02, YN-M01, YN-M02`).
* CatBoost: gradient-boosted regression trees (RMSE loss, 1000 iterations, depth 6, learning rate 0.05, random seed 20260822) trained on the EXTENDED_FOLD_IMPUTED route as a robust alternate.
* Final selection: TabDPT (champion) by G5 macro MAE among UQ-floor-passing configurations; CatBoost (robust alternate). Selection protocol recorded in `artifacts/frozen/g5_selection_freeze.json`.

## 3. Uncertainty quantification

* One-sided conformal upper bound from mine-balanced, inner-LOMO, cross-site residual calibration.
* Residual computed on `log1p(post_lugeon) - pred_log1p` scale; reported coverage uses `expm1(UCB_log1p)` vs `post_lugeon` on natural-Lu scale.
* Primary alpha: 0.10 (90% coverage); sensitivity 0.05 (95%) and 0.15 (85%) also reported.
* Empirical finite-sample "higher" right-continuous quantile; mine-balanced residual distribution.

## 4. Optimization (NSGA-III C7, three objectives)

* Three objectives: `f1 = predicted post_lugeon`, `f2 = grout_take_L_per_m`, `f3 = grouting_time_h`.
* Constraints:
  * **engineering bounds** (per-record `min/max` from the data);
  * **conformal UCB** (one-sided; never an objective) — `expm1(pred_log1p + q_alpha) <= acceptance_threshold_lu`;
  * **joint support** (development-mined nearest-neighbor distance below threshold).
* Optimizer: NSGA-III, population 40, n_generations 25, total budget 1000; five seeds (20260822-20260826).
* One decision per context, three roles: `conservative` (min f1), `resource_saving` (min f2), `balanced` (closest to normalized centroid).

## 5. Cross-model verification and abstention

* Independent TabDPT deployment replica is audited against the same context; its UCB must be safe.
* Level-B = `tree_ucb_safe AND tabdpt_audit_ucb_safe AND engineering_pass AND support_pass`.
* Abstention: tree-safe / TabDPT-unsafe ⇒ `MODEL_DISAGREEMENT_NO_CROSS_MODEL_VERIFIED_RECOMMENDATION` (no substitution).

## 6. External validation (single untouched mine)

* External mine: `SC-M01` (declared in `configs/external_validation.yaml`).
* Cohort: 275 raw → 269 prediction-evaluable → 259 optimization-eligible.
* Six prediction-evaluable exclusions are records with a missing 7-feature predictor; all six have `qualified=False` in the frozen outcome parquet.
* No protocol modification after unblind. Subsequent changes restricted to reporting / provenance / denominator / scale correction (per `artifacts/frozen/g8_external_validation_completion_manifest.json`).
* All classification metrics reported in the manuscript use the **n=269 prediction-evaluable cohort** after the G8.1 reconciliation (FINAL STATUS: RESOLVED).

## 7. Reading the headline metrics

* Regression: `MAE / RMSE / R²` on natural-Lu scale on n=269.
* UQ coverage: empirical coverage of the 90% / 95% conformal UCB on n=269.
* Classification (thresholded): `MCC / balanced accuracy / sensitivity / specificity` on the 2x2 confusion matrix computed on n=269 records with finite predictions.
* Ranking (continuous): `AUROC / PR-AUC` of the qualification margin on n=269.
* Observed-action safety: counted on the n=275 historical-action cohort with the UCB-safe numerator/denominator masked to n=269 prediction-evaluable records (8/269 TabDPT, 7/269 CatBoost).
* Decision availability: tree-safe / Level-B / disagreement-only / no-tree-safe counted on the n=259 optimization-eligible cohort.