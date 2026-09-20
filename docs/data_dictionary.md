# Data Dictionary

This document describes every column referenced by the public PIS-CMOO release. The columns come from three sources:

* the **canonical source xlsx** (Level C, not distributed — see `data/README.md`),
* the **frozen CSV** derived from the canonical source (Level C, not distributed),
* the **synthetic example dataset** at `data/example/synthetic_demo.csv` (Level A, distributed).

## 1. Predictor columns (the seven final predictors)

Used by **every** model in the pipeline. These are the only columns that enter the model.

| column | dtype | unit | description |
|---|---|---|---|
| `WCR` | float | ratio (water-cement ratio) | grout mix water-to-cement ratio |
| `GP_MPa` | float | MPa | grouting pressure |
| `grout_take_L_per_m` | float | L/m | grout take per metre of section |
| `grouting_time_h` | float | h | grouting duration |
| `depth_m` | float | m | borehole section depth |
| `section_length_m` | float | m | grouted section length |
| `pre_lugeon` | float | Lu | pre-grouting Lugeon value |

## 2. Outcome / qualification columns

| column | dtype | unit | description |
|---|---|---|---|
| `post_lugeon` | float | Lu | post-grouting Lugeon value |
| `qualified` | bool | n/a | `True` if `post_lugeon <= acceptance_threshold_lu` |
| `acceptance_threshold_lu` | float | Lu | per-record acceptance threshold (no universal value) |

## 3. Engineering-bound columns

| column | dtype | unit | description |
|---|---|---|---|
| `min_WCR` | float | ratio | lower engineering bound on `WCR` |
| `max_WCR` | float | ratio | upper engineering bound on `WCR` |
| `allowable_GP_MPa` | float | MPa | upper engineering bound on `GP_MPa` |
| `max_grout_take_L_per_m` | float | L/m | upper engineering bound on `grout_take_L_per_m` |
| `max_grouting_time_h` | float | h | upper engineering bound on `grouting_time_h` |

## 4. Context columns (NOT used as predictors)

These are descriptive context columns that are **not** added to the predictor set. They appear only for record-traceability and for tabular-feature usage inside TabDPT. They are listed here for completeness.

| column | dtype | description |
|---|---|---|
| `province` | str | province code (Level C — not in public artifacts) |
| `mine_id` | str | mine identifier |
| `project_id` | str | project identifier |
| `borehole_id` | str | borehole identifier |
| `record_id` | str | unique record identifier |
| `mine_type` | str | mine type |
| `lithology` | str | lithology code |
| `karst_class` | int | karst class |
| `fracture_class` | int | fracture class |
| `hydrogeological_regime` | str | hydrogeological regime |
| `project_stage` | str | project stage |
| `groundwater_pressure_MPa` | float | groundwater pressure (Level C in pre-unblind contexts) |

## 5. Derived outcome columns (informational)

| column | dtype | description |
|---|---|---|
| `delta_lugeon` | float | `pre_lugeon - post_lugeon` |
| `relative_reduction` | float | `delta_lugeon / pre_lugeon` |
| `log_reduction` | float | `log(pre_lugeon) - log(post_lugeon)` |

## 6. Frozen-artifact-only columns

The frozen aggregate artifacts do not contain per-record data. The columns they DO contain are self-documenting:

* `g5_macro_metrics.csv` → per-route / per-model / per-target_variant summary metrics
* `g5_uq_summary.csv` → coverage and width summary
* `g7_6_full1387_status.json` → n_contexts, n_safe_contexts, per-mine safe counts, n_solution_rows
* `g8_external_prediction_metrics.json` → n_records_with_outcome, regression metrics, qualification classification metrics, UQ coverage, observed-action safety
* `g8_observed_action_safety_corrected.csv` → n_evaluable, n_not_evaluable, point_safe_n, ucb90_safe_n, etc.
* `g8_recommendation_availability_corrected.csv` → metric, numerator, denominator, rate

See each artifact's first row for the full column list.

## 7. Synthetic example dataset

`data/example/synthetic_demo.csv` contains:

| column | dtype | notes |
|---|---|---|
| `record_id` | str | formatted `SYN-NNNN` |
| The 7 predictors + outcome + acceptance threshold + 5 engineering bounds | as above | synthetic values; **NOT** derived from real data |

The synthetic dataset is explicitly **not** a stand-in for the real field dataset and must not be used to reproduce the manuscript's numerical results.