# PIS-CMOO

A cross-site uncertainty-aware prediction, verification, and multi-objective decision-support framework for underground grouting.

> **Public research-code release.** This repository contains the **frozen scientific pipeline** of the PIS-CMOO framework. The framework targets prediction, calibrated uncertainty, post-hoc interpretation, and constrained multi-objective optimization of grouting operating parameters in underground construction, with a single untouched external mine held out for honest evaluation.

---

## Overview

PIS-CMOO is a research codebase that supports the following scientific workflow:

| stage | module | description |
|---|---|---|
| prediction | `src/pis_cmoo/prediction` | TabDPT (primary) and CatBoost (secondary) regression of post-grouting Lugeon value, leave-one-mine-out (LOMO) across six development mines. |
| uncertainty | `src/pis_cmoo/uncertainty` | One-sided conformal upper bound from mine-balanced, inner-LOMO, cross-site residual calibration. |
| interpretation | `src/pis_cmoo/interpretation` | SHAP attribution on the fitted tree model; direction semantics; interaction supplements. |
| optimization | `src/pis_cmoo/optimization` | Engineering-bound-aware NSGA-III (PIS C7 config); engineering-bounds constraint; conformal UCB constraint; joint-support constraint; one-decision + three roles (conservative / resource_saving / balanced) per context. |
| cross-model verification | `src/pis_cmoo/optimization` | Independent TabDPT deployment-replica audit; cross-model Level-B rule; abstention when models disagree. |
| external validation | `src/pis_cmoo/validation` | Untouched single external mine evaluation with preoutcome freeze, one-time outcome unblind, observed-action safety audit, and G8.1 reporting reconciliation. |

---

## Scientific scope

| role | model / module |
|---|---|
| Primary prediction model | **TabDPT** (regression of `log1p(post_lugeon)`; CORE_CC route; leave-one-mine-out) |
| Optimization response model | **CatBoost** (gradient-boosted regression trees, RMSE loss) |
| Cross-model verification | TabDPT deployment replica |
| Optimizer | **NSGA-III**, C7 constraint configuration, B1=1000 budget (pop=40, n_gen=25), 5 seeds |
| UQ | mine-balanced empirical residual distribution, finite-sample "higher" right-continuous quantile |

The seven final predictors used everywhere are:

```
WCR, GP_MPa, grout_take_L_per_m, grouting_time_h, depth_m, section_length_m, pre_lugeon
```

No additional geology / hydrogeology variable is automatically added to the model. `log1p(post_lugeon)` is the regression target; natural-Lu `post_lugeon <= acceptance_threshold_lu` defines qualification.

---

## Repository structure

```
PIS-CMOO/
├── README.md                  # this file
├── LICENSE_PENDING.md          # license decision pending author confirmation
├── CITATION.cff               # citation metadata
├── pyproject.toml             # Python packaging metadata
├── requirements.txt           # runtime dependencies (pinned)
├── .gitignore
├── .gitattributes
├── src/pis_cmoo/              # importable package, mirrors the frozen script layout
│   ├── prediction/            # G5 prediction pipeline (TabDPT, CatBoost, model selection)
│   ├── uncertainty/           # G5 conformal UQ
│   ├── interpretation/        # G6 SHAP / direction semantics
│   ├── optimization/          # G7.0 / G7.5 / G7.6 engineering-bound + NSGA-III
│   ├── validation/            # G8 external + G8.1 reconciliation
│   └── utils/                 # shared path / hashing / loader utilities
├── scripts/                   # top-level CLI entry points (also under src/ as modules)
├── configs/                   # YAML configs mirroring the frozen protocol
├── data/
│   ├── README.md              # data disclosure assessment (Level A / B / C)
│   ├── example/               # SYNTHETIC demonstration dataset only
│   ├── source_data/           # placeholder for Level A (not populated; see data/README)
│   └── derived/               # placeholder for Level B (not populated; see data/README)
├── artifacts/
│   └── frozen/                # frozen numerical evidence (no record-level raw data)
├── docs/                      # method overview, evidence manifest, reproducibility notes
├── tests/                     # smoke tests + frozen-result verification
├── notebooks/                 # placeholder; pipeline lives under scripts/
├── paper/                     # placeholder for paper-facing PDFs / TeX (not populated)
└── results/                   # placeholder for downstream summary outputs
```

---

## Installation

```bash
git clone https://github.com/ailiwood/PIS-CMOO.git
cd PIS-CMOO
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions used by the frozen pipeline. The TabDPT model is described in `configs/prediction.yaml` (its commit / package version was fixed at G5 freeze time).

---

## Reproducing the workflow

The full pipeline consists of pre-freeze stages (G0–G7.6) and the post-freeze external validation (G8). The shipped entry points assume the **real field dataset** is present at `data/true_data08221157.xlsx` (SHA256 `31e6236cf8fc4792c07fc11c9ae78f1eb0b89dd34baa5f855707cc2026912492`) and that the analysis workspace `010_true_data_analysis/` is present. **The real field dataset is not distributed in this public release** — see *Data availability* below.

The scripts are organized in the same numbered sequence as the original frozen analysis. They can be invoked directly:

```bash
python scripts/15_g5_pipeline.py        # G5 prediction (TabDPT/CatBoost LOMO)
python scripts/17_g5_uq.py              # G5 cross-site residual conformal UQ
python scripts/27_g6_interpretation.py  # G6 SHAP interpretation
python scripts/40_g7_0_eligibility_constraints.py  # G7.0 engineering bounds
python scripts/99_g7_5_optimization.py  # G7.5 NSGA-III C7 optimization
python scripts/209_g8_preoutcome_freeze.py  # G8 pre-unblind hash freeze
python scripts/210_g8_unblind.py        # G8 one-time SC-M01 outcome unblind
python scripts/211_g8_external_evaluation.py  # G8 external metrics
python scripts/215_g8_1_reconciliation.py     # G8.1 reporting reconciliation
```

Each script reads from the frozen analysis workspace at `010_true_data_analysis/` (relative to the parent project root) and writes its outputs to the corresponding gate subdirectory. None of these scripts requires network access or external credentials.

---

## Frozen-result verification

Run the bundled verification script to confirm the published numerical evidence is intact and matches the protocol-frozen numbers:

```bash
python scripts/verify_frozen_results.py
```

This script performs a **read-only** check of `artifacts/frozen/*` against the manuscript headline metrics (see `docs/evidence_manifest.md`). It does **not** train, calibrate, or re-derive any model.

---

## Data availability

**The original field dataset is NOT distributed with this public release.** The data was collected under commercial confidentiality and contains operationally sensitive parameters. The framework was developed and frozen against:

- one canonical source file (SHA256 `31e6236cf8fc4792c07fc11c9ae78f1eb0b89dd34baa5f855707cc2026912492`), and
- one frozen CSV (SHA256 `f46e6794eeb1357788d892528fe1bd2ba26df54d45c17fe8ddb7625ab41481cc`)

The aggregate statistics in `artifacts/frozen/` (model-selection freeze, G5 macro metrics, G7.6 optimization status, G8 external metrics, G8.1 corrected tables) **are** distributed because they are non-identifiable aggregate measurements.

A small synthetic dataset (`data/example/synthetic_demo.csv`) is included **only for software smoke testing** and does not reproduce the manuscript numerical results.

See `data/README.md` for the full data-disclosure assessment (Level A / B / C).

---

## Model roles

| model | role in pipeline |
|---|---|
| **TabDPT** | Primary prediction model (regression of `log1p(post_lugeon)`) AND independent cross-model verification model (deployment replica). |
| **CatBoost** | Optimization response model used inside NSGA-III objective `f1 = predicted post-grouting Lu`. |
| **NSGA-III (C7)** | Primary multi-objective optimizer. Three objectives: `f1 = predicted post_lugeon`, `f2 = grout_take_L_per_m`, `f3 = grouting_time_h`. |

The conformal upper bound is a **hard constraint**, never an objective.

---

## Stochasticity and seeds

- Master seed: `20260822` (recorded in `src/pis_cmoo/utils/_common.py`).
- Optimizer seeds: `20260822, 20260823, 20260824, 20260825, 20260826` (5 seeds, see `configs/optimization.yaml`).
- Conformal calibration is deterministic given a frozen residual distribution.
- TabDPT may carry its own internal stochasticity (model implementation). Numerical results on identical hardware should match to within the manuscript's stated precision; cross-platform / cross-version re-runs may exhibit small deviations in the third decimal place.

---

## Citation

See `CITATION.cff`. The associated manuscript is in submission; a permanent DOI will be added after acceptance.

---

## License

`LICENSE_PENDING.md` describes the license status. Code and data licenses are documented separately in `data/README.md`.

---

## Provenance and freeze hierarchy (top-level)

This repository preserves the **post-freeze** scientific outputs from the G0–G8 / G8.1 protocol. Pre-freeze Gate chronology is documented in the project's internal handoffs but is **not** part of this public release.

| Gate | role | public artifact |
|---|---|---|
| G5 | prediction model selection | `artifacts/frozen/g5_*` |
| G5 | cross-site residual conformal UQ | `artifacts/frozen/g5_uq_summary.csv` |
| G6 | interpretation | internal scripts (`src/pis_cmoo/interpretation/`) |
| G7.0 / G7.5 / G7.6 | engineering-bound + NSGA-III C7 | `artifacts/frozen/g7_6_*` |
| G8 / G8.1 | external validation + reconciliation | `artifacts/frozen/g8_*` |
| G8.1 | completion manifest | `artifacts/frozen/g8_external_validation_completion_manifest.json` |

For the full evidence manifest including SHA256 of every artifact, see `docs/evidence_manifest.md`.

---

## Contact

Issues and questions are welcome via the GitHub issue tracker. The associated manuscript describes the methodology and frozen experimental protocol in detail.