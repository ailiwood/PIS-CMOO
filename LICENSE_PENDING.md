# License — Pending Author Confirmation

> **Status:** *License decision pending author confirmation.*

This repository is a public research-code release accompanying the manuscript *PIS-CMOO* (under review / in submission). The license for the **code** in this repository, and a separate license for the **synthetic example data** in `data/example/`, will be set by the repository owner once the manuscript's open-science policy is finalized.

Until then, the following applies:

* **Source code** in `src/`, `scripts/`, `tests/`, and `configs/`: all rights reserved. No reuse, redistribution, or modification is granted.
* **Synthetic example data** in `data/example/synthetic_demo.csv`: explicitly NOT derived from the real field dataset; see `data/README.md` for the data-disclosure assessment.
* **Aggregate frozen artifacts** in `artifacts/frozen/`: all rights reserved.
* **Documentation** in `README.md`, `docs/`: all rights reserved.

The most likely candidates once the author decides are **MIT License** (permissive, common for scientific Python) or **BSD-3-Clause** (also permissive, attribution + non-endorsement). The repository does not vendor any third-party source code; third-party dependencies (CatBoost, pymoo, scikit-learn, etc.) are referenced via `requirements.txt` and remain under their original licenses.

If you intend to reuse any part of this repository before a license is chosen, please contact the repository owner through the GitHub issue tracker.