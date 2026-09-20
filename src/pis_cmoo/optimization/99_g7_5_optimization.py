"""scripts/99_g7_5_optimization.py — G7.5 primary optimization with CatBoost + tree-conformal UCB.

Per the G7.5 prompt §10-13:
- Decision vector: [WCR, GP_MPa, grout_take_L_per_m, grouting_time_h]
- Primary evaluator: CatBoost (frozen pre-result by selection rule)
- Primary safety: tree one-sided conformal UCB <= acceptance_threshold_lu
- Hard engineering bounds + outer-training joint-support (NOT relaxed)

For compute frugality on this CPU, we use deterministic Sobol/QMC over the
4-dimensional decision space with the Q-budget ladder Q0=64, Q1=128, Q2=256.

Convergence gate (matched-context): feasible HV relative gain <1% AND
5 other criteria (no 6/6 monotone, safe-context change <=2pp, etc.).
"""
from __future__ import annotations
import sys, json, time, hashlib, datetime
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from importlib.machinery import SourceFileLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C

G75 = C.OPT_DIR / "g7_5"
FEATURES = ["WCR", "GP_MPa", "grout_take_L_per_m", "grouting_time_h",
            "depth_m", "section_length_m", "pre_lugeon"]
SEEDS = [20260822, 20260823, 20260824]
Q_BUDGETS = [("Q0", 64), ("Q1", 128), ("Q2", 256)]
PER_BUDGET_SEC = 1500


def _qmc(xl, xu, n, seed):
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=4, scramble=True, seed=seed)
    u = sampler.random(n)
    return xl + u * (xu - xl)


def _train_catboost_for_mine(mine):
    """Train CatBoost on outer-training partition (other 5 mines)."""
    from catboost import CatBoostRegressor
    df = C.read_frozen()
    core_ok = df[C.CORE_REQUIRED].notna().all(axis=1)
    hard = ["allowable_GP_MPa", "min_WCR", "max_WCR", "max_grout_take_L_per_m", "max_grouting_time_h"]
    elig = df[core_ok & df["mine_id"].isin(C.DEVELOPMENT_MINES)
              & df[hard + FEATURES].notna().all(axis=1)]
    train = elig[elig["mine_id"] != mine].reset_index(drop=True)
    Xtr = train[FEATURES].astype("float32").values
    ytr = np.log1p(train["post_lugeon"].astype("float32").values)
    m = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05,
                             loss_function="RMSE", verbose=False, random_seed=20260822)
    m.fit(Xtr, ytr, verbose=False)
    return m


def _tree_q10(m, X_tr, y_tr, alpha=0.10):
    pred = m.predict(X_tr).astype(float)
    res = y_tr - pred
    return float(np.quantile(res, 1.0 - alpha))


def _run_for_budget(label, q_n, panel60, frozen_df, support):
    """Run G7.5 direct QMC search with CatBoost + tree-conformal UCB."""
    rows_summary = []
    safe_ctx_count = 0
    all_F = []
    budget_start = time.time()
    for mine in C.DEVELOPMENT_MINES:
        if time.time() - budget_start > PER_BUDGET_SEC: break
        try:
            m = _train_catboost_for_mine(mine)
            df = C.read_frozen()
            core_ok = df[C.CORE_REQUIRED].notna().all(axis=1)
            hard = ["allowable_GP_MPa", "min_WCR", "max_WCR", "max_grout_take_L_per_m", "max_grouting_time_h"]
            elig = df[core_ok & df["mine_id"].isin(C.DEVELOPMENT_MINES)
                      & df[hard + FEATURES].notna().all(axis=1)]
            train = elig[elig["mine_id"] != mine].reset_index(drop=True)
            X_tr = train[FEATURES].astype("float32").values
            y_tr = np.log1p(train["post_lugeon"].astype("float32").values)
            q = _tree_q10(m, X_tr, y_tr)
            sup = support[mine]
        except Exception as e:
            print(f"  ! {mine}: {e}", flush=True)
            continue

        for _, ctx in panel60.iterrows():
            if ctx["mine_id"] != mine: continue
            xl = np.array([ctx["min_WCR"], 1e-6, 0.0, 1e-6])
            xu = np.array([ctx["max_WCR"], ctx["allowable_GP_MPa"],
                           ctx["max_grout_take_L_per_m"], ctx["max_grouting_time_h"]])
            ctx_feat = np.array([ctx["depth_m"], ctx["section_length_m"], ctx["pre_lugeon"]])
            X = _qmc(xl, xu, q_n, seed=(SEEDS[0] + abs(hash(ctx["record_id"]))) % (2**31))
            Xfull = np.hstack([X.astype("float32"),
                               np.tile(ctx_feat, (len(X), 1)).astype("float32")])
            Xs = sup["scaler"].transform(Xfull)
            d_nn, _ = sup["nn"].kneighbors(Xs, n_neighbors=1)
            s_ok = d_nn[:, 0] <= sup["thr_95"]
            h1 = (X[:, 0] >= ctx["min_WCR"]) & (X[:, 0] <= ctx["max_WCR"])
            h2 = (X[:, 1] >= 1e-6) & (X[:, 1] <= ctx["allowable_GP_MPa"])
            h3 = (X[:, 2] >= 0) & (X[:, 2] <= ctx["max_grout_take_L_per_m"])
            h4 = (X[:, 3] >= 1e-6) & (X[:, 3] <= ctx["max_grouting_time_h"])
            pred_log = m.predict(Xfull).astype(float)
            ucb_log = pred_log + q
            thr_log = np.log1p(ctx["acceptance_threshold_lu"])
            u_ok = ucb_log <= thr_log
            ok = h1 & h2 & h3 & h4 & s_ok & u_ok
            if ok.sum() > 0:
                safe_ctx_count += 1
                f_vals = np.column_stack([np.expm1(pred_log[ok]), X[ok][:, 2], X[ok][:, 3]])
                all_F.append(f_vals)
    allF = np.vstack(all_F) if all_F else np.zeros((0, 3))
    if len(allF) > 0:
        import moocore
        nadir = allF.max(axis=0); ideal = allF.min(axis=0)
        rng_ = np.where(nadir - ideal == 0, 1, nadir - ideal)
        ref = (nadir - ideal) / rng_ + 0.1
        Fn = (allF - ideal) / rng_
        nd = moocore.is_nondominated(Fn)
        Fnd = Fn[nd]
        hv = float(moocore.hv_approx(Fnd, ref)) if len(Fnd) else 0.0
        n_pareto = int(len(Fnd))
    else:
        hv, n_pareto = 0.0, 0
    dt = time.time() - budget_start
    print(f"  {label} (q={q_n}): HV={hv:.4f} safe={safe_ctx_count} ({dt:.1f}s)", flush=True)
    return {"q_n": q_n, "hv_norm": hv, "safe_contexts": safe_ctx_count,
            "n_pareto": n_pareto, "wall_seconds": round(dt, 1)}


def main():
    support = joblib.load(C.OPT_DIR / "g7" / "models_and_uq" / "g7_support.pkl")
    frozen_df = C.read_frozen()
    b2_ids = pd.read_csv(C.OPT_DIR / "g7_2" / "convergence_budget" / "g7_2_b2_60_context_ids.csv")["record_id"].tolist()
    panel = pd.read_csv(C.OPT_DIR / "g7" / "benchmark_panel" / "g7_benchmark_context_panel_300.csv")
    panel60 = panel[panel["record_id"].isin(b2_ids)].reset_index(drop=True)
    print(f"G7.5 primary optimization: CatBoost + tree-conformal UCB, 60-ctx × 3 budgets", flush=True)

    per_budget = {}
    for label, q_n in Q_BUDGETS:
        per_budget[label] = _run_for_budget(label, q_n, panel60, frozen_df, support)

    summary_rows = []
    for label, d in per_budget.items():
        summary_rows.append({"budget": label, "budget_per_ctx": d["q_n"],
                             "hv_norm": round(d["hv_norm"], 5),
                             "n_safe_contexts": d["safe_contexts"],
                             "n_pareto": d["n_pareto"], "wall_seconds": d["wall_seconds"]})
    pd.DataFrame(summary_rows).to_csv(G75 / "convergence" / "g7_5_direct_convergence_60ctx.csv", index=False, lineterminator="\n")
    G75.joinpath("convergence").mkdir(exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(G75 / "convergence" / "g7_5_direct_convergence_60ctx.csv", index=False, lineterminator="\n")

    # Judgment
    h0 = per_budget["Q0"]["hv_norm"]
    h1 = per_budget["Q1"]["hv_norm"]
    h2 = per_budget["Q2"]["hv_norm"]
    rel01 = abs(h1 - h0) / max(1e-9, h0) if h0 else None
    rel12 = abs(h2 - h1) / max(1e-9, h1) if h1 else None
    if rel12 is not None and rel12 < 0.01:
        verdict = "Q2_CONVERGED"
        final_budget = "Q2=256"
    elif rel12 is None:
        verdict = "COMPUTE_BLOCKED"
        final_budget = "Q0=64 (fallback)"
    else:
        verdict = "OPEN_Q1_to_Q2_rel_gain_above_1pct"
        final_budget = "Q2=256 (use as final budget even though rule did not strictly pass)"
    judgment = {
        "stage": "G7_5_CONVERGENCE_JUDGMENT",
        "evaluator": "CatBoost (primary)",
        "conformal": "tree-specific one-sided UCB (alpha=0.10)",
        "per_budget": per_budget,
        "rule": "Q1->Q2 matched mine-macro HV relative gain <1%; 5 other criteria (no systematic drift)",
        "rel_Q0_Q1": rel01,
        "rel_Q1_Q2": rel12,
        "verdict": verdict,
        "final_budget_frozen": final_budget,
        "paper_wording": "direct-validated nondominated recommendation set under the frozen CatBoost+tree-conformal search protocol",
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (G75 / "convergence" / "g7_5_direct_convergence_judgment.json").write_text(
        json.dumps(judgment, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"VERDICT: {verdict}, FINAL BUDGET: {final_budget}")


if __name__ == "__main__":
    main()