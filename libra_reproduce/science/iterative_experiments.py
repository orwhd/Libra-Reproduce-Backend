from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..harness.scheduler import ResourceScheduler
from ..security.experiment_runner import SafeExperimentRunner
from ..security.tool_permissions import ToolPermissionGate

ROUND2_COUNTERFACTUAL_SIMULATION_SCRIPT = '''"""Multi-Round Confounder Control, Permutation Null, Jackknife & Generative Simulation Suite.

Rounds executed inside SafeExperimentRunner (`python -E`, stripped env, AST verified):
  - Round 2A: Citation-Quartile Confounder Control + Field x Decade Fixed-Effects OLS (FWL demeaning)
  - Round 2B: Within-Stratum Monte Carlo Permutation Null Experiment (n_perm = 200)
  - Round 2C: Leave-One-Discipline-Out Jackknife Stability Trial
  - Round 2D: Mechanistic Generative Citation-Network Simulation (Treatment M1 vs Size-Invariant Control M2)
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def _simulate_network_mechanism(rng: np.random.Generator, differential_breadth: bool) -> dict:
    """
    Generate an evolving directed citation DAG:
      - 240 background predecessor nodes with age rank + latent citation degree
      - 360 focal works (60 per team size k in [1, 2, 3, 5, 8, 12])
      - 600 forward citing papers citing focal works (+ co-citing focal predecessors)
    Computes exact D = (n_i - n_j) / (n_i + n_j + n_k) from simulated edges!
    """
    team_levels = [1, 2, 3, 5, 8, 12]
    n_per = 60
    n_bg = 240
    bg_recency = np.linspace(0.05, 1.0, n_bg)
    bg_popularity = rng.pareto(1.8, size=n_bg) + 1.0

    focal_rows = []
    for k in team_levels:
        for _ in range(n_per):
            if differential_breadth:
                # Larger teams review broader recent consensus hubs; solo/small teams probe older/niche roots
                n_refs = min(22, 3 + 2 * k)
                bias_power = 0.25 + 0.14 * k
                weights = (bg_recency ** bias_power) * (bg_popularity ** (0.15 + 0.10 * k))
            else:
                # Control M2: reference count & hub preference invariant to team size k
                n_refs = 8
                weights = (bg_recency ** 0.7) * (bg_popularity ** 0.5)
            probs = weights / weights.sum()
            ref_set = set(int(x) for x in rng.choice(n_bg, size=n_refs, replace=False, p=probs))
            mean_ref_hub = float(np.mean([bg_popularity[idx] for idx in ref_set]))

            # Simulate 24 forward citing attempts per focal paper cohort
            n_i = 0
            n_j = 0
            n_k = 0
            n_future = 28
            for _f in range(n_future):
                cites_focal = rng.random() < (0.36 + (0.015 * k if differential_breadth else 0.0))
                # Papers resting heavily on consensus hubs get co-cited alongside those hubs (j)
                co_cite_prob = min(0.88, 0.18 + 0.085 * mean_ref_hub * (len(ref_set) / 8.0))
                cites_ref = rng.random() < co_cite_prob
                if cites_focal and not cites_ref:
                    n_i += 1
                elif cites_focal and cites_ref:
                    n_j += 1
                elif (not cites_focal) and cites_ref:
                    n_k += 1

            denom = n_i + n_j + n_k
            d_val = float((n_i - n_j) / denom) if denom > 0 else float("nan")
            focal_rows.append({"team_size": k, "n_i": n_i, "n_j": n_j, "n_k": n_k, "d_sim": d_val})

    sim_df = pd.DataFrame(focal_rows).dropna(subset=["d_sim"])
    curve = sim_df.groupby("team_size")["d_sim"].agg(["mean", "std", "count"]).reset_index()
    slope, _, _, p_val, _ = stats.linregress(sim_df["team_size"].to_numpy(dtype=float), sim_df["d_sim"].to_numpy(dtype=float))
    return {
        "slope": round(float(slope), 6),
        "p_value": float(p_val),
        "curve": [
            {
                "team_size": int(r["team_size"]),
                "mean_d": round(float(r["mean"]), 5),
                "se": round(float((r["std"] or 0.0) / math.sqrt(max(1, r["count"]))), 5),
            }
            for _, r in curve.iterrows()
        ],
    }


def main() -> None:
    input_csv = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    if (df["year"] >= 1954).sum() >= 30 and (df["year"] <= 2014).sum() >= 30:
        df_p = df[(df["year"] >= 1954) & (df["year"] <= 2014)].copy()
    else:
        df_p = df.copy()
    df_p = df_p.dropna(subset=["disruption", "team_size", "citation_count", "year"]).copy()
    if "field_name" not in df_p.columns:
        df_p["field_name"] = "General Science"

    df_p["capped_ts"] = np.minimum(df_p["team_size"].to_numpy(dtype=float), 15.0)
    df_p["decade"] = (df_p["year"].astype(int) // 10) * 10
    df_p["stratum"] = df_p["field_name"].astype(str) + "__" + df_p["decade"].astype(str)

    # --- Round 2A: Two-Way Field x Decade Fixed Effects (FWL residualization) + Citation-Quartile Confounder Control ---
    d_dm = (df_p["disruption"] - df_p.groupby("stratum")["disruption"].transform("mean")).to_numpy(dtype=float)
    ts_dm = (df_p["capped_ts"] - df_p.groupby("stratum")["capped_ts"].transform("mean")).to_numpy(dtype=float)
    denom_fe = float(np.dot(ts_dm, ts_dm))
    fe_slope = float(np.dot(ts_dm, d_dm) / denom_fe) if denom_fe > 0 else 0.0
    resid = d_dm - fe_slope * ts_dm
    dof = max(1, len(df_p) - int(df_p["stratum"].nunique()) - 1)
    se_fe = float(math.sqrt(max(1e-18, float(np.dot(resid, resid)) / dof / max(1e-12, denom_fe))))
    t_fe = fe_slope / se_fe
    fe_p = float(2.0 * stats.t.sf(abs(t_fe), df=dof))

    # Citation quartile stratification
    try:
        df_p["cite_q"] = pd.qcut(df_p["citation_count"].rank(method="first"), q=4, labels=["Q1 Low", "Q2 Mid-Low", "Q3 Mid-High", "Q4 Top"])
    except Exception:
        df_p["cite_q"] = "All"

    quartile_rows = []
    experiment_rows = []
    for q_label in ["Q1 Low", "Q2 Mid-Low", "Q3 Mid-High", "Q4 Top"]:
        qdf = df_p[df_p["cite_q"] == q_label]
        if len(qdf) < 10:
            continue
        solo = qdf[qdf["team_size"] == 1]["disruption"]
        large = qdf[qdf["team_size"] >= 5]["disruption"]
        q_slope, _, _, q_p, _ = stats.linregress(qdf["capped_ts"], qdf["disruption"])
        row = {
            "quartile": q_label,
            "n": int(len(qdf)),
            "solo_mean_d": round(float(solo.mean()), 6) if len(solo) else float("nan"),
            "large5plus_mean_d": round(float(large.mean()), 6) if len(large) else float("nan"),
            "ols_slope": round(float(q_slope), 6),
            "p_value": float(q_p),
        }
        quartile_rows.append(row)
        experiment_rows.append({"round_id": "2A_citation_quartile", "condition": q_label, "metric": "ols_slope", "value": row["ols_slope"], "extra": f"n={row['n']}"})

    # --- Round 2B: Within-Stratum Monte Carlo Permutation Null Experiment (n_perm = 200) ---
    rng = np.random.default_rng(20261126)
    n_perm = 200
    obs_slope, _, _, _, _ = stats.linregress(df_p["capped_ts"].to_numpy(dtype=float), df_p["disruption"].to_numpy(dtype=float))

    # Sort once by stratum for ultra-fast within-stratum shuffling
    order_idx = np.argsort(df_p["stratum"].astype(str).to_numpy())
    stratum_sorted = df_p["stratum"].astype(str).to_numpy()[order_idx]
    ts_sorted = df_p["capped_ts"].to_numpy(dtype=float)[order_idx]
    d_sorted = df_p["disruption"].to_numpy(dtype=float)[order_idx]
    _, split_starts = np.unique(stratum_sorted, return_index=True)
    split_bounds = list(zip(split_starts, np.append(split_starts[1:], len(stratum_sorted))))
    ts_center = ts_sorted - ts_sorted.mean()
    d_center = d_sorted - d_sorted.mean()
    denom_perm = float(np.dot(ts_center, ts_center))

    null_slopes = np.empty(n_perm, dtype=float)
    perm_buf = ts_center.copy()
    for p_i in range(n_perm):
        for s_lo, s_hi in split_bounds:
            if s_hi - s_lo > 1:
                perm_buf[s_lo:s_hi] = rng.permutation(ts_center[s_lo:s_hi])
        null_slopes[p_i] = float(np.dot(perm_buf, d_center) / denom_perm) if denom_perm > 0 else 0.0

    null_mean = float(np.mean(null_slopes))
    null_std = float(np.std(null_slopes, ddof=1)) if n_perm > 1 else 1e-6
    perm_p = float((np.sum(np.abs(null_slopes) >= abs(obs_slope)) + 1.0) / (n_perm + 1.0))
    z_score = float((obs_slope - null_mean) / max(1e-9, null_std))
    experiment_rows.append({"round_id": "2B_permutation_null", "condition": f"n_perm={n_perm}", "metric": "observed_slope_vs_null_z", "value": round(z_score, 3), "extra": f"p_perm={perm_p:.5f}"})

    # --- Round 2C: Leave-One-Discipline-Out Jackknife Experiment ---
    field_counts = df_p["field_name"].value_counts()
    jack_strata = field_counts[field_counts >= 25].index.tolist()
    strat_col = "field_name"
    if len(jack_strata) < 3:
        strat_col = "decade"
        dec_counts = df_p["decade"].value_counts()
        jack_strata = dec_counts[dec_counts >= 10].index.tolist()

    jack_rows = []
    for st_val in jack_strata:
        sub_j = df_p[df_p[strat_col] != st_val]
        if len(sub_j) < 20:
            continue
        sl, _, _, pv, _ = stats.linregress(sub_j["capped_ts"], sub_j["disruption"])
        jack_rows.append({"dropped_stratum": str(st_val), "slope": round(float(sl), 6), "p_value": float(pv)})
        experiment_rows.append({"round_id": "2C_jackknife", "condition": f"drop={st_val}", "metric": "ols_slope", "value": round(float(sl), 6), "extra": f"n={len(sub_j)}"})

    jack_slopes = [r["slope"] for r in jack_rows] or [float(obs_slope)]

    # --- Round 2D: Mechanistic Generative Citation-Network Simulation (M1 Treatment vs M2 Control) ---
    sim_m1 = _simulate_network_mechanism(rng, differential_breadth=True)
    sim_m2 = _simulate_network_mechanism(rng, differential_breadth=False)
    experiment_rows.append({"round_id": "2D_mechanistic_sim", "condition": "M1_differential_breadth", "metric": "sim_slope", "value": sim_m1["slope"], "extra": f"p={sim_m1['p_value']:.2e}"})
    experiment_rows.append({"round_id": "2D_mechanistic_sim", "condition": "M2_size_invariant_ctrl", "metric": "sim_slope", "value": sim_m2["slope"], "extra": f"p={sim_m2['p_value']:.2e}"})

    pd.DataFrame(experiment_rows).to_csv(out_dir / "multiround_experiments.csv", index=False)

    # --- Publication 2x2 Figure ---
    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6), dpi=220)
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]

    # Panel A: Citation-Quartile Confounder Control
    q_names = [r["quartile"] for r in quartile_rows]
    q_x = np.arange(len(q_names))
    q_solo = [r["solo_mean_d"] for r in quartile_rows]
    q_large = [r["large5plus_mean_d"] for r in quartile_rows]
    ax_a.bar(q_x - 0.18, q_solo, width=0.35, color="#ea580c", label="Solo authors (k=1)")
    ax_a.bar(q_x + 0.18, q_large, width=0.35, color="#1d4ed8", label="Larger teams (k>=5)")
    ax_a.axhline(0.0, color="#94a3b8", ls=":", lw=1.0)
    ax_a.set_xticks(q_x)
    ax_a.set_xticklabels(q_names)
    ax_a.set_ylabel("Mean Disruption Index (D)")
    ax_a.set_title(f"Round 2A: Citation-Quartile Confounder Control\\n(Field×Decade FE slope = {fe_slope:.5f}, p = {fe_p:.1e})", fontweight="bold")
    ax_a.legend(loc="upper right", frameon=True, fontsize=8.0)

    # Panel B: Permutation Null Distribution
    ax_b.hist(null_slopes, bins=22, color="#94a3b8", alpha=0.85, edgecolor="white", label=f"Null slopes (n={n_perm})")
    ax_b.axvline(obs_slope, color="#dc2626", lw=2.3, ls="--", label=f"Observed = {obs_slope:.5f} (z={z_score:.1f})")
    ax_b.set_xlabel("Team-Size OLS Slope Under Within-Stratum Permutation")
    ax_b.set_ylabel("Monte Carlo Count")
    ax_b.set_title(f"Round 2B: Within-Stratum Permutation Null Test\\n(Empirical p = {perm_p:.4f})", fontweight="bold")
    ax_b.legend(loc="upper left", frameon=True, fontsize=8.0)

    # Panel C: Leave-One-Field-Out Jackknife Stability
    j_labels = [r["dropped_stratum"][:14] for r in jack_rows[:12]]
    j_vals = [r["slope"] for r in jack_rows[:12]]
    ax_c.barh(np.arange(len(j_labels)), j_vals, color="#0f766e", alpha=0.88)
    ax_c.axvline(obs_slope, color="#ea580c", ls="--", lw=1.5, label="Full-sample slope")
    ax_c.axvline(0.0, color="#64748b", ls=":", lw=1.0)
    ax_c.set_yticks(np.arange(len(j_labels)))
    ax_c.set_yticklabels(j_labels, fontsize=7.8)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("OLS Slope Excluding Discipline")
    ax_c.set_title(f"Round 2C: Leave-One-Out Jackknife Stability\\n({sum(1 for v in jack_slopes if v < 0)}/{len(jack_slopes)} strata negative)", fontweight="bold")
    ax_c.legend(loc="lower left", frameon=True, fontsize=8.0)

    # Panel D: Mechanistic Citation-Network Simulation (M1 vs M2)
    t1 = [pt["team_size"] for pt in sim_m1["curve"]]
    y1 = [pt["mean_d"] for pt in sim_m1["curve"]]
    e1 = [pt["se"] for pt in sim_m1["curve"]]
    t2 = [pt["team_size"] for pt in sim_m2["curve"]]
    y2 = [pt["mean_d"] for pt in sim_m2["curve"]]
    e2 = [pt["se"] for pt in sim_m2["curve"]]
    ax_d.errorbar(t1, y1, yerr=e1, marker="o", lw=2.2, color="#7c3aed", capsize=3, label=f"M1 Search Breadth Model (slope={sim_m1['slope']:.4f})")
    ax_d.errorbar(t2, y2, yerr=e2, marker="s", lw=1.8, ls="--", color="#64748b", capsize=3, label=f"M2 Size-Invariant Ctrl (slope={sim_m2['slope']:.4f})")
    ax_d.axhline(0.0, color="#94a3b8", ls=":", lw=1.0)
    ax_d.set_xlabel("Simulated Team Size (k)")
    ax_d.set_ylabel("Simulated Exact CD Index D = (ni - nj)/(ni + nj + nk)")
    ax_d.set_title("Round 2D: Mechanistic Citation-Network Simulation\\n(Treatment M1 vs Size-Invariant Control M2)", fontweight="bold")
    ax_d.legend(loc="best", frameon=True, fontsize=8.0)

    fig.tight_layout(pad=1.6)
    fig_path = out_dir / "figure_multiround_experiments.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    summary = {
        "fixed_effects_field_decade": {
            "slope": round(fe_slope, 6),
            "se": round(se_fe, 6),
            "p_value": fe_p,
            "n_strata": int(df_p["stratum"].nunique()),
        },
        "citation_quartiles": quartile_rows,
        "permutation_null": {
            "n_permutations": n_perm,
            "observed_slope": round(float(obs_slope), 6),
            "null_mean": round(null_mean, 6),
            "null_std": round(null_std, 6),
            "z_score": round(z_score, 3),
            "p_value": round(perm_p, 5),
        },
        "jackknife": {
            "total_strata": len(jack_slopes),
            "negative_count": sum(1 for v in jack_slopes if v < 0),
            "min_slope": round(min(jack_slopes), 6),
            "max_slope": round(max(jack_slopes), 6),
            "rows": jack_rows,
        },
        "mechanistic_simulation": {
            "m1_slope": sim_m1["slope"],
            "m1_p_value": sim_m1["p_value"],
            "m2_slope": sim_m2["slope"],
            "m2_p_value": sim_m2["p_value"],
            "m1_curve": sim_m1["curve"],
            "m2_curve": sim_m2["curve"],
        },
    }
    (out_dir / "experiment_rounds_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("MULTIROUND_OK", json.dumps({"fe_slope": fe_slope, "perm_p": perm_p, "sim_m1": sim_m1["slope"], "sim_m2": sim_m2["slope"]}))


if __name__ == "__main__":
    main()
'''

TARGETED_FOLLOWUP_SCRIPT = '''"""Executed by dynamically spawned specialist workers under SafeExperimentRunner."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

csv_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42

df = pd.read_csv(csv_path).dropna(subset=["team_size", "disruption", "citation_count"])
rng = np.random.default_rng(seed)
# Subsample bootstrap & extreme-citation trimming sensitivity check
sub = df.sample(n=min(len(df), 8000), random_state=int(seed))
q95_cite = float(sub["citation_count"].quantile(0.95))
trimmed = sub[sub["citation_count"] <= q95_cite]
ts_t = np.minimum(trimmed["team_size"].to_numpy(dtype=float), 15.0)
sl, _, _, pv, _ = stats.linregress(ts_t, trimmed["disruption"].to_numpy(dtype=float))
res = {
    "n_evaluated": int(len(trimmed)),
    "trimmed_top5pct_citations_slope": round(float(sl), 6),
    "trimmed_p_value": float(pv),
    "direction_negative": bool(sl < 0),
}
out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
print("FOLLOWUP_EXP_OK", json.dumps(res))
'''


async def run_multiround_counterfactual_suite(
    workspace_dir: Path,
    sample_csv_path: Path,
) -> dict[str, Any]:
    runner = SafeExperimentRunner(workspace_dir)
    out_dir = workspace_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    exec_res = await runner.run_async(
        "scripts/exp_round2_counterfactual_simulation.py",
        ROUND2_COUNTERFACTUAL_SIMULATION_SCRIPT,
        args=[str(sample_csv_path), str(out_dir)],
    )
    if not exec_res.ok:
        raise RuntimeError(f"Multi-round counterfactual/simulation runner failed: {exec_res.error}")
    summary_path = out_dir / "experiment_rounds_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "execution": exec_res,
        "summary": summary,
        "summary_path": summary_path,
        "csv_path": out_dir / "multiround_experiments.csv",
        "fig_path": out_dir / "figure_multiround_experiments.png",
    }


async def run_targeted_followup_experiment(
    workspace_dir: Path,
    sample_csv_path: Path,
    *,
    script_rel_path: str,
    seed_offset: int,
) -> dict[str, Any]:
    runner = SafeExperimentRunner(workspace_dir)
    out_json = workspace_dir / "outputs" / f"followup_{seed_offset}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    exec_res = await runner.run_async(
        script_rel_path,
        TARGETED_FOLLOWUP_SCRIPT,
        args=[str(sample_csv_path), str(out_json), str(2026 + seed_offset)],
    )
    if not exec_res.ok:
        raise RuntimeError(f"Follow-up experiment failed: {exec_res.error}")
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    return {"execution": exec_res, "result": payload, "json_path": out_json}


def build_live_experiment_tools(
    *,
    run_id: str,
    workspace_dir: Path,
    sample_csv_path: Path,
    scheduler: ResourceScheduler,
    owner_task_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """
    Build real controlled MCP tools that empower live agents to think across turns,
    inspect empirical outputs, author custom Python experiments inside `SafeExperimentRunner`,
    diagnose failures/stderr and re-run across rounds, and log verified findings.
    """
    gate = ToolPermissionGate(workspace_dir)
    recorded_trials: list[dict[str, Any]] = []

    async def _inspect_workspace_data(args: dict[str, Any]) -> dict[str, Any]:
        rel = str(args.get("relative_path") or "").strip()
        if rel:
            target = gate.check_path(rel)
            if not target.exists():
                return {"ok": False, "error": f"File '{rel}' does not exist."}
            text = target.read_text(encoding="utf-8", errors="replace")[:4500]
            return {"ok": True, "file": rel, "preview": text}

        out_dir = workspace_dir / "outputs"
        summaries: dict[str, Any] = {
            "ok": True,
            "sample_csv_rel": str(sample_csv_path.relative_to(workspace_dir)) if sample_csv_path.is_relative_to(workspace_dir) else sample_csv_path.name,
            "available_outputs": sorted(p.name for p in out_dir.glob("*")) if out_dir.exists() else [],
        }
        for name in ("experiment_summary.json", "experiment_rounds_summary.json"):
            p = out_dir / name
            if p.exists():
                summaries[name] = json.loads(p.read_text(encoding="utf-8"))
        return summaries

    async def _run_python_experiment(args: dict[str, Any]) -> dict[str, Any]:
        raw_name = str(args.get("script_name") or "agent_trial.py").strip()
        hypothesis = str(args.get("hypothesis") or "Custom follow-up experimental test").strip()
        code = str(args.get("python_code") or "").strip()
        safe_stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(raw_name).stem).strip("_") or "trial"
        rel_script = f"scripts/agent_trials/{safe_stem}.py"

        if not code:
            return {"ok": False, "error": "python_code is required"}

        runner = SafeExperimentRunner(workspace_dir, timeout_seconds=45)
        out_dir = workspace_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        async with scheduler.experiment_slot(run_id=run_id, owner_id=f"{owner_task_id}:{safe_stem}"):
            exec_res = await runner.run_async(
                rel_script,
                code,
                args=[str(sample_csv_path), str(out_dir)],
            )

        entry = {
            "kind": "agent_python_trial",
            "script_name": f"{safe_stem}.py",
            "script_path": str(exec_res.script_path),
            "hypothesis": hypothesis,
            "ok": exec_res.ok,
            "exit_code": exec_res.exit_code,
            "duration_ms": exec_res.duration_ms,
            "script_sha256": exec_res.script_sha256,
            "stdout": exec_res.stdout[:1800],
            "stderr": exec_res.stderr[:1200],
            "error": exec_res.error,
        }
        recorded_trials.append(entry)
        return entry

    async def _record_experiment_finding(args: dict[str, Any]) -> dict[str, Any]:
        finding = {
            "kind": "recorded_finding",
            "round_id": str(args.get("round_id") or f"round_{len(recorded_trials) + 1}"),
            "hypothesis": str(args.get("hypothesis") or ""),
            "method": str(args.get("method") or ""),
            "quantitative_result": str(args.get("quantitative_result") or ""),
            "verdict": str(args.get("verdict") or "supported"),
            "next_question": str(args.get("next_question") or ""),
        }
        recorded_trials.append(finding)
        return {"ok": True, "recorded": finding}

    tools_spec = [
        {
            "name": "inspect_workspace_data",
            "description": "Inspect available empirical CSV/JSON outputs and summary statistics inside the jailed run workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "relative_path": {
                        "type": "string",
                        "description": "Optional relative path inside workspace (e.g. 'outputs/experiment_summary.json'). Omit to list outputs and JSON summaries.",
                    }
                },
            },
        },
        {
            "name": "run_python_experiment",
            "description": (
                "Execute an authored Python experiment inside the isolated `-E` AST-verified sandbox (`pandas`, `numpy`, `scipy`, `matplotlib`, `json`, `math`, `pathlib`; NO `os`, `subprocess`, or network imports). "
                "sys.argv[1] is the input empirical dataset CSV path, and sys.argv[2] is the outputs directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "script_name": {"type": "string", "description": "Short filename like 'nonlinear_log_test.py'."},
                    "hypothesis": {"type": "string", "description": "Falsifiable hypothesis or simulation question tested."},
                    "python_code": {"type": "string", "description": "Complete self-contained Python 3 script reading sys.argv[1] and printing JSON summary metrics to stdout."},
                },
                "required": ["script_name", "hypothesis", "python_code"],
            },
        },
        {
            "name": "record_experiment_finding",
            "description": "Record a structured quantitative finding and next open question after running an experiment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "round_id": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "method": {"type": "string"},
                    "quantitative_result": {"type": "string"},
                    "verdict": {"type": "string"},
                    "next_question": {"type": "string"},
                },
                "required": ["round_id", "hypothesis", "quantitative_result", "verdict"],
            },
        },
    ]
    handlers = {
        "inspect_workspace_data": _inspect_workspace_data,
        "run_python_experiment": _run_python_experiment,
        "record_experiment_finding": _record_experiment_finding,
    }
    return tools_spec, handlers, recorded_trials
