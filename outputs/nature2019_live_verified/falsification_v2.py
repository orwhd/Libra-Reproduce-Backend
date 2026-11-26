import sys, json, pathlib
import numpy as np
import pandas as pd
from scipy import stats

outdir = pathlib.Path(sys.argv[2])
df = pd.read_csv(sys.argv[1]).copy()
df["ln_t"] = np.log(df["team_size"].astype(float))
df["decade"] = (df["year"] // 10) * 10
df["stratum"] = df["field_name"].astype(str) + "|" + df["decade"].astype(str)

OUT = {}
r4 = lambda x: None if x is None else (round(float(x), 6) if abs(float(x)) < 1e6 else float(x))


def ols(X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = (X * (resid ** 2)[:, None]).T @ X
    V = XtX_inv @ meat @ XtX_inv * (n / max(n - k, 1))
    se = np.sqrt(np.abs(np.diag(V)))
    p = 2 * stats.t.sf(np.abs(beta / se), max(n - k, 1))
    sse = float(resid @ resid); sst = float(((y - y.mean()) ** 2).sum())
    return dict(beta=beta, se=se, p=p, sse=sse, r2=1 - sse / sst if sst else np.nan,
                n=int(n), k=int(k), resid=resid)


def fe_resid(vals, strata):
    D = pd.get_dummies(pd.Series(strata), drop_first=True).astype(float).values
    X = np.column_stack([np.ones(len(vals)), D])
    b, *_ = np.linalg.lstsq(X, np.asarray(vals, float), rcond=None)
    return np.asarray(vals, float) - X @ b


def slope_of(dv, x, strat=None):
    dv = np.asarray(dv, float); x = np.asarray(x, float)
    if strat is None:
        m = ols(np.column_stack([np.ones(len(x)), x]), dv)
        return dict(slope=r4(m["beta"][1]), se=r4(m["se"][1]), p=float(m["p"][1]),
                    r2=r4(m["r2"]), n=int(m["n"]))
    m = ols(fe_resid(x, strat).reshape(-1, 1), fe_resid(dv, strat))
    return dict(slope=r4(m["beta"][0]), se=r4(m["se"][0]), p=float(m["p"][0]),
                r2=r4(m["r2"]), n=int(m["n"]))


aic = lambda m: float(m["n"] * np.log(m["sse"] / m["n"]) + 2 * m["k"])

# ---- sample construction ----
cw = df["cohort_window"].value_counts()
OUT["cohort_windows"] = {str(k): int(v) for k, v in cw.items()}
primary_mask = ~df["cohort_window"].astype(str).str.contains("truncated")
prim = df[primary_mask].reset_index(drop=True)
OUT["primary_sample"] = {"n": int(len(prim)),
                         "windows_included": sorted(prim["cohort_window"].unique().tolist()),
                         "matches_reported_n_45335": bool(len(prim) == 45335)}

OUT["baseline_full_sample_incl_truncated"] = slope_of(df["disruption"], df["team_size"].astype(float))
base = slope_of(prim["disruption"], prim["team_size"].astype(float))
OUT["baseline_primary"] = base
OUT["baseline_delta_vs_headline"] = base["slope"] - (-0.00187)
OUT["baseline_pct_delta_vs_headline"] = 100 * (base["slope"] + 0.00187) / 0.00187
OUT["baseline_truncated_window_only"] = slope_of(
    df.loc[~primary_mask, "disruption"], df.loc[~primary_mask, "team_size"].astype(float))

d = prim["disruption"].values
t = prim["team_size"].values.astype(float)
lt = prim["ln_t"].values
strat = prim["stratum"].values

# ---- H1 functional form ----
lin = ols(np.column_stack([np.ones(len(t)), t]), d)
logm = ols(np.column_stack([np.ones(len(t)), lt]), d)
quad = ols(np.column_stack([np.ones(len(t)), t, t ** 2]), d)
logquad = ols(np.column_stack([np.ones(len(t)), lt, lt ** 2]), d)
OUT["H1_functional_form"] = {
    "linear_in_count": {"slope": r4(lin["beta"][1]), "p": float(lin["p"][1]), "r2": r4(lin["r2"]), "aic": r4(aic(lin))},
    "log_team_size": {"slope": r4(logm["beta"][1]), "p": float(logm["p"][1]), "r2": r4(logm["r2"]), "aic": r4(aic(logm))},
    "quadratic_count": {"lin": r4(quad["beta"][1]), "lin_p": float(quad["p"][1]),
                        "quad": r4(quad["beta"][2]), "quad_p": float(quad["p"][2]),
                        "r2": r4(quad["r2"]), "aic": r4(aic(quad))},
    "quadratic_log": {"lin": r4(logquad["beta"][1]), "lin_p": float(logquad["p"][1]),
                      "quad": r4(logquad["beta"][2]), "quad_p": float(logquad["p"][2]),
                      "r2": r4(logquad["r2"]), "aic": r4(aic(logquad))},
    "aic_gain_log_vs_linear": r4(aic(lin) - aic(logm)),
    "aic_gain_quad_log_vs_linear": r4(aic(lin) - aic(logquad)),
    "r2_gain_log_vs_linear": r4(logm["r2"] - lin["r2"]),
}
OUT["H1b_field_x_decade_fe"] = {
    "fe_linear_count": slope_of(d, t, strat),
    "fe_log": slope_of(d, lt, strat),
    "fe_quad_term": r4(ols(fe_resid(t ** 2, strat).reshape(-1, 1), fe_resid(d, strat))["beta"][0]),
    "fe_quad_p": float(ols(fe_resid(t ** 2, strat).reshape(-1, 1), fe_resid(d, strat))["p"][0]),
}

# ---- H2 piecewise ----
uniq_t = np.array(sorted(set(t[(t >= 2) & (t <= 12)])))
best, grid = None, []
for b in uniq_t:
    X = np.column_stack([np.ones(len(t)), np.minimum(t, b), np.maximum(0.0, t - b)])
    m = ols(X, d)
    if best is None or m["sse"] < best[1]["sse"]:
        best = (float(b), m)
    grid.append({"break": float(b), "sse": r4(m["sse"])})
bk, bm = best
Xp = np.column_stack([np.ones(len(t)), np.minimum(t, bk), np.maximum(0.0, t - bk)])
c = np.array([0.0, -1.0, 1.0])
XtX_inv = np.linalg.pinv(Xp.T @ Xp)
meat = (Xp * (bm["resid"] ** 2)[:, None]).T @ Xp
cov = XtX_inv @ meat @ XtX_inv * (len(t) / (len(t) - 3))
diff = float(c @ bm["beta"]); diff_se = float(np.sqrt(c @ cov @ c))
diff_p = float(2 * stats.t.sf(abs(diff / diff_se), len(t) - 3))
F = ((float(lin["sse"]) - float(bm["sse"])) / 1) / (float(bm["sse"]) / (len(t) - 3))
OUT["H2_piecewise"] = {
    "best_break": bk,
    "low_slope": r4(bm["beta"][1]), "low_p": float(bm["p"][1]),
    "high_slope": r4(bm["beta"][2]), "high_p": float(bm["p"][2]),
    "slope_change_high_minus_low": r4(diff), "change_se": r4(diff_se), "change_p": float(diff_p),
    "sse_reduction_vs_linear": r4(float(lin["sse"]) - float(bm["sse"])),
    "approx_F": r4(F), "approx_F_p": float(stats.f.sf(F, 1, len(t) - 3)),
    "high_slope_positive": bool(bm["beta"][2] > 0),
}
OUT["H2_piecewise_grid"] = grid

rng = np.random.default_rng(20261127)
boot = np.array([np.linalg.lstsq(Xp[i], d[i], rcond=None)[0][2]
                 for i in (rng.integers(0, len(t), len(t)) for _ in range(300))])
OUT["H2b_bootstrap_high_slope"] = {"n_boot": 300, "mean": r4(boot.mean()),
                                   "ci95": [r4(np.percentile(boot, 2.5)), r4(np.percentile(boot, 97.5))],
                                   "share_positive": r4((boot > 0).mean())}

bins = []
for k in range(1, 21):
    s = d[t == k]
    if len(s) >= 20:
        se_k = float(s.std(ddof=1) / np.sqrt(len(s)))
        bins.append({"t": k, "n": int(len(s)), "mean_d": r4(s.mean()), "se": r4(se_k),
                     "p": float(2 * stats.t.sf(abs(s.mean() / se_k), len(s) - 1))})
tail = d[t >= 21]
if len(tail) >= 20:
    se_t = float(tail.std(ddof=1) / np.sqrt(len(tail)))
    bins.append({"t": 21, "n": int(len(tail)), "mean_d": r4(tail.mean()), "se": r4(se_t),
                 "p": float(2 * stats.t.sf(abs(tail.mean() / se_t), len(tail) - 1))})
OUT["bin_means_exact"] = bins

w = np.array([b["n"] for b in bins], float); mu = np.array([b["mean_d"] for b in bins], float)
ts_b = np.array([b["t"] for b in bins], float)


def wls_slope(x, y, ww):
    X = np.column_stack([np.ones(len(x)), x]); W = np.diag(ww)
    b = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    return float(b[1])


sl_lo = wls_slope(ts_b[ts_b <= 5], mu[ts_b <= 5], w[ts_b <= 5])
sl_hi = wls_slope(ts_b[(ts_b >= 6) & (ts_b <= 20)], mu[(ts_b >= 6) & (ts_b <= 20)], w[(ts_b >= 6) & (ts_b <= 20)])
OUT["H2c_bin_mean_slopes"] = {"slope_1_to_5": r4(sl_lo), "slope_6_to_20": r4(sl_hi),
                              "sign_reversal": bool(np.sign(sl_lo) != np.sign(sl_hi))}

aa = d[(t >= 6) & (t <= 10)]; b11 = d[t >= 11]
wel = stats.ttest_ind(b11, aa, equal_var=False)
OUT["H2d_6_10_vs_11plus"] = {"mean_6_10": r4(aa.mean()), "n_6_10": int(len(aa)),
                             "mean_11plus": r4(b11.mean()), "n_11plus": int(len(b11)),
                             "diff": r4(b11.mean() - aa.mean()),
                             "welch_t": r4(wel.statistic), "welch_p": float(wel.pvalue)}

# ---- H3 outlier sensitivity ----
cites = prim["citation_count"].values.astype(float)
p99_g = np.percentile(cites, 99)
str_p99 = prim.groupby("stratum")["citation_count"].transform(lambda s: s.quantile(0.99)).values
q01, q99 = np.percentile(d, 1), np.percentile(d, 99)
dw = np.clip(d, q01, q99)
variants = {}
variants["drop_citation_gt_global_p99"] = slope_of(d[cites <= p99_g], t[cites <= p99_g])
mw = ols(np.column_stack([np.ones(len(t)), t]), dw)
variants["winsorize_disruption_1_99"] = {"slope": r4(mw["beta"][1]), "se": r4(mw["se"][1]),
                                         "p": float(mw["p"][1]), "r2": r4(mw["r2"]), "n": int(mw["n"])}
variants["drop_citation_gt_within_stratum_p99"] = slope_of(d[cites <= str_p99], t[cites <= str_p99])
mk = cites <= p99_g
variants["drop_citation_p99_AND_winsor"] = slope_of(dw[mk], t[mk])
Xb2 = np.column_stack([np.ones(len(t)), t])
h = np.einsum("ij,jk,ik->i", Xb2, np.linalg.pinv(Xb2.T @ Xb2), Xb2)
cook = (lin["resid"] ** 2 / (2 * (lin["sse"] / (len(t) - 2)))) * (h / (1 - h) ** 2)
keep = np.ones(len(t), bool); keep[np.argsort(-cook)[:100]] = False
variants["drop_top100_cook"] = slope_of(d[keep], t[keep])
for k in variants:
    variants[k]["pct_change_vs_baseline"] = r4(100 * (variants[k]["slope"] - base["slope"]) / abs(base["slope"]))
OUT["H3_outlier_sensitivity"] = {
    "thresholds": {"citation_p99_global": r4(p99_g),
                   "citation_p99_within_stratum_median": r4(np.median(str_p99)),
                   "disruption_p01": r4(q01), "disruption_p99": r4(q99)},
    "variants": variants}
m99 = cites > p99_g
OUT["H3b_trimming_composition"] = {
    "n_citation_outliers": int(m99.sum()),
    "share_team_ge6_full": r4((t >= 6).mean()),
    "share_team_ge6_outliers": r4((t[m99] >= 6).mean()) if m99.sum() else None,
    "share_team_ge11_full": r4((t >= 11).mean()),
    "share_team_ge11_outliers": r4((t[m99] >= 11).mean()) if m99.sum() else None,
    "mean_team_size_outliers": r4(t[m99].mean()) if m99.sum() else None,
    "mean_team_size_rest": r4(t[~m99].mean()),
    "mean_disruption_outliers": r4(d[m99].mean()) if m99.sum() else None,
    "mean_disruption_rest": r4(d[~m99].mean())}

OUT["H4_range_restricted"] = {
    "team_le_5": slope_of(d[t <= 5], t[t <= 5]),
    "team_le_10": slope_of(d[t <= 10], t[t <= 10]),
    "team_ge_3": slope_of(d[t >= 3], t[t >= 3]),
    "team_ge_6_log": slope_of(d[t >= 6], np.log(t[t >= 6])),
    "team_ge_6_linear": slope_of(d[t >= 6], t[t >= 6])}
OUT["H5_support"] = {"share_team_ge11": r4((t >= 11).mean()), "share_team_ge6": r4((t >= 6).mean()),
                     "median_team": float(np.median(t)), "p95_team": float(np.percentile(t, 95)),
                     "max_team": float(t.max()), "n_fields": int(prim["field_name"].nunique()),
                     "n_strata": int(prim["stratum"].nunique())}

(outdir / "falsification_full.json").write_text(json.dumps(OUT, indent=1, default=str))
print(json.dumps({k: OUT[k] for k in OUT if k not in ("bin_means_exact", "H2_piecewise_grid")}, indent=1, default=str))
print("BIN_MEANS", json.dumps(OUT["bin_means_exact"]))