import sys, json
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv(sys.argv[1])
df = df.copy()
df["ln_t"] = np.log(df["team_size"].astype(float))
df["decade"] = (df["year"] // 10) * 10
df["stratum"] = df["field_name"].astype(str) + "|" + df["decade"].astype(str)

OUT = {}


def ols(X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = (X * (resid ** 2)[:, None]).T @ X
    V = XtX_inv @ meat @ XtX_inv * (n / max(n - k, 1))
    se = np.sqrt(np.abs(np.diag(V)))
    tstat = beta / se
    p = 2 * stats.t.sf(np.abs(tstat), max(n - k, 1))
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - sse / sst if sst > 0 else np.nan
    return dict(beta=beta, se=se, p=p, sse=sse, r2=r2, n=int(n), k=int(k), resid=resid)


def fe_resid(vals, strata):
    D = pd.get_dummies(pd.Series(strata), drop_first=True).astype(float).values
    X = np.column_stack([np.ones(len(vals)), D])
    b, *_ = np.linalg.lstsq(X, np.asarray(vals, float), rcond=None)
    return np.asarray(vals, float) - X @ b


def slope_of(dv, x, strat=None):
    dv = np.asarray(dv, float); x = np.asarray(x, float)
    if strat is None:
        m = ols(np.column_stack([np.ones(len(x)), x]), dv)
        return dict(slope=float(m["beta"][1]), se=float(m["se"][1]),
                    p=float(m["p"][1]), r2=float(m["r2"]), n=int(m["n"]))
    dr = fe_resid(dv, strat); xr = fe_resid(x, strat)
    m = ols(xr.reshape(-1, 1), dr)
    return dict(slope=float(m["beta"][0]), se=float(m["se"][0]),
                p=float(m["p"][0]), r2=float(m["r2"]), n=int(m["n"]))


def aic(m):
    return float(m["n"] * np.log(m["sse"] / m["n"]) + 2 * m["k"])


# ---------- window composition ----------
cw = df["cohort_window"].value_counts()
OUT["cohort_windows"] = {str(k): int(v) for k, v in cw.items()}
primary_window = cw.index[0]
OUT["primary_window_used"] = str(primary_window)

full_slope = slope_of(df["disruption"].values, df["team_size"].values.astype(float))
prim = df[df["cohort_window"] == primary_window].reset_index(drop=True)
base = slope_of(prim["disruption"].values, prim["team_size"].values.astype(float))
OUT["baseline_full_sample"] = full_slope
OUT["baseline_primary_window"] = base
OUT["headline_target"] = -0.00187

d = prim["disruption"].values
t = prim["team_size"].values.astype(float)
lt = prim["ln_t"].values
strat = prim["stratum"].values

# ---------- H1: functional form ----------
lin = ols(np.column_stack([np.ones(len(t)), t]), d)
logm = ols(np.column_stack([np.ones(len(t)), lt]), d)
quad = ols(np.column_stack([np.ones(len(t)), t, t ** 2]), d)
logquad = ols(np.column_stack([np.ones(len(t)), lt, lt ** 2]), d)

OUT["H1_functional_form"] = {
    "linear_in_count": {"slope": float(lin["beta"][1]), "p": float(lin["p"][1]),
                        "r2": float(lin["r2"]), "aic": aic(lin)},
    "log_team_size": {"slope": float(logm["beta"][1]), "p": float(logm["p"][1]),
                      "r2": float(logm["r2"]), "aic": aic(logm)},
    "quadratic_count": {"lin_term": float(quad["beta"][1]), "lin_p": float(quad["p"][1]),
                        "quad_term": float(quad["beta"][2]), "quad_p": float(quad["p"][2]),
                        "r2": float(quad["r2"]), "aic": aic(quad)},
    "quadratic_log": {"lin_term": float(logquad["beta"][1]), "lin_p": float(logquad["p"][1]),
                      "quad_term": float(logquad["beta"][2]), "quad_p": float(logquad["p"][2]),
                      "r2": float(logquad["r2"]), "aic": aic(logquad)},
    "aic_gain_log_vs_linear": aic(lin) - aic(logm),
    "aic_gain_quad_vs_linear": aic(lin) - aic(quad),
    "r2_gain_log_vs_linear": float(logm["r2"] - lin["r2"]),
}

dr = fe_resid(d, strat); tr = fe_resid(t, strat); ltr = fe_resid(lt, strat)
fe_lin = ols(tr.reshape(-1, 1), dr)
fe_log = ols(ltr.reshape(-1, 1), dr)
OUT["H1b_fe_specs"] = {
    "fe_linear_count_slope": float(fe_lin["beta"][0]), "p": float(fe_lin["p"][0]),
    "fe_log_slope": float(fe_log["beta"][0]), "p": float(fe_log["p"][0]),
}

# ---------- H2: piecewise / segmented ----------
uniq_t = np.array(sorted(set(t[(t >= 2) & (t <= 15)])))
best = None
grid = []
for b in uniq_t:
    X = np.column_stack([np.ones(len(t)), np.minimum(t, b), np.maximum(0.0, t - b)])
    m = ols(X, d)
    if best is None or m["sse"] < best[1]["sse"]:
        best = (float(b), m)
    grid.append({"break": float(b), "sse": float(m["sse"])})
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
    "low_slope": float(bm["beta"][1]), "low_p": float(bm["p"][1]),
    "high_slope": float(bm["beta"][2]), "high_p": float(bm["p"][2]),
    "slope_change_high_minus_low": diff, "se": diff_se, "p": diff_p,
    "sse_reduction_vs_linear": float(lin["sse"]) - float(bm["sse"]),
    "approx_F": float(F), "approx_F_p": float(stats.f.sf(F, 1, len(t) - 3)),
    "high_slope_positive": bool(bm["beta"][2] > 0),
}
OUT["H2_piecewise_grid_top5"] = sorted(grid, key=lambda g: g["sse"])[:5]

rng = np.random.default_rng(20261127)
boot = []
n = len(t)
for _ in range(300):
    idx = rng.integers(0, n, n)
    bb, *_ = np.linalg.lstsq(Xp[idx], d[idx], rcond=None)
    boot.append(bb[2])
boot = np.array(boot)
OUT["H2b_bootstrap_high_slope"] = {
    "n_boot": 300, "mean": float(boot.mean()),
    "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    "share_positive": float((boot > 0).mean()),
}

bins = []
for k in range(1, 21):
    s = d[t == k]
    if len(s) >= 20:
        se_k = float(s.std(ddof=1) / np.sqrt(len(s)))
        bins.append({"team_size": k, "n": int(len(s)), "mean_d": float(s.mean()), "se": se_k,
                     "p_vs_zero": float(2 * stats.t.sf(abs(s.mean() / se_k), len(s) - 1))})
tail = d[t >= 21]
if len(tail) >= 20:
    se_t = float(tail.std(ddof=1) / np.sqrt(len(tail)))
    bins.append({"team_size": 21, "n": int(len(tail)), "mean_d": float(tail.mean()), "se": se_t,
                 "p_vs_zero": float(2 * stats.t.sf(abs(tail.mean() / se_t), len(tail) - 1))})
OUT["bin_means_exact"] = bins

w = np.array([b["n"] for b in bins], float)
mu = np.array([b["mean_d"] for b in bins])
ts_b = np.array([b["team_size"] for b in bins], float)


def wls_slope(x, y, ww):
    X = np.column_stack([np.ones(len(x)), x])
    W = np.diag(ww)
    b = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    r = y - X @ b
    return float(b[1]), float(np.sqrt((r ** 2 * ww).sum() / (ww.sum() - 2)))


seg_lo = ts_b <= 5; seg_hi = (ts_b >= 6) & (ts_b <= 20)
sl_lo, _ = wls_slope(ts_b[seg_lo], mu[seg_lo], w[seg_lo])
sl_hi, _ = wls_slope(ts_b[seg_hi], mu[seg_hi], w[seg_hi])
OUT["H2c_bin_slopes"] = {"slope_1_to_5_per_author": sl_lo, "slope_6_to_20_per_author": sl_hi,
                         "sign_reversal": bool(np.sign(sl_lo) != np.sign(sl_hi))}

aa = d[(t >= 6) & (t <= 10)]; b11 = d[t >= 11]
tt = stats.ttest_ind(b11, aa, equal_var=False)
OUT["H2d_bin_6_10_vs_11plus"] = {"mean_6_10": float(aa.mean()), "n_6_10": int(len(aa)),
                                 "mean_11plus": float(b11.mean()), "n_11plus": int(len(b11)),
                                 "diff": float(b11.mean() - aa.mean()),
                                 "welch_t": float(tt.statistic), "welch_p": float(tt.pvalue)}

# ---------- H3: outlier sensitivity ----------
cites = prim["citation_count"].values.astype(float)
p99_g = np.percentile(cites, 99)
str_p99 = prim.groupby("stratum")["citation_count"].transform(lambda s: s.quantile(0.99)).values
q01, q99 = np.percentile(d, 1), np.percentile(d, 99)
dw = np.clip(d, q01, q99)

variants = {}
variants["drop_citation_gt_global_p99"] = slope_of(d[cites <= p99_g], t[cites <= p99_g])
mw = ols(np.column_stack([np.ones(len(t)), t]), dw)
variants["winsorize_disruption_1_99"] = {"slope": float(mw["beta"][1]), "se": float(mw["se"][1]),
                                         "p": float(mw["p"][1]), "r2": float(mw["r2"]), "n": int(mw["n"])}
variants["drop_citation_gt_within_stratum_p99"] = slope_of(d[cites <= str_p99], t[cites <= str_p99])
mk = cites <= p99_g
variants["drop_citation_p99_AND_winsor_disruption"] = slope_of(dw[mk], t[mk])
XtX_inv_d = np.linalg.pinv(np.column_stack([np.ones(len(t)), t]).T @ np.column_stack([np.ones(len(t)), t]))
Xb2 = np.column_stack([np.ones(len(t)), t])
h = np.einsum("ij,jk,ik->i", Xb2, XtX_inv_d, Xb2)
resid_b = lin["resid"]; pp = 2
cook = (resid_b ** 2 / (pp * (lin["sse"] / (len(t) - pp)))) * (h / (1 - h) ** 2)
keep = np.ones(len(t), bool); keep[np.argsort(-cook)[:100]] = False
variants["drop_top100_cooks_distance"] = slope_of(d[keep], t[keep])

for k in variants:
    variants[k]["pct_change_vs_baseline"] = 100 * (variants[k]["slope"] - base["slope"]) / abs(base["slope"])

OUT["H3_outlier_sensitivity"] = {
    "thresholds": {"citation_p99_global": float(p99_g),
                   "citation_p99_within_stratum_median": float(np.median(str_p99)),
                   "disruption_p01": float(q01), "disruption_p99": float(q99)},
    "variants": variants,
}

mask_p99 = cites > p99_g
OUT["H3b_trimming_composition"] = {
    "share_team_ge6_full_sample": float((t >= 6).mean()),
    "share_team_ge6_among_citation_outliers": float((t[mask_p99] >= 6).mean()) if mask_p99.sum() else None,
    "share_team_ge11_full_sample": float((t >= 11).mean()),
    "share_team_ge11_among_citation_outliers": float((t[mask_p99] >= 11).mean()) if mask_p99.sum() else None,
    "mean_team_size_citation_outliers": float(t[mask_p99].mean()) if mask_p99.sum() else None,
    "mean_team_size_rest": float(t[~mask_p99].mean()),
    "mean_disruption_citation_outliers": float(d[mask_p99].mean()) if mask_p99.sum() else None,
    "n_citation_outliers": int(mask_p99.sum()),
}

OUT["H4_range_restricted_slopes"] = {
    "team_le_5": slope_of(d[t <= 5], t[t <= 5]),
    "team_le_10": slope_of(d[t <= 10], t[t <= 10]),
    "team_ge_3": slope_of(d[t >= 3], t[t >= 3]),
    "team_ge_6_log": slope_of(d[t >= 6], np.log(t[t >= 6])),
}

OUT["H5_mass_check"] = {"share_rows_team_ge11": float((t >= 11).mean()),
                        "share_rows_team_ge6": float((t >= 6).mean()),
                        "median_team_size": float(np.median(t)),
                        "p95_team_size": float(np.percentile(t, 95))}

print(json.dumps(OUT, indent=1, default=str))