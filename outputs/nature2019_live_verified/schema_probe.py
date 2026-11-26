import sys, json
import pandas as pd, numpy as np

df = pd.read_csv(sys.argv[1])
out = {"n_rows": int(len(df)), "n_cols": int(df.shape[1]),
       "columns": list(df.columns)}
dtypes = {c: str(df[c].dtype) for c in df.columns}
out["dtypes"] = dtypes
na = {c: int(df[c].isna().sum()) for c in df.columns}
out["n_missing"] = {k: v for k, v in na.items() if v > 0}
nun = {c: int(df[c].nunique(dropna=True)) for c in df.columns}
out["n_unique"] = nun

# numeric summary for plausible numeric cols
num = df.select_dtypes(include=[np.number])
summ = {}
for c in num.columns:
    s = num[c].dropna()
    if len(s) == 0:
        continue
    summ[c] = {"min": float(s.min()), "p01": float(s.quantile(0.01)),
               "p50": float(s.quantile(0.5)), "p99": float(s.quantile(0.99)),
               "max": float(s.max()), "mean": float(s.mean())}
out["numeric_summary"] = summ

# show head of a few rows as dicts
out["head"] = df.head(3).astype(str).to_dict(orient="records")
print(json.dumps(out, indent=1)[:6000])