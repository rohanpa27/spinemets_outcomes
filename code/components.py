"""
Component-specific models: for each of the six recovery criteria, the association of each
selected factor with failure of that criterion (1 = adverse event), full analytic cohort.
Continuous factors are scaled to per-1-SD odds ratios so effects are comparable across factors.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from statsmodels.stats.multitest import multipletests

import cohort

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
res = json.loads((ROOT / "outputs/results.json").read_text())
retained = res["enet_retained"]

c, _, _ = cohort.load()
X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(c[retained]), columns=retained, index=c.index)
binary = [v for v in retained if c[v].dropna().nunique() <= 2]
for v in retained:
    if v not in binary:
        X[v] = (X[v] - X[v].mean()) / X[v].std()  # per 1 SD

rows, log = [], []
for comp, col in cohort.COMPONENTS.items():
    y = c[col].astype(int)
    use = [v for v in retained if X[v].nunique() > 1]
    skipped = [v for v in retained if v not in use]
    if skipped:
        log.append(f"{comp}: skipped zero-variance {skipped}")
    D = sm.add_constant(X[use].astype(float))
    penalized = False
    try:
        m = sm.Logit(y, D).fit(disp=0, method="newton", maxiter=200)
        params, ci, pv = m.params, m.conf_int(), m.pvalues
        if not np.isfinite(m.bse).all() or (m.bse > 50).any():
            raise ValueError("unstable fit")
    except Exception as e:
        log.append(f"{comp}: unpenalized fit failed ({e}); using lightly penalized fit")
        m = sm.Logit(y, D).fit_regularized(alpha=1e-3, disp=0, maxiter=500)
        params = m.params
        ci = pd.DataFrame(np.nan, index=params.index, columns=[0, 1])
        pv = pd.Series(np.nan, index=params.index)
        penalized = True
    for v in use:
        rows.append({"Predictor": v, "Component": comp, "aOR": float(np.exp(params[v])),
                     "ci_lo": float(np.exp(ci.loc[v, 0])), "ci_hi": float(np.exp(ci.loc[v, 1])),
                     "p": float(pv[v]), "penalized": penalized, "events": int(y.sum())})

long = pd.DataFrame(rows)
ok = long["p"].notna()
long.loc[ok, "fdr_q"] = multipletests(long.loc[ok, "p"], method="fdr_bh")[1]
long["aOR"] = long["aOR"].clip(0.001, 1000)

long.to_csv(ROOT / "outputs/tables/table5_component_models_long.csv", index=False)

wide = pd.DataFrame({"Predictor": retained})
for comp in cohort.COMPONENTS:
    s = long[long["Component"] == comp].set_index("Predictor")
    wide[comp] = [("" if v not in s.index else
                   f"{s.loc[v, 'aOR']:.2f} ({s.loc[v, 'ci_lo']:.2f}-{s.loc[v, 'ci_hi']:.2f})"
                   f"{'*' if s.loc[v, 'fdr_q'] < 0.05 else ''}") for v in retained]
wide.to_csv(ROOT / "outputs/tables/table5_component_models.csv", index=False)
(ROOT / "outputs/logs/components.log").write_text("\n".join(log) + "\n")
print(wide.to_string(index=False))
print(*log, sep="\n")
