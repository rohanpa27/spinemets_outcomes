"""
Full-cohort multiple-imputation multivariable model (robustness check).
Safeguards: outcome included as an auxiliary imputation column, features standardized
before imputing, imputations clipped to the observed range. Pooled with Rubin's rules.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

import cohort

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
M = 20
res = json.loads((ROOT / "outputs/results.json").read_text())
retained = res["enet_retained"]
c, _, _ = cohort.load()
OUT = "optimal_recovery"

D = c[retained + [OUT]].astype(float)
mu, sd = D.mean(), D.std().replace(0, 1)
Z = (D - mu) / sd
lo, hi = Z.min(), Z.max()

params, variances = [], []
for i in range(M):
    imp = IterativeImputer(sample_posterior=True, random_state=i, max_iter=10,
                           min_value=lo.values, max_value=hi.values)
    Zi = pd.DataFrame(imp.fit_transform(Z), columns=Z.columns, index=Z.index)
    Di = Zi * sd + mu
    m = sm.Logit(D[OUT].fillna(0).astype(int), sm.add_constant(Di[retained])).fit(disp=0, method="newton", maxiter=200)
    params.append(m.params[retained].values)
    variances.append((m.bse[retained].values) ** 2)

P, V = np.array(params), np.array(variances)
qbar, ubar, b = P.mean(0), V.mean(0), P.var(0, ddof=1)
T = ubar + (1 + 1 / M) * b
df = (M - 1) * (1 + ubar / ((1 + 1 / M) * b)) ** 2
tcrit = stats.t.ppf(0.975, df)
pv = 2 * (1 - stats.t.cdf(np.abs(qbar / np.sqrt(T)), df))
t = pd.DataFrame({"Predictor": retained, "aOR": np.exp(qbar),
                  "ci_lo": np.exp(qbar - tcrit * np.sqrt(T)), "ci_hi": np.exp(qbar + tcrit * np.sqrt(T)), "p": pv})
t["95% CI"] = t.apply(lambda r: f"{r['ci_lo']:.2f}-{r['ci_hi']:.2f}", axis=1)
t.sort_values("aOR").to_csv(ROOT / "outputs/tables/table7b_mice_multivariable.csv", index=False)
msg = f"MICE (m={M}, n={len(c)}): albumin aOR {t.set_index('Predictor').loc['preop_albumin', 'aOR']:.2f}"
(ROOT / "outputs/logs/mice.log").write_text(msg + "\n")
print(t.sort_values("aOR").to_string(index=False)); print(msg)
