"""
Factor analysis for optimal recovery after surgery for spinal metastases
(ACS-NSQIP 2016-2021). Writes tables, supplement tables, results.json, and a log.

Outcome: optimal_recovery = no 30-day mortality, no major complication, no unplanned
reoperation, no unplanned readmission, LOS at or below the cohort median, home discharge.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportion_confint

import cohort
import code_labels as L

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
TABLES, SUPP, LOGS = ROOT / "outputs/tables", ROOT / "outputs/supplement", ROOT / "outputs/logs"
for d in (TABLES, SUPP, LOGS, ROOT / "data"):
    d.mkdir(parents=True, exist_ok=True)

OUTCOME = "optimal_recovery"
PREDICTORS = cohort.PREDICTORS
RNG_SEED = 42
L1_RATIOS = [0.1, 0.5, 0.9]
HEADLINE = ["preop_albumin", "preop_hematocrit", "asa_class", "preop_sodium", "preop_bun"]
LOG_LINES = []


def log(msg=""):
    print(msg)
    LOG_LINES.append(str(msg))


def enet(Xs, y, seed=RNG_SEED, Cs=50, cv=10, max_iter=10000):
    m = LogisticRegressionCV(Cs=Cs, cv=cv, penalty="elasticnet", solver="saga", l1_ratios=L1_RATIOS,
                             scoring="roc_auc", random_state=seed, max_iter=max_iter)
    return m.fit(Xs, y)


def logit_table(df, y, cols):
    """Adjusted odds ratios from an unpenalized logistic model on rows with complete data."""
    d = df[cols + [y]].dropna()
    cols = [v for v in cols if d[v].nunique() > 1]
    m = sm.Logit(d[y], sm.add_constant(d[cols].astype(float))).fit(disp=0, method="newton", maxiter=200)
    ci = m.conf_int()
    rows = [{"Predictor": p, "aOR": float(np.exp(m.params[p])), "ci_lo": float(np.exp(ci.loc[p, 0])),
             "ci_hi": float(np.exp(ci.loc[p, 1])), "p": float(m.pvalues[p])} for p in cols]
    return m, pd.DataFrame(rows), len(d)


def median_impute(df, cols):
    out = df.copy()
    out[cols] = SimpleImputer(strategy="median").fit_transform(df[cols])
    return out


def fmt_ci(r):
    return f"{r['ci_lo']:.2f}-{r['ci_hi']:.2f}"


def smd(a, b):
    s = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return abs(a.mean() - b.mean()) / s if s > 0 else 0.0


# ------------------------------------------------------------------ data
c, flow, median_los = cohort.load()
cohort.strobe()
analytic = c[PREDICTORS + [OUTCOME]].copy()
n, n_rec = len(c), int(c[OUTCOME].sum())
log(f"Analytic cohort {n}; optimal recovery {n_rec} ({n_rec / n:.1%}); median LOS {median_los:.0f} d")

res = {"n": n, "recovery_n": n_rec, "recovery_rate": n_rec / n, "median_los": median_los,
       "transfusion_rate": float(c["transfusion"].mean()),
       "recovery_rate_no_transfusion": float(c["optimal_recovery_no_transfusion"].mean()),
       "recovery_rate_no_discharge_criterion": float(c["recovery_no_discharge_criterion"].mean()),
       "component_failure_rates": {k: float(c[v].mean()) for k, v in cohort.COMPONENTS.items()},
       "recovery_by_year": {int(y): {"rate": float(g[OUTCOME].mean()), "n": int(len(g))}
                            for y, g in c.groupby("year_of_surgery")}}

# ------------------------------------------------------------------ near-zero variance, split
nzv = []
for p in PREDICTORS:
    col = analytic[p].dropna()
    if col.nunique() < 2 or (col.nunique() == 2 and col.value_counts(normalize=True).min() < 0.01):
        nzv.append(p)
model_predictors = [p for p in PREDICTORS if p not in nzv]
res["nzv_dropped"] = nzv
X, y = analytic[model_predictors], analytic[OUTCOME]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=RNG_SEED, stratify=y)
res["train_n"], res["test_n"] = len(X_train), len(X_test)
log(f"Split 70/30: train {len(X_train)} ({y_train.mean():.1%}), test {len(X_test)} ({y_test.mean():.1%}); nzv dropped {nzv}")

# ------------------------------------------------------------------ Table 1 baseline
g1, g0 = analytic[analytic[OUTCOME] == 1], analytic[analytic[OUTCOME] == 0]
t1 = []
for p in PREDICTORS:
    a, b = g1[p].dropna().astype(float), g0[p].dropna().astype(float)
    if len(a) == 0 or len(b) == 0:
        continue
    d = smd(a, b)
    if analytic[p].dropna().nunique() <= 2:
        try:
            pv = chi2_contingency(pd.crosstab(analytic[OUTCOME], analytic[p]))[1]
        except Exception:
            pv = np.nan
        t1.append({"Variable": p, "Suboptimal recovery": f"{b.mean():.1%}", "Optimal recovery": f"{a.mean():.1%}",
                   "P value": pv, "SMD": d})
    else:
        try:
            pv = mannwhitneyu(a, b)[1]
        except Exception:
            pv = np.nan
        t1.append({"Variable": p,
                   "Suboptimal recovery": f"{b.median():.1f} ({b.quantile(.25):.1f}-{b.quantile(.75):.1f})",
                   "Optimal recovery": f"{a.median():.1f} ({a.quantile(.25):.1f}-{a.quantile(.75):.1f})",
                   "P value": pv, "SMD": d})
table1 = pd.DataFrame(t1).sort_values("SMD", ascending=False)
table1.to_csv(TABLES / "table1_baseline.csv", index=False)

# ------------------------------------------------------------------ procedure mix
PROC_LABELS = {
    "corpectomy_vertebral_body_resection": "Corpectomy / vertebral body resection",
    "intraspinal_tumor_excision": "Intraspinal tumor excision",
    "decompression_laminectomy": "Decompression / laminectomy",
    "arthrodesis_fusion": "Arthrodesis / fusion",
    "partial_vertebral_excision_osteotomy": "Partial vertebral excision / osteotomy",
    "instrumentation_or_unlisted": "Instrumentation / unlisted spine", "other_spine": "Other spine"}
mix = c.groupby("PROC_CATEGORY").agg(n=(OUTCOME, "size"), rec=(OUTCOME, "mean")).reset_index()
mix["Procedure"] = mix["PROC_CATEGORY"].map(lambda v: PROC_LABELS.get(v, v))
mix["% of cohort"] = (100 * mix["n"] / n).round(1)
mix["Optimal recovery %"] = (100 * mix["rec"]).round(1)
mix.sort_values("n", ascending=False)[["Procedure", "n", "% of cohort", "Optimal recovery %"]].to_csv(
    TABLES / "table2_procedure_mix.csv", index=False)

# ------------------------------------------------------------------ univariate screen (full cohort)
uni = []
for p in model_predictors:
    sub = analytic[[p, OUTCOME]].dropna()
    try:
        m = sm.Logit(sub[OUTCOME], sm.add_constant(sub[p].astype(float))).fit(disp=0)
        ci = m.conf_int()
        uni.append({"Predictor": p, "OR": np.exp(m.params[p]), "ci_lo": np.exp(ci.loc[p, 0]),
                    "ci_hi": np.exp(ci.loc[p, 1]), "p": m.pvalues[p]})
    except Exception:
        uni.append({"Predictor": p, "OR": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan})
uni_df = pd.DataFrame(uni).sort_values("p")
uni_df["95% CI"] = uni_df.apply(fmt_ci, axis=1)
uni_df.to_csv(TABLES / "table3_univariate.csv", index=False)

# ------------------------------------------------------------------ elastic-net selection (training set)
imp = SimpleImputer(strategy="median")
Xtr_i, Xte_i = imp.fit_transform(X_train), imp.transform(X_test)
sc = StandardScaler()
Xtr_s = sc.fit_transform(Xtr_i)
en = enet(Xtr_s, y_train)
retained = [v for v, co in zip(model_predictors, en.coef_[0]) if co != 0]
dropped = [v for v in model_predictors if v not in retained]
res["enet_retained"], res["enet_dropped"] = retained, dropped
log(f"Elastic net retained {len(retained)}/{len(model_predictors)}; dropped {dropped}")

# ------------------------------------------------------------------ multivariable (complete training cases)
tr = analytic.loc[X_train.index]
multi, multi_df, cc_n = logit_table(tr, OUTCOME, retained)
res["complete_cases_n"] = int(cc_n)
res["complete_cases_pct"] = cc_n / len(X_train)
multi_df = multi_df.sort_values("aOR")
multi_df["95% CI"] = multi_df.apply(fmt_ci, axis=1)
multi_df.to_csv(TABLES / "table4_multivariable.csv", index=False)
res["multivariable"] = multi_df[["Predictor", "aOR", "ci_lo", "ci_hi", "p"]].rename(
    columns={"Predictor": "predictor"}).to_dict("records")
log(f"Multivariable on {cc_n} complete training cases ({res['complete_cases_pct']:.1%})")
fit_vars = [v for v in multi.params.index if v != "const"]  # model column order

# ------------------------------------------------------------------ supporting model fit (held-out)
Xte_df = pd.DataFrame(Xte_i, columns=model_predictors, index=X_test.index)
proba = multi.predict(sm.add_constant(Xte_df[fit_vars].astype(float), has_constant="add"))
auc = roc_auc_score(y_test, proba)
rng = np.random.default_rng(RNG_SEED)
boot = []
for _ in range(1000):
    idx = rng.integers(0, len(y_test), len(y_test))
    if y_test.iloc[idx].nunique() > 1:
        boot.append(roc_auc_score(y_test.iloc[idx], proba.iloc[idx]))
lo, hi = np.percentile(boot, [2.5, 97.5])
pc = np.clip(np.asarray(proba, float), 1e-6, 1 - 1e-6)
lp = np.log(pc / (1 - pc))
slope = float(sm.Logit(y_test.values, sm.add_constant(lp)).fit(disp=0).params[1])
res["model_fit"] = {"auc": float(auc), "auc_ci_lo": float(lo), "auc_ci_hi": float(hi),
                    "calibration_slope": slope, "oe_ratio": float(y_test.sum() / proba.sum()),
                    "brier": float(brier_score_loss(y_test, proba))}
log(f"Supporting fit: AUC {auc:.3f} ({lo:.3f}-{hi:.3f}), slope {slope:.2f}")


def split_stability(n_seeds=200):
    aucs = []
    for seed in range(n_seeds):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
        i2 = SimpleImputer(strategy="median")
        s2 = StandardScaler()
        a, b = s2.fit_transform(i2.fit_transform(Xtr)), s2.transform(i2.transform(Xte))
        m = enet(a, ytr, seed=seed, Cs=10, cv=5, max_iter=5000)
        aucs.append(roc_auc_score(yte, m.predict_proba(b)[:, 1]))
    a = np.array(aucs)
    return {"median": float(np.median(a)), "p2_5": float(np.percentile(a, 2.5)),
            "p97_5": float(np.percentile(a, 97.5)), "n": len(a)}


ss = split_stability()
res["model_fit"].update({"auc_median_200": ss["median"], "auc_p2_5": ss["p2_5"], "auc_p97_5": ss["p97_5"]})
log(f"200-split median AUC {ss['median']:.3f} ({ss['p2_5']:.3f}-{ss['p97_5']:.3f})")

# ------------------------------------------------------------------ sensitivity: imputation handling
def headline_row(df, outcome, cols, label, extra=None):
    try:
        _, t, _ = logit_table(df, outcome, cols)
        t = t.set_index("Predictor")
        row = {"Analysis": label}
        for h in HEADLINE:
            if h in t.index:
                r = t.loc[h]
                row[h] = f"{r['aOR']:.2f} ({r['ci_lo']:.2f}-{r['ci_hi']:.2f})"
        row.update(extra or {})
        return row
    except Exception as e:
        return {"Analysis": label, "note": f"fit failed: {e}"}


imp_rows = []
for label, strat, fill in [("Median imputation (primary)", "median", None), ("Zero imputation", "constant", 0),
                           ("Minimum-value imputation", "constant", X_train[retained].min().min()),
                           ("Maximum-value imputation", "constant", X_train[retained].max().max())]:
    i3 = SimpleImputer(strategy=strat, fill_value=fill) if strat == "constant" else SimpleImputer(strategy=strat)
    Xa = pd.DataFrame(i3.fit_transform(X_train[retained]), columns=retained, index=X_train.index)
    Xb = pd.DataFrame(i3.transform(X_test[retained]), columns=retained, index=X_test.index)
    m = sm.Logit(y_train, sm.add_constant(Xa.astype(float))).fit(disp=0, method="newton", maxiter=200)
    pr = m.predict(sm.add_constant(Xb.astype(float), has_constant="add"))
    ci = m.conf_int()
    row = {"Analysis": label}
    for h in HEADLINE:  # odds ratios are only interpretable for the median rule; the crude
        if h in retained and strat == "median":  # single-value rules are reported by test AUC alone
            row[h] = f"{np.exp(m.params[h]):.2f} ({np.exp(ci.loc[h, 0]):.2f}-{np.exp(ci.loc[h, 1]):.2f})"
    row["Test AUC"] = round(roc_auc_score(y_test, pr), 3)
    imp_rows.append(row)
pd.DataFrame(imp_rows).to_csv(TABLES / "table6_imputation_sensitivity.csv", index=False)

# ------------------------------------------------------------------ sensitivity: other analyses
full = median_impute(c, retained)  # full cohort with median-imputed retained predictors
sens = {}
rows = []


def add(label, df, outcome=OUTCOME, extra=None):
    r = headline_row(df, outcome, retained, label, extra)
    rows.append(r)


add("Primary definition, full cohort (median-imputed)", full, extra={"n": n, "Recovery rate": f"{full[OUTCOME].mean():.1%}"})
for lab, yrs in [("Operated 2016-2019", range(2016, 2020)), ("Operated 2020-2021", range(2020, 2022))]:
    s = full[full["year_of_surgery"].isin(list(yrs))]
    add(lab, s,
        extra={"n": len(s), "Recovery rate": f"{s[OUTCOME].mean():.1%}"})
add("Recovery without the discharge criterion", full, "recovery_no_discharge_criterion",
    {"n": n, "Recovery rate": f"{full['recovery_no_discharge_criterion'].mean():.1%}"})
add("Recovery without transfusion in the complication criterion", full, "optimal_recovery_no_transfusion",
    {"n": n, "Recovery rate": f"{full['optimal_recovery_no_transfusion'].mean():.1%}"})
p75 = c["TOTHLOS_n"].quantile(0.75)
full["recovery_los75"] = ((full["mortality_30d"] == 0) & (full["major_complication"] == 0) &
                          (full["unplanned_reoperation"] == 0) & (full["unplanned_readmission"] == 0) &
                          (full["TOTHLOS_n"] <= p75) & (full["home_discharge"] == 1)).astype(int)
add(f"Length-of-stay criterion at the 75th percentile ({p75:.0f} days)", full, "recovery_los75",
    {"n": n, "Recovery rate": f"{full['recovery_los75'].mean():.1%}"})
el = full[(full["year_of_surgery"] <= 2020) & (full["elective"] == 1)]
ur = full[(full["year_of_surgery"] <= 2020) & (full["elective"] == 0)]
add("Elective operations only (2016-2020)", el, extra={"n": len(el), "Recovery rate": f"{el[OUTCOME].mean():.1%}"})
add("Urgent or emergent operations only (2016-2020)", ur, extra={"n": len(ur), "Recovery rate": f"{ur[OUTCOME].mean():.1%}"})
res["elective_rates"] = {"elective": float(el[OUTCOME].mean()), "urgent": float(ur[OUTCOME].mean()),
                         "n_elective": int(len(el)), "n_urgent": int(len(ur))}

cB, _, _ = cohort.load(ROOT / "data/cohort_broader.pkl", label="broader")
cB_full = median_impute(cB, retained)
add("Broader cohort (adds neoplastic pathologic fracture and myeloma)", cB_full,
    extra={"n": len(cB), "Recovery rate": f"{cB[OUTCOME].mean():.1%}"})
pd.DataFrame(rows).to_csv(TABLES / "table7_other_sensitivity.csv", index=False)
res["sensitivity"] = {r["Analysis"]: r for r in rows}
res["broader_cohort"] = {"n": int(len(cB)), "recovery_rate": float(cB[OUTCOME].mean())}

# ------------------------------------------------------------------ subgroup associations
sub_specs = [
    ("Procedure: intraspinal tumor excision", c["PROC_CATEGORY"] == "intraspinal_tumor_excision"),
    ("Procedure: arthrodesis / fusion", c["PROC_CATEGORY"] == "arthrodesis_fusion"),
    ("Procedure: corpectomy", c["PROC_CATEGORY"] == "corpectomy_vertebral_body_resection"),
    ("Procedure: decompression / laminectomy", c["PROC_CATEGORY"] == "decompression_laminectomy"),
    ("Specialty: neurosurgery", c["SURGSPEC"].astype(str).str.contains("Neuro", case=False)),
    ("Specialty: orthopedics", c["SURGSPEC"].astype(str).str.contains("Ortho", case=False)),
    ("Metastasis site: bone", c["MET_SITE"] == "bone"),
    ("Metastasis site: nervous system / cord", c["MET_SITE"] == "nervous_system_cord"),
    ("Elective (2016-2020)", (c["year_of_surgery"] <= 2020) & (c["elective"] == 1)),
    ("Urgent / emergent (2016-2020)", (c["year_of_surgery"] <= 2020) & (c["elective"] == 0))]
sub_rows = []
for name, mask in sub_specs:
    s = full[mask.values]
    ev = int(s[OUTCOME].sum())
    row = {"Subgroup": name, "n": len(s), "Recovery events": ev, "Recovery rate": f"{s[OUTCOME].mean():.1%}"}
    if ev >= 10 and (len(s) - ev) >= 10:
        keep = [v for v in ["preop_albumin", "preop_hematocrit", "asa_class"] if s[v].nunique() > 1]
        try:
            _, t, _ = logit_table(s, OUTCOME, keep)
            t = t.set_index("Predictor")
            for v in keep:
                r = t.loc[v]
                row[v] = f"{r['aOR']:.2f} ({r['ci_lo']:.2f}-{r['ci_hi']:.2f})"
        except Exception:
            for v in ["preop_albumin", "preop_hematocrit", "asa_class"]:
                row[v] = "not estimable"
    else:
        for v in ["preop_albumin", "preop_hematocrit", "asa_class"]:
            row[v] = "too sparse"
    sub_rows.append(row)
pd.DataFrame(sub_rows).to_csv(TABLES / "table8_subgroup_associations.csv", index=False)

# ------------------------------------------------------------------ supplement tables
miss = pd.DataFrame([{"Variable": p, "N missing": int(analytic[p].isna().sum()),
                      "% missing": round(100 * analytic[p].isna().mean(), 1)} for p in PREDICTORS])
miss.sort_values("% missing", ascending=False).to_csv(SUPP / "missingness.csv", index=False)

cpt = c.assign(CPT_s=c["CPT"].astype(str).str.strip()).groupby(["PROC_CATEGORY", "CPT_s"]).size().reset_index(name="n")
cpt["Category"] = cpt["PROC_CATEGORY"].map(L.CAT_LABEL).fillna(cpt["PROC_CATEGORY"])
cpt["Description"] = cpt["CPT_s"].map(L.CPT_DESC).fillna("")
cpt["order"] = cpt["PROC_CATEGORY"].map({k: i for i, k in enumerate(L.CAT_ORDER)})
cpt = cpt.sort_values(["order", "n"], ascending=[True, False]).rename(columns={"CPT_s": "CPT"})
cpt[["Category", "CPT", "Description", "n"]].to_csv(SUPP / "cpt_codes.csv", index=False)

icd = c["PODIAG10"].astype(str).str.strip().value_counts().rename_axis("ICD-10").reset_index(name="n")
icd["Description"] = icd["ICD-10"].map(L.ICD_DESC).fillna("")
icd[["ICD-10", "Description", "n"]].to_csv(SUPP / "icd10_codes.csv", index=False)

# ------------------------------------------------------------------ write
c.to_pickle(ROOT / "data/analytic.pkl")
(ROOT / "outputs/results.json").write_text(json.dumps(res, indent=2, default=float))
(LOGS / "analysis_log.txt").write_text("\n".join(LOG_LINES) + "\n")
log("Done.")
