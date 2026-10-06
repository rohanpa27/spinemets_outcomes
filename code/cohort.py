"""
Cohort, outcome, and predictor derivation for the optimal recovery study
(surgically treated spinal metastases, ACS-NSQIP 2016-2021).

Single source of truth for inclusion/exclusion and variable construction.
  load()   -> (analytic DataFrame, flow list, median LOS)
  strobe() -> DataFrame of the inclusion flow (also written to outputs/tables)

NSQIP encoding notes:
  * Complication columns store the event label (for example "Pneumonia"), not "Yes".
    A complication is positive when the value is not in the missing/none set.
  * 30-day mortality = DOPERTOD in 0..30 or DISCHDEST == "Expired".
  * Home discharge is matched case-insensitively on "home".
  * -99 is the missing sentinel for height, weight, length of stay, and labs.
  * AGE is text because 90 and over is stored as "90+".
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SENTINEL = -99
EXCLUDE_TXT = {"", "nan", "unknown", "null", "none"}

MAJOR_COMP_COLS = ["SUPINFEC", "WNDINFD", "ORGSPCSSI", "DEHIS", "OUPNEUMO",
                   "REINTUB", "PULEMBOL", "FAILWEAN", "OPRENAFL", "URNINFEC",
                   "CNSCVA", "CDARREST", "CDMI", "OTHDVT", "OTHSYSEP",
                   "OTHSESHOCK", "OTHCDIFF"]
TRANSFUSION_COL = "OTHBLEED"

LAB_MAP = {"preop_wbc": "PRWBC", "preop_hematocrit": "PRHCT", "preop_albumin": "PRALBUM",
           "preop_creatinine": "PRCREAT", "preop_inr": "PRINR", "preop_sodium": "PRSODM",
           "preop_bun": "PRBUN"}

PREDICTORS = ["age", "female_sex", "hispanic", "bmi",
              "diabetes", "smoker", "hypertension", "chf",
              "dialysis", "chronic_steroid", "bleeding_disorder", "weight_loss",
              "renal_insufficiency", "preop_sepsis", "dyspnea", "preop_transfusion",
              "functional_status", "asa_class", "inpatient", "orthopedic_specialty",
              "preop_wbc", "preop_hematocrit", "preop_albumin",
              "preop_creatinine", "preop_inr", "preop_sodium", "preop_bun",
              "year_of_surgery"]

# display name -> failure indicator column (1 = the adverse event occurred)
COMPONENTS = {"30-day mortality": "mortality_30d",
              "Major complication": "major_complication",
              "Unplanned reoperation": "unplanned_reoperation",
              "Unplanned readmission": "unplanned_readmission",
              "Prolonged length of stay": "los_failed",
              "Non-home discharge": "home_failed"}

LOG = []
_FLOW_CACHE = {}


def _log(msg=""):
    LOG.append(str(msg))


def num(s):
    return pd.to_numeric(s.astype(str).str.strip().replace({"": np.nan}), errors="coerce")


def comp_positive(series):
    s = series.astype(str).str.strip().str.lower()
    return pd.Series(np.where(s.isin(EXCLUDE_TXT), np.nan,
                              (s != "no complication").astype(float)), index=series.index)


def yn(series):
    s = series.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out[s == "yes"] = 1.0
    out[s == "no"] = 0.0
    return out


def _recovery(c, comp):
    return ((c["mortality_30d"] == 0) & (c[comp] == 0) &
            (c["unplanned_reoperation"] == 0) & (c["unplanned_readmission"] == 0) &
            (c["los_met"] == 1) & (c["home_discharge"] == 1)).astype(int)


def load(path=None, label="Spine CPT with secondary-malignancy diagnosis, 2016-2021"):
    path = Path(path) if path else ROOT / "data" / "cohort.pkl"
    raw = pd.read_pickle(path)
    raw.columns = [c.strip().upper() for c in raw.columns]  # neutralize the 2019 case quirk
    c = raw.copy()
    c["AGE_n"] = pd.to_numeric(
        c["AGE"].astype(str).str.replace("+", "", regex=False).str.strip().replace({"": np.nan}),
        errors="coerce")
    c["ASA_num"] = pd.to_numeric(c["ASACLAS"].astype(str).str.strip().str[0], errors="coerce")
    c["TOTHLOS_n"] = num(c["TOTHLOS"]).replace(SENTINEL, np.nan)

    flow = [(label, len(c))]

    def step(mask, lbl):
        nonlocal c
        c = c[mask].copy()
        flow.append((lbl, len(c)))
        _log(f"After {lbl}: n = {len(c)}")

    step(c["AGE_n"] >= 18, "Adults aged 18 years or older")
    step(c["ASA_num"] != 5, "After excluding ASA class 5 (moribund)")
    step(c["TOTHLOS_n"].notna(), "With length of stay recorded")
    dd2 = c["DISCHDEST"].astype(str).str.strip().str.lower()
    step(~dd2.isin(EXCLUDE_TXT), "With discharge destination recorded (analytic cohort)")
    dd = c["DISCHDEST"].astype(str).str.strip()

    # ---- outcome components ----
    for col in set(MAJOR_COMP_COLS + [TRANSFUSION_COL]):
        c[col + "_pos"] = comp_positive(c[col])
    c["transfusion"] = c[TRANSFUSION_COL + "_pos"]
    comp_primary = MAJOR_COMP_COLS + [TRANSFUSION_COL]
    c["major_complication"] = c[[x + "_pos" for x in comp_primary]].max(axis=1).fillna(0).astype(int)
    c["major_complication_notxf"] = c[[x + "_pos" for x in MAJOR_COMP_COLS]].max(axis=1).fillna(0).astype(int)
    c["unplanned_reoperation"] = (c["REOPERATION1"].astype(str).str.strip().str.lower() == "yes").astype(int)
    c["unplanned_readmission"] = (c["UNPLANNEDREADMISSION1"].astype(str).str.strip().str.lower() == "yes").astype(int)
    dtd = num(c["DOPERTOD"]).replace(SENTINEL, np.nan)
    c["mortality_30d"] = (((dtd >= 0) & (dtd <= 30)) | (dd.str.lower() == "expired")).astype(int)
    median_los = float(c["TOTHLOS_n"].median())
    c["los_met"] = (c["TOTHLOS_n"] <= median_los).astype(int)
    c["home_discharge"] = dd.str.upper().str.contains("HOME").astype(int)
    c["los_failed"] = 1 - c["los_met"]
    c["home_failed"] = 1 - c["home_discharge"]

    c["optimal_recovery"] = _recovery(c, "major_complication")
    c["optimal_recovery_no_transfusion"] = _recovery(c, "major_complication_notxf")
    c["recovery_no_discharge_criterion"] = ((c["mortality_30d"] == 0) & (c["major_complication"] == 0) &
                                            (c["unplanned_reoperation"] == 0) &
                                            (c["unplanned_readmission"] == 0) &
                                            (c["los_met"] == 1)).astype(int)

    # ---- candidate factors ----
    c["age"] = c["AGE_n"].astype(float)
    c["female_sex"] = (c["SEX"].astype(str).str.strip().str.lower() == "female").astype(int)
    c["hispanic"] = (c["ETHNICITY_HISPANIC"].astype(str).str.strip().str.lower() == "yes").astype(int)
    H = num(c["HEIGHT"]).replace(SENTINEL, np.nan)
    W = num(c["WEIGHT"]).replace(SENTINEL, np.nan)
    c["bmi"] = np.where((H > 0) & (W > 0), 703.0 * W / (H ** 2), np.nan)
    c["diabetes"] = (c["DIABETES"].astype(str).str.strip().str.lower() != "no").astype(int)
    c["smoker"] = yn(c["SMOKE"])
    c["hypertension"] = yn(c["HYPERMED"])
    c["chf"] = yn(c["HXCHF"])
    c["dialysis"] = yn(c["DIALYSIS"])
    c["chronic_steroid"] = yn(c["STEROID"])
    c["bleeding_disorder"] = yn(c["BLEEDDIS"])
    c["weight_loss"] = yn(c["WTLOSS"])
    c["renal_insufficiency"] = yn(c["RENAFAIL"])
    c["preop_sepsis"] = (c["PRSEPIS"].astype(str).str.strip().str.lower()
                         .isin(["sirs", "sepsis", "septic shock"])).astype(int)
    c["dyspnea"] = (c["DYSPNEA"].astype(str).str.strip().str.lower()
                    .isin(["moderate exertion", "at rest"])).astype(int)
    c["preop_transfusion"] = yn(c["TRANSFUS"])
    fn = c["FNSTATUS2"].astype(str).str.strip().str.lower()
    c["functional_status"] = np.where(fn == "independent", 0.0,
                              np.where(fn == "partially dependent", 1.0,
                              np.where(fn == "totally dependent", 2.0, np.nan)))
    c["asa_class"] = c["ASA_num"]
    c["inpatient"] = (c["INOUT"].astype(str).str.strip().str.lower() == "inpatient").astype(int)
    c["orthopedic_specialty"] = (c["SURGSPEC"].astype(str).str.upper().str.contains("ORTHO")).astype(int)
    for new, src in LAB_MAP.items():
        c[new] = num(c[src]).replace(SENTINEL, np.nan)
    c["year_of_surgery"] = num(c["YEAR"])
    es = c["ELECTSURG"].astype(str).str.strip().str.lower()
    c["elective"] = np.where(es == "yes", 1.0, np.where(es == "no", 0.0, np.nan))  # NaN in 2021

    _FLOW_CACHE[str(path)] = flow
    return c, flow, median_los


def strobe():
    c, flow, _ = load()
    df = pd.DataFrame(flow, columns=["Step", "n"])
    out = ROOT / "outputs" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "strobe_flow.csv", index=False)
    (ROOT / "outputs" / "logs").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "logs" / "cohort.log").write_text("\n".join(LOG) + "\n")
    return df


if __name__ == "__main__":
    print(strobe().to_string(index=False))
