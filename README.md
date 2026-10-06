# Spine metastasis surgery outcomes (ACS-NSQIP 2016-2021): analysis code

Analysis code for a retrospective study of preoperative factors associated with optimal recovery after surgery
for spinal metastases. Optimal recovery is an all-or-none composite of no 30-day mortality, no major complication,
no unplanned reoperation, no unplanned readmission, length of stay at or below the cohort median, and discharge home.

## Contents

| File | Purpose |
|---|---|
| `code/cohort.py` | Inclusion and exclusion, outcome and component derivation, candidate factor construction |
| `code/analysis.py` | Baseline table, univariate screen, elastic-net selection, multivariable model, sensitivity and subgroup analyses |
| `code/components.py` | Separate models for each recovery criterion with false discovery rate correction |
| `code/mice_sensitivity.py` | Full-cohort multiple imputation (20 imputations, Rubin pooling) |
| `code/code_labels.py` | CPT and ICD-10 description lookup for the code tables |
| `tests/test_cohort.py` | Checks on cohort counts and variable cleaning |
| `run_analysis.sh` | Runs the stages in order |

## Data

No data are included. The American College of Surgeons NSQIP Participant Use Data Files are released to participating
institutions under a Data Use Agreement and may not be redistributed. The code expects two pickled DataFrames in `data/`
built from the 2016-2021 files, `cohort.pkl` and `cohort_broader.pkl`, with the native PUF columns plus `YEAR`, `MET_SITE`,
and `PROC_CATEGORY` (and `DX_GROUP` in the broader file).

- Primary cohort: principal spine CPT (22010 to 22899 or 63001 to 63746) with a principal postoperative ICD-10 diagnosis
  beginning C79 or C7B.
- Broader sensitivity cohort: the primary cohort plus M84.5x (neoplastic pathologic fracture) and C90.x (myeloma).

Outputs are written to `outputs/` and are ignored by git.

## Requirements

Python 3 with pandas, numpy, scipy, scikit-learn, statsmodels, and pytest. The analysis stage takes about ten minutes.

## License

MIT, see `LICENSE`.
