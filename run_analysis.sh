#!/bin/bash
# Run the analysis stages. Requires data/cohort.pkl and data/cohort_broader.pkl (not distributed; see README).
set -e
cd "$(dirname "$0")/code"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -u analysis.py
python -u components.py
python -u mice_sensitivity.py
cd .. && python -m pytest tests -q
