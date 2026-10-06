import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import cohort


def test_counts_reproduce_v8():
    c, flow, med = cohort.load()
    assert len(c) == 2953
    assert int(c["optimal_recovery"].sum()) == 789
    assert med == 8.0
    assert [n for _, n in flow] == [3038, 3038, 3033, 2958, 2953]


def test_component_rates_match_v8():
    c, _, _ = cohort.load()
    assert abs(c["major_complication"].mean() - 0.4006) < 0.002
    assert abs(1 - c["los_met"].mean() - 0.4761) < 0.002
    assert abs(1 - c["home_discharge"].mean() - 0.3332) < 0.002
    assert abs(c["mortality_30d"].mean() - 0.0691) < 0.002


def test_sentinels_and_age_clean():
    c, _, _ = cohort.load()
    for col in ["preop_albumin", "preop_hematocrit", "preop_bun", "bmi"]:
        assert (c[col].dropna() > 0).all()
    assert c["age"].dtype.kind == "f" and c["age"].min() >= 18
    assert "operative_time" not in cohort.PREDICTORS and "copd" not in cohort.PREDICTORS
