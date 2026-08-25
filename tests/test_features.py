import numpy as np
import pandas as pd
import pytest

from model.features import build_preprocessor, clean_raw_types, derive_target, select_features


def test_derive_target_marks_charged_off_and_default_positive():
    df = pd.DataFrame({
        "loan_status": ["Charged Off", "Default", "Fully Paid"],
        "loan_amnt": [1000, 2000, 3000],
    })
    result = derive_target(df)
    labels = dict(zip(result["loan_amnt"], result["default_flag"]))
    assert labels[1000] == 1
    assert labels[2000] == 1
    assert labels[3000] == 0


def test_derive_target_drops_non_terminal_statuses():
    df = pd.DataFrame({
        "loan_status": [
            "Current", "Late (31-120 days)", "In Grace Period",
            "Late (16-30 days)", "Fully Paid", "Charged Off",
        ],
        "loan_amnt": [1, 2, 3, 4, 5, 6],
    })
    result = derive_target(df)
    assert set(result["loan_amnt"]) == {5, 6}


def test_derive_target_drops_does_not_meet_credit_policy_rows():
    df = pd.DataFrame({
        "loan_status": [
            "Does not meet the credit policy. Status:Fully Paid",
            "Does not meet the credit policy. Status:Charged Off",
            "Fully Paid",
        ],
        "loan_amnt": [1, 2, 3],
    })
    result = derive_target(df)
    assert set(result["loan_amnt"]) == {3}


def test_select_features_keeps_only_allow_listed_columns():
    df = pd.DataFrame({
        "loan_amnt": [1000],
        "term": [" 36 months"],
        "int_rate": ["10.65%"],
        "grade": ["B"],
        "sub_grade": ["B2"],
        "emp_length": ["3 years"],
        "home_ownership": ["RENT"],
        "annual_inc": [50000.0],
        "verification_status": ["Verified"],
        "dti": [15.0],
        "revol_util": ["30%"],
        "fico_range_low": [700],
        "fico_range_high": [704],
        "purpose": ["debt_consolidation"],
        "default_flag": [0],
        "recoveries": [123.0],  # post-origination leak column — must be dropped
        "total_pymnt": [5000.0],  # post-origination leak column — must be dropped
    })
    result = select_features(df)
    assert "recoveries" not in result.columns
    assert "total_pymnt" not in result.columns
    assert "default_flag" in result.columns
    assert set(result.columns) - {"default_flag"} == set(
        ["loan_amnt", "term", "int_rate", "grade", "sub_grade", "emp_length",
         "home_ownership", "annual_inc", "verification_status", "dti", "revol_util",
         "fico_range_low", "fico_range_high", "purpose"]
    )


def test_clean_raw_types_parses_term_rate_util_emp_length():
    df = pd.DataFrame({
        "term": [" 36 months", " 60 months"],
        "int_rate": ["10.65%", "7.2%"],
        "revol_util": ["30%", None],
        "emp_length": ["3 years", "10+ years"],
    })
    result = clean_raw_types(df)
    assert list(result["term"]) == [36, 60]
    assert result["int_rate"].iloc[0] == pytest.approx(10.65)
    assert result["revol_util"].iloc[0] == pytest.approx(30.0)
    assert np.isnan(result["revol_util"].iloc[1])
    assert result["emp_length"].iloc[0] == 3
    assert result["emp_length"].iloc[1] == 10


def test_build_preprocessor_fits_and_transforms():
    df = pd.DataFrame({
        "loan_amnt": [1000, 2000, 3000],
        "term": [36, 60, 36],
        "int_rate": [10.0, 12.0, 9.0],
        "emp_length": [1, 5, 10],
        "annual_inc": [40000.0, 60000.0, 80000.0],
        "dti": [10.0, 20.0, 15.0],
        "revol_util": [30.0, 40.0, 50.0],
        "fico_range_low": [700, 710, 690],
        "fico_range_high": [704, 714, 694],
        "grade": ["A", "B", "C"],
        "sub_grade": ["A1", "B2", "C3"],
        "home_ownership": ["RENT", "OWN", "MORTGAGE"],
        "verification_status": ["Verified", "Not Verified", "Verified"],
        "purpose": ["debt_consolidation", "credit_card", "other"],
    })
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == 3
