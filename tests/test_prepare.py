from pathlib import Path

import pandas as pd

from data.prepare import run_prepare, temporal_split


def test_temporal_split_separates_reference_and_eval_windows():
    df = pd.DataFrame({
        "issue_d": ["Dec-2018", "Jan-2019", "Jun-2020", "Dec-2020", "Feb-2021"],
        "loan_amnt": [1, 2, 3, 4, 5],
    })
    reference, eval_ = temporal_split(df)
    assert list(reference["loan_amnt"]) == [1]
    assert list(eval_["loan_amnt"]) == [2, 3, 4]


def test_temporal_split_handles_already_parsed_datetime_column():
    df = pd.DataFrame({
        "issue_d": pd.to_datetime(["2018-11-01", "2019-03-01"]),
        "loan_amnt": [1, 2],
    })
    reference, eval_ = temporal_split(df)
    assert list(reference["loan_amnt"]) == [1]
    assert list(eval_["loan_amnt"]) == [2]


def test_run_prepare_end_to_end(tmp_path: Path):
    csv_path = tmp_path / "fixture.csv"
    csv_path.write_text(
        "id,loan_amnt,term,int_rate,grade,sub_grade,emp_length,home_ownership,annual_inc,"
        "verification_status,dti,revol_util,fico_range_low,fico_range_high,purpose,"
        "issue_d,loan_status,recoveries\n"
        "101,1000, 36 months,10.65%,B,B2,3 years,RENT,50000,Verified,15.0,30%,700,704,"
        "debt_consolidation,Dec-2018,Fully Paid,0.0\n"
        "102,2000, 60 months,15.0%,D,D1,10+ years,MORTGAGE,70000,Not Verified,20.0,50%,650,654,"
        "credit_card,Jun-2019,Charged Off,500.0\n"
        "103,3000, 36 months,9.0%,A,A1,< 1 year,OWN,90000,Verified,5.0,10%,720,724,"
        "other,Mar-2018,Current,0.0\n"
    )
    output_dir = tmp_path / "processed"

    reference_df, eval_df = run_prepare(raw_csv_path=csv_path, output_dir=output_dir)

    assert len(reference_df) == 1
    assert len(eval_df) == 1
    assert "recoveries" not in reference_df.columns
    assert "default_flag" in reference_df.columns
    assert (output_dir / "reference.parquet").exists()
    assert (output_dir / "eval.parquet").exists()


def test_run_prepare_retains_loan_id(tmp_path: Path):
    csv_path = tmp_path / "fixture.csv"
    csv_path.write_text(
        "id,loan_amnt,term,int_rate,grade,sub_grade,emp_length,home_ownership,annual_inc,"
        "verification_status,dti,revol_util,fico_range_low,fico_range_high,purpose,"
        "issue_d,loan_status\n"
        "1001,1000, 36 months,10.65%,B,B2,3 years,RENT,50000,Verified,15.0,30%,700,704,"
        "debt_consolidation,Dec-2018,Fully Paid\n"
        "1002,2000, 60 months,15.0%,D,D1,10+ years,MORTGAGE,70000,Not Verified,20.0,50%,650,654,"
        "credit_card,Jun-2019,Charged Off\n"
    )
    output_dir = tmp_path / "processed"

    reference_df, eval_df = run_prepare(raw_csv_path=csv_path, output_dir=output_dir)

    assert list(reference_df["id"]) == [1001]
    assert list(eval_df["id"]) == [1002]
