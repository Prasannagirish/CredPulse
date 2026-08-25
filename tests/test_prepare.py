import pandas as pd

from data.prepare import temporal_split


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
