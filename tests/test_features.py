import pandas as pd

from model.features import derive_target


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
