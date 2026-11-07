import math

import pandas as pd
import pytest

from forecasting.evaluate import calculate_metrics, evaluate_forecast


def test_metrics_ignore_zero_targets_in_mape():
    result = calculate_metrics([0, 10], [2, 8])
    assert result["MAE"] == pytest.approx(2)
    assert result["RMSE"] == pytest.approx(2)
    assert result["MAPE"] == pytest.approx(20)


def test_all_zero_targets_yield_undefined_mape():
    assert math.isnan(calculate_metrics([0], [1])["MAPE"])


def test_evaluation_aligns_by_keys():
    actual = pd.DataFrame(
        {
            "unique_id": ["a", "b"],
            "ds": pd.to_datetime(["2020-01-01"] * 2),
            "y": [10, 20],
        }
    )
    forecast = pd.DataFrame(
        {
            "unique_id": ["b", "a"],
            "ds": pd.to_datetime(["2020-01-01"] * 2),
            "RF": [18, 9],
        }
    )
    assert evaluate_forecast(actual, forecast, "RF")["MAE"] == pytest.approx(1.5)
