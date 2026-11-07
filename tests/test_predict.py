"""Tests for the forecasting prediction module."""

import numpy as np
import pandas as pd
import pytest

from forecasting.predict import predict_model
from forecasting.train import create_model, train_model


@pytest.fixture
def trained_forecast():
    """Train a lightweight model on synthetic daily sales."""
    dates = pd.date_range("2023-01-01", periods=150, freq="D")

    data = pd.DataFrame(
        {
            "unique_id": ["store_1"] * len(dates),
            "ds": dates,
            "y": [float(20 + 5 * (i % 7) + i % 3) for i in range(len(dates))],
        }
    )

    forecast = create_model()

    # Use only two trees to keep tests fast.
    forecast.models["RandomForestRegressor"].set_params(
        n_estimators=2,
        n_jobs=1,
    )

    return train_model(data, forecast)


def test_predict_90_days(trained_forecast):
    """Check that a 90-day forecast has the expected structure."""
    predictions = predict_model(trained_forecast, horizon=90)

    assert len(predictions) == 90
    assert list(predictions.columns) == [
        "unique_id",
        "ds",
        "RandomForestRegressor",
    ]
    assert predictions["unique_id"].nunique() == 1
    assert predictions["ds"].min() == pd.Timestamp("2023-05-31")
    assert predictions["ds"].max() == pd.Timestamp("2023-08-28")

    assert predictions["RandomForestRegressor"].notna().all()
    assert np.isfinite(predictions["RandomForestRegressor"]).all()


def test_predict_invalid_horizon(trained_forecast):
    """Reject a non-positive forecast horizon."""
    with pytest.raises(ValueError, match="horizon"):
        predict_model(trained_forecast, horizon=0)


def test_predict_unfitted_model():
    """Reject prediction before model training."""
    forecast = create_model()

    with pytest.raises(ValueError, match="fitted"):
        predict_model(forecast, horizon=90)
