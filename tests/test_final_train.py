"""Tests for final model training."""

from datetime import timedelta

import pandas as pd
import pytest

from forecasting.final_train import train_final_model
from forecasting.model_io import load_model


@pytest.fixture
def sample_data(tmp_path):
    """Create a small synthetic sales dataset."""
    dates = pd.date_range("2023-01-01", periods=150, freq="D")

    data = pd.DataFrame(
        {
            "date": dates,
            "store": 1,
            "item": 1,
            "sales": [20 + i % 7 for i in range(len(dates))],
        }
    )

    data_path = tmp_path / "train.csv"
    data.to_csv(data_path, index=False)

    return data_path, dates[-1].to_pydatetime()


def test_final_model_saved(sample_data, tmp_path):
    """Check that final training saves a valid model."""
    data_path, _ = sample_data
    model_path = tmp_path / "final_model.joblib"

    train_final_model(data_path, model_path)

    assert model_path.exists()
    assert load_model(model_path) is not None


def test_final_forecast_starts_after_training(sample_data, tmp_path):
    """Check forecast dates follow the full training history."""
    data_path, last_date = sample_data
    model_path = tmp_path / "final_model.joblib"

    train_final_model(data_path, model_path)

    model = load_model(model_path)
    predictions = model.predict(h=7)

    assert len(predictions) == 7

    expected_date = last_date.date() + timedelta(days=1)
    actual_date = pd.Timestamp(predictions["ds"].min()).date()

    assert actual_date == expected_date


def test_final_training_invalid_max_series(sample_data, tmp_path):
    """Check invalid series limit is rejected."""
    data_path, _ = sample_data
    model_path = tmp_path / "final_model.joblib"

    with pytest.raises(ValueError, match="max_series must be positive"):
        train_final_model(data_path, model_path, max_series=0)
