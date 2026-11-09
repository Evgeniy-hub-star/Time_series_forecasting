"""Tests for inference using a saved forecasting model."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from forecasting.inference import run_inference
from forecasting.model_io import save_model
from forecasting.train import create_model, train_model


@pytest.fixture
def saved_model(tmp_path):
    """Create and save a fitted forecasting model."""
    dates = pd.date_range("2023-01-01", periods=150, freq="D")

    data = pd.DataFrame(
        {
            "unique_id": ["store_1_item_1"] * len(dates),
            "ds": dates,
            "y": [float(20 + i % 7) for i in range(len(dates))],
        }
    )

    model = create_model()
    model = train_model(data, model)

    model_path = tmp_path / "model.joblib"
    save_model(model, model_path)

    return model_path, model


def test_inference_returns_predictions(saved_model):
    """Check forecast structure and horizon."""
    model_path, _ = saved_model

    predictions = run_inference(model_path, horizon=7)

    assert len(predictions) == 7
    assert "unique_id" in predictions.columns
    assert "ds" in predictions.columns


def test_inference_matches_original_model(saved_model):
    """Check predictions are identical after loading."""
    model_path, original_model = saved_model

    expected = original_model.predict(h=7)
    actual = run_inference(model_path, horizon=7)

    assert_frame_equal(expected, actual)


def test_inference_saves_csv(saved_model, tmp_path):
    """Check forecast CSV is created correctly."""
    model_path, _ = saved_model
    output_path = tmp_path / "predictions" / "forecast.csv"

    predictions = run_inference(
        model_path,
        horizon=7,
        output_path=output_path,
    )

    assert output_path.exists()

    saved_predictions = pd.read_csv(output_path)

    assert len(saved_predictions) == len(predictions)
    assert list(saved_predictions.columns) == list(predictions.columns)


def test_inference_missing_model(tmp_path):
    """Check missing model produces an informative error."""
    missing_path = tmp_path / "missing.joblib"

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        run_inference(missing_path, horizon=7)
