import numpy as np
import pandas as pd
import pytest

from forecasting.model_io import load_model
from forecasting.pipeline import run_pipeline


@pytest.fixture
def sample_csv(tmp_path):
    """Create a small synthetic sales dataset."""
    dates = pd.date_range("2023-01-01", periods=160, freq="D")

    data = pd.DataFrame(
        {
            "date": list(dates),
            "store": [1] * len(dates),
            "item": [1] * len(dates),
            "sales": [float(20 + 5 * (i % 7) + i % 3) for i in range(len(dates))],
        }
    )

    file_path = tmp_path / "sales.csv"
    data.to_csv(file_path, index=False)

    return file_path


def test_pipeline_success(sample_csv, monkeypatch):
    """Test the complete forecasting pipeline."""
    from forecasting import pipeline
    from forecasting.train import create_model

    def fast_create_model():
        forecast = create_model()
        forecast.models["RandomForestRegressor"].set_params(
            n_estimators=5,
            n_jobs=1,
        )
        return forecast

    monkeypatch.setattr(
        pipeline,
        "create_model",
        fast_create_model,
    )

    metrics = run_pipeline(
        data_path=str(sample_csv),
        horizon=7,
        max_series=1,
    )

    assert set(metrics) == {"MAE", "RMSE", "MAPE"}
    assert all(np.isfinite(value) for value in metrics.values())
    assert all(value >= 0 for value in metrics.values())


def test_pipeline_invalid_horizon(sample_csv):
    """Reject a non-positive forecast horizon."""
    with pytest.raises(ValueError, match="Horizon must be positive"):
        run_pipeline(
            data_path=str(sample_csv),
            horizon=0,
        )


def test_pipeline_insufficient_history(tmp_path):
    """Reject series with insufficient training history."""
    dates = pd.date_range("2023-01-01", periods=30, freq="D")

    data = pd.DataFrame(
        {
            "date": dates,
            "store": [1] * 30,
            "item": [1] * 30,
            "sales": [20.0] * 30,
        }
    )

    file_path = tmp_path / "short_sales.csv"
    data.to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="118 training rows"):
        run_pipeline(
            data_path=str(file_path),
            horizon=7,
        )


def test_pipeline_saves_model(tmp_path):
    import pandas as pd

    from forecasting.pipeline import run_pipeline

    dates = pd.date_range("2023-01-01", periods=220, freq="D")

    data = pd.DataFrame(
        {
            "date": dates,
            "store": 1,
            "item": 1,
            "sales": [20 + i % 7 for i in range(len(dates))],
        }
    )

    data_path = tmp_path / "train.csv"
    model_path = tmp_path / "model.joblib"

    data.to_csv(data_path, index=False)

    metrics = run_pipeline(
        data_path=str(data_path),
        horizon=7,
        model_path=model_path,
    )

    assert model_path.exists()

    loaded_model = load_model(model_path)

    assert loaded_model is not None
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "MAPE" in metrics
