import joblib
import pandas as pd
import pytest
from mlforecast import MLForecast
from sklearn.ensemble import RandomForestRegressor

from forecasting.model_io import load_model, save_model


def test_save_and_load_model(tmp_path):
    """Test saving and loading an MLForecast model."""
    model = MLForecast(
        models=[RandomForestRegressor(n_estimators=10, random_state=42)],
        freq="D",
        lags=[1, 7, 14],
        date_features=["dayofweek", "month"],
    )

    model_path = tmp_path / "models" / "forecast.joblib"

    save_model(model, model_path)

    assert model_path.is_file()

    loaded_model = load_model(model_path)

    assert isinstance(loaded_model, MLForecast)
    assert set(loaded_model.models) == set(model.models)
    assert loaded_model.ts.lags == model.ts.lags


def test_load_model_missing_file(tmp_path):
    """Test loading a model from a nonexistent file."""
    model_path = tmp_path / "missing.joblib"

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        load_model(model_path)


def test_load_model_invalid_object(tmp_path):
    """Test rejection of an invalid saved object."""
    model_path = tmp_path / "invalid.joblib"

    joblib.dump({"invalid": True}, model_path)

    with pytest.raises(TypeError, match="MLForecast"):
        load_model(model_path)


def test_trained_model_save_load_predict(tmp_path):
    """Check predictions after saving and loading a fitted model."""
    dates = pd.date_range("2023-01-01", periods=150, freq="D")

    data = pd.DataFrame(
        {
            "unique_id": ["store_1_item_1"] * len(dates),
            "ds": dates,
            "y": [float(20 + 5 * (i % 7) + i % 3) for i in range(len(dates))],
        }
    )

    model = MLForecast(
        models=[
            RandomForestRegressor(
                n_estimators=10,
                random_state=42,
                n_jobs=1,
            )
        ],
        freq="D",
        lags=[1, 7, 14],
        date_features=["dayofweek", "month"],
    )

    model.fit(data)

    predictions_before = model.predict(h=7)

    model_path = tmp_path / "trained_model.joblib"
    save_model(model, model_path)

    loaded_model = load_model(model_path)
    predictions_after = loaded_model.predict(h=7)

    pd.testing.assert_frame_equal(
        predictions_before,
        predictions_after,
    )
