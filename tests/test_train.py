import pandas as pd

from forecasting.train import create_model, train_model


def test_create_model():
    forecast = create_model()

    assert forecast.freq == "D"
    assert "RandomForestRegressor" in forecast.models


def test_train_model():
    dates = pd.date_range("2023-01-01", periods=120, freq="D")

    data = pd.DataFrame(
        {
            "unique_id": ["store_1"] * 120,
            "ds": dates,
            "y": [float(10 + i % 7) for i in range(120)],
        }
    )

    forecast = create_model()

    # Reduce training time for the unit test.
    forecast.models["RandomForestRegressor"].set_params(
        n_estimators=2,
        n_jobs=1,
    )

    trained = train_model(data, forecast)

    assert trained is forecast
    assert "RandomForestRegressor" in trained.models_
