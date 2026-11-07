"""Generate forecasts using a trained MLForecast model."""

import pandas as pd
from mlforecast import MLForecast

DEFAULT_HORIZON = 90


def predict_model(
    forecast: MLForecast,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """Generate future forecasts for all trained series."""
    if horizon <= 0:
        raise ValueError("Forecast horizon must be positive")

    if not hasattr(forecast, "models_"):
        raise ValueError("Forecast model must be fitted before prediction")

    predictions = forecast.predict(h=horizon)

    required_columns = {
        "unique_id",
        "ds",
        "RandomForestRegressor",
    }

    missing = required_columns - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    return predictions.sort_values(["unique_id", "ds"]).reset_index(drop=True)
