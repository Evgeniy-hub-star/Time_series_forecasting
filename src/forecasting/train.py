import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
from sklearn.ensemble import RandomForestRegressor


def create_model() -> MLForecast:
    """Create the Random Forest forecasting pipeline."""
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    return MLForecast(
        models=[model],
        freq="D",
        lags=[1, 7, 14, 28, 90],
        lag_transforms={
            1: [
                RollingMean(window_size=7),
                RollingMean(window_size=28),
            ],
        },
        date_features=["dayofweek", "month", "year"],
    )


def train_model(
    train_data: pd.DataFrame,
    forecast: MLForecast | None = None,
) -> MLForecast:
    """Fit the forecasting model on prepared training data."""
    if forecast is None:
        forecast = create_model()

    forecast.fit(train_data)
    return forecast
