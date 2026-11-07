"""Run an end-to-end sales forecasting pipeline."""

import argparse
from pathlib import Path

import pandas as pd

from forecasting.data import prepare_forecast_format
from forecasting.evaluate import evaluate_forecast
from forecasting.predict import predict_model
from forecasting.train import create_model, train_model


def run_pipeline(
    data_path: str,
    horizon: int = 90,
    max_series: int | None = None,
) -> dict[str, float]:
    """Train, predict and evaluate a Random Forest model."""
    if horizon <= 0:
        raise ValueError("Horizon must be positive")
    if max_series is not None and max_series <= 0:
        raise ValueError("max_series must be positive")

    raw_data = pd.read_csv(data_path)
    data = prepare_forecast_format(raw_data)

    if max_series is not None:
        series_ids = data["unique_id"].unique()[:max_series]
        data = data[data["unique_id"].isin(series_ids)].copy()

    # Use the final horizon observations of each series for testing.
    data = data.sort_values(["unique_id", "ds"])
    test = data.groupby("unique_id", group_keys=False).tail(horizon)
    train = data.drop(index=test.index)

    # Require sufficient history for lag features.
    counts = train.groupby("unique_id").size()
    if counts.empty or (counts <= 118).any():
        raise ValueError("Each series needs more than 118 training rows")

    forecast = create_model()
    forecast = train_model(train, forecast)

    predictions = predict_model(forecast, horizon=horizon)
    metrics = evaluate_forecast(
        actual=test,
        forecast=predictions,
        model_name="RandomForestRegressor",
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random Forest sales forecasting pipeline"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=90)
    parser.add_argument("--max-series", type=int, default=None)

    args = parser.parse_args()

    metrics = run_pipeline(
        data_path=str(args.data),
        horizon=args.horizon,
        max_series=args.max_series,
    )

    print("Forecast evaluation:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
