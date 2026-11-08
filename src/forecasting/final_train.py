"""Train a final forecasting model on all available data."""

import argparse
from pathlib import Path

import pandas as pd

from forecasting.data import prepare_forecast_format
from forecasting.model_io import save_model
from forecasting.train import create_model, train_model


def train_final_model(
    data_path: str | Path,
    model_path: str | Path,
    max_series: int | None = None,
) -> None:
    """Train on the complete dataset and save the fitted model."""
    if max_series is not None and max_series <= 0:
        raise ValueError("max_series must be positive")

    raw_data = pd.read_csv(data_path)
    data = prepare_forecast_format(raw_data)

    if max_series is not None:
        series_ids = data["unique_id"].unique()[:max_series]
        data = data[data["unique_id"].isin(series_ids)].copy()

    counts = data.groupby("unique_id").size()
    if counts.empty or (counts <= 118).any():
        raise ValueError("Each series needs more than 118 rows")

    model = create_model()
    model = train_model(data, model)

    save_model(model, model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train final forecasting model")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--max-series", type=int, default=None)

    args = parser.parse_args()

    train_final_model(
        data_path=args.data,
        model_path=args.model_path,
        max_series=args.max_series,
    )

    print(f"Final model saved to: {args.model_path}")


if __name__ == "__main__":
    main()
