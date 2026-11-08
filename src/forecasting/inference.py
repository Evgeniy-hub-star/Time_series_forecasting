"""Generate forecasts using a saved MLForecast model."""

import argparse
from pathlib import Path

import pandas as pd

from forecasting.model_io import load_model
from forecasting.predict import predict_model


def run_inference(
    model_path: str | Path,
    horizon: int = 90,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load a trained model and generate forecasts."""
    model = load_model(model_path)

    predictions = predict_model(model, horizon=horizon)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)

    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate forecasts using a saved model"
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=90)
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    predictions = run_inference(
        model_path=args.model_path,
        horizon=args.horizon,
        output_path=args.output,
    )

    print(f"Generated {len(predictions)} forecast rows")


if __name__ == "__main__":
    main()
