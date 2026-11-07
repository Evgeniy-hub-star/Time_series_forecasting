"""Расчет единых метрик оценки качества прогноза"""

import numpy as np
import pandas as pd


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute MAE, RMSE and nonzero-target MAPE (percent)."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.ndim != 1 or predicted.ndim != 1 or actual.size != predicted.size:
        raise ValueError("Targets and predictions must be 1D arrays of equal length")
    if (
        actual.size == 0
        or not np.isfinite(actual).all()
        or not np.isfinite(predicted).all()
    ):
        raise ValueError("Targets and predictions must be nonempty and finite")
    errors = actual - predicted
    mask = actual != 0
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "MAPE": float(np.mean(np.abs(errors[mask] / actual[mask])) * 100)
        if mask.any()
        else float("nan"),
    }


def evaluate_forecast(
    actual: pd.DataFrame, forecast: pd.DataFrame, model_name: str
) -> dict[str, float]:
    """Align forecasts by series and date before evaluating."""
    keys = ["unique_id", "ds"]
    for frame, columns in (
        (actual, keys + ["y"]),
        (forecast, keys + [model_name]),
    ):
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate series/date keys")
    merged = actual[keys + ["y"]].merge(
        forecast[keys + [model_name]], on=keys, how="inner", validate="one_to_one"
    )
    if len(merged) != len(actual) or len(merged) != len(forecast):
        raise ValueError("Actual and forecast keys do not match exactly")
    return calculate_metrics(merged["y"], merged[model_name])
