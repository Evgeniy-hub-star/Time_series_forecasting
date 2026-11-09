from pathlib import Path

import joblib
from mlforecast import MLForecast


def save_model(model: MLForecast, path: str | Path) -> None:
    """Save a trained forecasting model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)


def load_model(path: str | Path) -> MLForecast:
    """Load a previously saved forecasting model."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    model = joblib.load(path)

    if not isinstance(model, MLForecast):
        raise TypeError("Loaded object is not an MLForecast model")

    return model
