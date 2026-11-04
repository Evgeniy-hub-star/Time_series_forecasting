"""ПОДГОТОВКА И ВАЛИДАЦИЯ ВХОДЯЩИХ ДАННЫХ"""

import pandas as pd

REQUIRED_COLUMNS = ("store", "item", "date", "sales")


def prepare_forecast_format(data: pd.DataFrame) -> pd.DataFrame:
    """Convert raw store/item sales to sorted unique_id/ds/y records."""
    missing = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required columns contain missing values")

    result = data.loc[:, REQUIRED_COLUMNS].copy()
    result["ds"] = pd.to_datetime(result["date"], errors="raise")
    result["unique_id"] = (
        result["store"].astype(str) + "_" + result["item"].astype(str)
    )
    result = result.rename(columns={"sales": "y"})
    if not pd.api.types.is_numeric_dtype(result["y"]):
        raise ValueError("sales must be numeric")
    result = result[["unique_id", "ds", "y"]]
    if result.duplicated(["unique_id", "ds"]).any():
        raise ValueError("Duplicate unique_id/ds combinations")
    return result.sort_values(["unique_id", "ds"]).reset_index(drop=True)
