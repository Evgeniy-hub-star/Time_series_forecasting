import pandas as pd
import pytest

from forecasting.data import prepare_forecast_format


def test_prepare_forecast_format_sorts_and_renames():
    raw = pd.DataFrame(
        {
            "store": [2, 1],
            "item": [3, 3],
            "date": ["2017-01-02", "2017-01-01"],
            "sales": [12, 8],
        }
    )
    result = prepare_forecast_format(raw)
    assert result.columns.tolist() == ["unique_id", "ds", "y"]
    assert result["unique_id"].tolist() == ["1_3", "2_3"]
    assert pd.api.types.is_datetime64_any_dtype(result["ds"])


def test_prepare_rejects_missing_column():
    with pytest.raises(ValueError, match="Missing required columns"):
        prepare_forecast_format(pd.DataFrame({"store": [1]}))


def test_prepare_rejects_duplicate_keys():
    raw = pd.DataFrame(
        {
            "store": [1, 1],
            "item": [2, 2],
            "date": ["2017-01-01", "2017-01-01"],
            "sales": [5, 6],
        }
    )
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_forecast_format(raw)
