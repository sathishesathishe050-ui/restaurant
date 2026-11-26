import pandas as pd
import pytest
from app.model import prepare_data, train_model, make_forecast

def sample_data():
    rows = []
    for day in pd.date_range("2026-01-01", periods=10):
        for hour in [10, 12, 19, 20]:
            rows.append({
                "order_date": day.strftime("%Y-%m-%d"),
                "order_time": f"{hour:02d}:00",
                "item": "Dosa",
                "quantity": 3 if hour in [12, 19] else 1
            })
    return pd.DataFrame(rows)

def test_preprocessing_creates_hourly_rows():
    cleaned, item_map = prepare_data(sample_data())
    assert len(cleaned) == 10 * 24
    assert item_map == {"Dosa": 0}
    assert cleaned["quantity"].notna().all()

def test_missing_required_column_is_rejected():
    with pytest.raises(ValueError, match="Missing required columns"):
        prepare_data(pd.DataFrame({"order_date": ["2026-01-01"]}))

def test_training_and_forecast(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics = train_model(sample_data())
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    rows = make_forecast("2026-01-11", "2026-01-11", "Dosa")
    assert len(rows) == 24
    assert all(row["predicted_quantity"] >= 0 for row in rows)

def test_unknown_menu_item_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_model(sample_data())
    with pytest.raises(ValueError, match="Unknown item"):
        make_forecast("2026-01-11", "2026-01-11", "Pizza")
