import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

MODEL_PATH = os.path.join("artifacts", "demand_model.joblib")
FEATURES = ["item_code", "hour", "day_of_week", "day_of_month", "month", "is_weekend"]

def prepare_data(df: pd.DataFrame):
    required = {"order_date", "order_time", "item", "quantity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    data = df.copy()
    data["order_date"] = pd.to_datetime(data["order_date"], errors="coerce")
    data["order_time"] = pd.to_datetime(data["order_time"], format="%H:%M", errors="coerce")
    data["quantity"] = pd.to_numeric(data["quantity"], errors="coerce")
    data["item"] = data["item"].astype(str).str.strip()
    data = data.dropna(subset=["order_date", "order_time", "quantity"])
    data = data[(data["quantity"] >= 0) & (data["item"] != "")]
    if data.empty:
        raise ValueError("No valid rows found after cleaning.")
    data["hour"] = data["order_time"].dt.hour
    data["date"] = data["order_date"].dt.normalize()
    hourly = (data.groupby(["date", "item", "hour"], as_index=False)["quantity"]
              .sum().sort_values(["item", "date", "hour"]))
    # Include zero-order hours so the model learns quiet periods.
    all_dates = pd.date_range(hourly["date"].min(), hourly["date"].max(), freq="D")
    items = sorted(hourly["item"].unique().tolist())
    full = pd.MultiIndex.from_product([all_dates, items, range(24)],
                                      names=["date", "item", "hour"]).to_frame(index=False)
    full = full.merge(hourly, on=["date", "item", "hour"], how="left")
    full["quantity"] = full["quantity"].fillna(0)
    full["day_of_week"] = full["date"].dt.dayofweek
    full["day_of_month"] = full["date"].dt.day
    full["month"] = full["date"].dt.month
    full["is_weekend"] = (full["day_of_week"] >= 5).astype(int)
    full["item_code"], item_names = pd.factorize(full["item"], sort=True)
    return full, {str(name): int(i) for i, name in enumerate(item_names)}

def train_model(df: pd.DataFrame):
    full, item_map = prepare_data(df)
    if len(full) < 48:
        raise ValueError("Please provide at least 2 days of valid order history.")
    X, y = full[FEATURES], full["quantity"]
    # Chronological holdout: last 20% of unique dates are validation data.
    dates = sorted(full["date"].unique())
    split_index = max(1, int(len(dates) * 0.8))
    cutoff = dates[min(split_index, len(dates)-1)]
    train_mask = full["date"] < cutoff
    if train_mask.sum() < 24 or (~train_mask).sum() == 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test = X[train_mask], X[~train_mask]
        y_train, y_test = y[train_mask], y[~train_mask]
    model = RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    predictions = np.maximum(model.predict(X_test), 0)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 3),
        "rmse": round(rmse, 3),
        "r2": round(float(r2_score(y_test, predictions)), 3) if len(y_test) > 1 else None,
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_test)),
        "items": list(item_map.keys()),
        "history_start": str(full["date"].min().date()),
        "history_end": str(full["date"].max().date())
    }
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump({"model": model, "item_map": item_map, "metrics": metrics}, MODEL_PATH)
    return metrics

def load_artifact():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not trained yet. Upload order history to /train first.")
    return joblib.load(MODEL_PATH)

def make_forecast(start_date: str, end_date: str, item: str | None = None):
    artifact = load_artifact()
    model, item_map = artifact["model"], artifact["item_map"]
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date.")
    if (end - start).days > 90:
        raise ValueError("Please request a maximum of 90 days at a time.")
    selected = [item] if item else list(item_map.keys())
    unknown = [name for name in selected if name not in item_map]
    if unknown:
        raise ValueError(f"Unknown item(s): {', '.join(unknown)}. Available items: {', '.join(item_map)}")
    rows = []
    for date in pd.date_range(start, end, freq="D"):
        for name in selected:
            for hour in range(24):
                dow = date.dayofweek
                features = pd.DataFrame([{
                    "item_code": item_map[name], "hour": hour, "day_of_week": dow,
                    "day_of_month": date.day, "month": date.month,
                    "is_weekend": int(dow >= 5)
                }], columns=FEATURES)
                qty = max(0.0, float(model.predict(features)[0]))
                rows.append({"date": date.date().isoformat(), "item": name,
                             "hour": hour, "predicted_quantity": round(qty, 2)})
    return rows
