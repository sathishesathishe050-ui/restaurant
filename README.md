# PRJ-070 — Restaurant Demand Predictor API

A beginner-friendly forecasting project using FastAPI, Pandas, scikit-learn and Streamlit.

## Features
- Upload historical order CSV data
- Train a Random Forest regression model
- Forecast item demand by date and hour
- Identify predicted rush hours
- Show MAE, RMSE and R² metrics
- Streamlit dashboard and CSV export

## 1. Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Start the API
```bash
uvicorn app.main:app --reload
```
API docs: http://127.0.0.1:8000/docs

## 3. Start the dashboard (new terminal)
```bash
streamlit run dashboard.py
```

## 4. Sample dataset
Use `data/sample_orders.csv`. Required columns:
- `order_date`: YYYY-MM-DD
- `order_time`: HH:MM (24-hour)
- `item`: menu item name
- `quantity`: number of units ordered

Optional: `order_id`.

## API endpoints
- `GET /health`
- `POST /train` — upload CSV as multipart field `file`
- `GET /items`
- `GET /metrics`
- `GET /forecast?start_date=2026-12-01&end_date=2026-12-07&item=Chicken%20Biryani`

## Notes
This is an educational baseline, not a production-grade restaurant forecasting system. Forecast quality depends on representative historical data. The model predicts hourly quantity for each menu item using calendar features and recent historical lag features. For a real deployment, monitor data drift, retrain regularly, and avoid using future information in training features.

## Run automated tests
```bash
pytest -q
```
The included tests cover preprocessing, missing-column validation, model training, forecast generation and unknown menu-item validation.
