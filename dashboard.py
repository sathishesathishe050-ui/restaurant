import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Restaurant Demand Predictor", page_icon="🍽️", layout="wide")
API = st.sidebar.text_input("API URL", "http://127.0.0.1:8000").rstrip("/")
st.title("🍽️ Restaurant Demand Predictor")
st.caption("Forecast menu-item demand and restaurant rush hours using machine learning.")

with st.sidebar:
    st.header("1. Upload order history")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    if st.button("Train ML Model", use_container_width=True):
        if uploaded is None:
            st.warning("Upload a CSV file first.")
        else:
            try:
                response = requests.post(f"{API}/train",
                    files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")}, timeout=180)
                if response.ok:
                    st.success("Model trained successfully!")
                    st.json(response.json()["metrics"])
                else:
                    st.error(response.json().get("detail", response.text))
            except requests.RequestException as e:
                st.error(f"Cannot connect to API: {e}")

try:
    health = requests.get(f"{API}/health", timeout=5)
    st.sidebar.success("API connected" if health.ok else "API unavailable")
except requests.RequestException:
    st.sidebar.error("Start the API first: uvicorn app.main:app --reload")

c1, c2, c3 = st.columns(3)
try:
    m = requests.get(f"{API}/metrics", timeout=5)
    if m.ok:
        metrics = m.json()
        c1.metric("MAE", metrics["mae"])
        c2.metric("RMSE", metrics["rmse"])
        c3.metric("R²", metrics["r2"] if metrics["r2"] is not None else "N/A")
        st.caption(f"Training rows: {metrics['training_rows']} · Validation rows: {metrics['validation_rows']} · History: {metrics['history_start']} to {metrics['history_end']}")
    else:
        st.info("Train a model using the sidebar to view metrics and forecasts.")
except requests.RequestException:
    st.warning("API is not reachable.")

st.header("2. Generate a date-range forecast")
try:
    item_response = requests.get(f"{API}/items", timeout=5)
    item_options = ["All menu items"] + item_response.json().get("items", []) if item_response.ok else ["All menu items"]
except requests.RequestException:
    item_options = ["All menu items"]

left, middle, right = st.columns([1, 1, 1])
start = left.date_input("Start date")
end = middle.date_input("End date", value=start)
selected = right.selectbox("Menu item", item_options)
if st.button("Generate Forecast", type="primary"):
    if end < start:
        st.error("End date must be on or after start date.")
    else:
        params = {"start_date": str(start), "end_date": str(end)}
        if selected != "All menu items":
            params["item"] = selected
        try:
            response = requests.get(f"{API}/forecast", params=params, timeout=60)
            if response.ok:
                result = pd.DataFrame(response.json()["forecast"])
                st.session_state["forecast"] = result
            else:
                st.error(response.json().get("detail", response.text))
        except requests.RequestException as e:
            st.error(f"Forecast request failed: {e}")

if "forecast" in st.session_state:
    result = st.session_state["forecast"]
    st.subheader("Forecast results")
    daily = result.groupby(["date", "item"], as_index=False)["predicted_quantity"].sum()
    st.dataframe(daily, use_container_width=True)
    st.subheader("Predicted demand by hour")
    hourly = result.groupby("hour", as_index=False)["predicted_quantity"].sum()
    fig, ax = plt.subplots()
    ax.bar(hourly["hour"], hourly["predicted_quantity"])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Predicted quantity")
    ax.set_xticks(range(0, 24, 2))
    st.pyplot(fig)
    peak = hourly.loc[hourly["predicted_quantity"].idxmax()]
    st.success(f"Predicted busiest hour: {int(peak['hour']):02d}:00")
    st.download_button("Download forecast CSV", result.to_csv(index=False),
                       file_name="restaurant_forecast.csv", mime="text/csv")
