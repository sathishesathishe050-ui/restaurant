from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.model import train_model, load_artifact, make_forecast
import pandas as pd
from io import BytesIO

app = FastAPI(title="Restaurant Demand Predictor API", version="1.0.0",
              description="Train a model from restaurant order history and forecast menu demand.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def home():
    return {"message": "Restaurant Demand Predictor API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/train")
async def train(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    try:
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
        metrics = train_model(df)
        return {"message": "Model trained successfully", "metrics": metrics}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.get("/items")
def items():
    try:
        return {"items": list(load_artifact()["item_map"].keys())}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.get("/metrics")
def metrics():
    try:
        return load_artifact()["metrics"]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.get("/forecast")
def forecast(start_date: str, end_date: str, item: str | None = None):
    try:
        rows = make_forecast(start_date, end_date, item)
        return {"count": len(rows), "forecast": rows}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
