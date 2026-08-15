"""
Credit Card Fraud Detection API
--------------------------------
Serves predictions from a trained XGBoost model (fraud_model.pkl).

Run locally:
    uvicorn main:app --reload --port 8000

Docs available at:
    http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Predicts whether a credit card transaction is fraudulent.",
    version="1.0.0",
)

# Allow the frontend (served from a different origin) to call this API.
# In production, replace "*" with your actual frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load model artifacts at startup
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fraud_model.pkl")
THRESHOLD_PATH = os.path.join(os.path.dirname(__file__), "threshold.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

model = None
scaler = None
threshold = 0.5  # fallback default

# Training column order was: Time, V1..V28, Amount_log (Amount_log LAST, not Amount)
FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount_log"]


@app.on_event("startup")
def load_model():
    global model, threshold, scaler
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            "Export it from your notebook with joblib.dump(model, 'fraud_model.pkl') "
            "and place it in the backend/ folder."
        )
    model = joblib.load(MODEL_PATH)

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    else:
        raise RuntimeError(
            f"scaler.pkl not found at {SCALER_PATH}. This model was trained on scaled "
            "Time/Amount_log features and requires the scaler to make correct predictions."
        )

    if os.path.exists(THRESHOLD_PATH):
        threshold = float(joblib.load(THRESHOLD_PATH))
    print(f"Model loaded. Using decision threshold: {threshold}")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class Transaction(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since first transaction in dataset")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., description="Transaction amount")

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 406.0, "V1": -2.3122, "V2": 1.9519, "V3": -1.6098,
                "V4": 3.9979, "V5": -0.5222, "V6": -1.4265, "V7": -2.5373,
                "V8": 1.3917, "V9": -2.7700, "V10": -2.7722, "V11": 3.2020,
                "V12": -2.8999, "V13": -0.5952, "V14": -4.2892, "V15": 0.3898,
                "V16": -1.1407, "V17": -2.8300, "V18": -0.0168, "V19": 0.4169,
                "V20": 0.1269, "V21": 0.5172, "V22": -0.0350, "V23": -0.4652,
                "V24": 0.3202, "V25": 0.0445, "V26": 0.1780, "V27": 0.2611,
                "V28": -0.1433, "Amount": 0.0,
            }
        }


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    threshold_used: float
    risk_level: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    fraud_count: int
    total_count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def risk_level_from_probability(prob: float) -> str:
    if prob >= 0.75:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    elif prob >= threshold:
        return "Elevated"
    return "Low"


def build_prediction(prob: float) -> PredictionResponse:
    return PredictionResponse(
        is_fraud=bool(prob >= threshold),
        fraud_probability=round(float(prob), 6),
        threshold_used=round(float(threshold), 6),
        risk_level=risk_level_from_probability(prob),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Fraud detection API is running. See /docs for usage."}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


def preprocess_transaction(transaction: Transaction) -> np.ndarray:
    """Replicates notebook preprocessing: Amount -> Amount_log -> scale(Time, Amount_log)."""
    amount_log = np.log1p(transaction.Amount)
    scaled = scaler.transform([[transaction.Time, amount_log]])[0]
    scaled_time, scaled_amount_log = scaled[0], scaled[1]

    v_values = [getattr(transaction, f"V{i}") for i in range(1, 29)]
    row = [scaled_time] + v_values + [scaled_amount_log]
    return np.array(row).reshape(1, -1)


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded.")

    try:
        X = preprocess_transaction(transaction)
        prob = model.predict_proba(X)[0, 1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    return build_prediction(prob)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(transactions: list[Transaction]):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded.")
    if len(transactions) == 0:
        raise HTTPException(status_code=400, detail="No transactions provided.")
    if len(transactions) > 5000:
        raise HTTPException(status_code=400, detail="Batch too large (max 5000).")

    try:
        X = np.vstack([preprocess_transaction(t) for t in transactions])
        probs = model.predict_proba(X)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {e}")

    preds = [build_prediction(p) for p in probs]
    fraud_count = sum(p.is_fraud for p in preds)

    return BatchPredictionResponse(
        predictions=preds, fraud_count=fraud_count, total_count=len(preds)
    )