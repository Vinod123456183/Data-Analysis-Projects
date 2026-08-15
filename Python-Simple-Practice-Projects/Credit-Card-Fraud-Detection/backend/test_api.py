

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

SAMPLE_TRANSACTION = {
    "Time": 406.0, "V1": -2.3122, "V2": 1.9519, "V3": -1.6098,
    "V4": 3.9979, "V5": -0.5222, "V6": -1.4265, "V7": -2.5373,
    "V8": 1.3917, "V9": -2.7700, "V10": -2.7722, "V11": 3.2020,
    "V12": -2.8999, "V13": -0.5952, "V14": -4.2892, "V15": 0.3898,
    "V16": -1.1407, "V17": -2.8300, "V18": -0.0168, "V19": 0.4169,
    "V20": 0.1269, "V21": 0.5172, "V22": -0.0350, "V23": -0.4652,
    "V24": 0.3202, "V25": 0.0445, "V26": 0.1780, "V27": 0.2611,
    "V28": -0.1433, "Amount": 0.0,
}


def test_root():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert "model_loaded" in res.json()


def test_predict_valid_transaction():
    res = client.post("/predict", json=SAMPLE_TRANSACTION)
    assert res.status_code == 200
    body = res.json()
    assert "is_fraud" in body
    assert "fraud_probability" in body
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["risk_level"] in {"Low", "Elevated", "Medium", "High"}


def test_predict_missing_field():
    bad_payload = {k: v for k, v in SAMPLE_TRANSACTION.items() if k != "V14"}
    res = client.post("/predict", json=bad_payload)
    assert res.status_code == 422  # Pydantic validation error


def test_predict_wrong_type():
    bad_payload = dict(SAMPLE_TRANSACTION)
    bad_payload["Amount"] = "not-a-number"
    res = client.post("/predict", json=bad_payload)
    assert res.status_code == 422


def test_predict_batch():
    payload = [SAMPLE_TRANSACTION, SAMPLE_TRANSACTION]
    res = client.post("/predict/batch", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["total_count"] == 2
    assert len(body["predictions"]) == 2


def test_predict_batch_empty():
    res = client.post("/predict/batch", json=[])
    assert res.status_code == 400


def test_predict_batch_too_large():
    payload = [SAMPLE_TRANSACTION] * 5001
    res = client.post("/predict/batch", json=payload)
    assert res.status_code == 400