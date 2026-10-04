import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

model_data = joblib.load("models/fraud_model.pkl")
model = model_data["model"]
encoder = model_data["encoder"]


def _run_inference(amount, country, hour, previous_txns):
    """Shared inference logic used by all routes."""
    df = pd.DataFrame([{
        "amount": amount,
        "country": country,
        "hour": hour,
        "previous_txns": previous_txns,
    }])
    df["country"] = encoder.transform(df["country"])
    return int(model.predict(df)[0])


@app.post("/predict")
def predict(transaction: dict):
    result = _run_inference(
        transaction["amount"],
        transaction["country"],
        transaction["hour"],
        transaction["previous_txns"],
    )
    return {"fraud": result}


# KServe v1 protocol — called by the transformer via predict()
@app.post("/v1/models/fraud-model:predict")
def v1_predict(request: dict):
    instances = request.get("instances", [request])
    instance = instances[0]
    result = _run_inference(
        instance["amount"],
        instance["country"],
        instance["hour"],
        instance["previous_txns"],
    )
    return {
        "id": "fraud-model",
        "model_name": "fraud-model",
        "outputs": [
            {
                "name": "fraud",
                "datatype": "INT64",
                "shape": [1],
                "data": [result],
            }
        ],
    }


# KServe v2 protocol — open inference protocol
@app.post("/v2/models/fraud-model/infer")
def v2_infer(request: dict):
    inputs = request["inputs"][0]
    data = inputs["data"]
    result = _run_inference(
        amount=data[0],
        country=data[1],
        hour=data[2],
        previous_txns=data[3],
    )
    return {
        "model_name": "fraud-model",
        "outputs": [
            {
                "name": "fraud",
                "datatype": "INT64",
                "shape": [1],
                "data": [result],
            }
        ],
    }