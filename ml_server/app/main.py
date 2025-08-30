from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

from .model_service import ModelService

app = FastAPI(title="ML Server", version="0.1.0")

model_service = ModelService()


class PredictRequest(BaseModel):
    features: List[List[float]] = Field(
        ..., description="2D array of features (n_samples x n_features)"
    )


class PredictResponse(BaseModel):
    predictions: List[float]


@app.on_event("startup")
def startup_event() -> None:
    # Ensure model is available at startup
    model_service.load_or_train()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    preds = model_service.predict(req.features)
    return PredictResponse(predictions=preds)
