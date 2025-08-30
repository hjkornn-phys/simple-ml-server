import os
import asyncio
import logging
from datetime import datetime, timedelta

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional

from .model_service import ModelService
from .logging_config import setup_logging

app = FastAPI(title="ML Server", version="0.1.0")
# Configure logging before obtaining the logger
setup_logging()
logger = logging.getLogger("ml_server")

model_service = ModelService()


class PredictRequest(BaseModel):
    features: List[List[float]] = Field(
        ..., description="2D array of features (n_samples x n_features)"
    )


class PredictResponse(BaseModel):
    predictions: List[float]


class TrainRequest(BaseModel):
    data_path: Optional[str] = Field(
        default=None,
        description="Path to CSV training data. If omitted, uses TRAIN_DATA_PATH env or data/train.csv.",
    )


@app.on_event("startup")
def startup_event() -> None:
    # Ensure model is available at startup
    model_service.load_or_train()

    # Schedule daily training at 02:00 AM local time, unless disabled
    if os.getenv("DISABLE_SCHEDULER") == "1":
        logger.info("Scheduler disabled via DISABLE_SCHEDULER=1")
    else:
        try:
            asyncio.get_event_loop().create_task(_schedule_daily_training())
            logger.info("Scheduled daily training at 02:00 AM local time")
        except RuntimeError:
            # In some server setups there may be no running loop at import time
            pass


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    preds = model_service.predict(req.features)
    return PredictResponse(predictions=preds)


@app.post("/train")
def train(req: TrainRequest) -> dict:
    data_path = req.data_path or os.getenv("TRAIN_DATA_PATH") or "data/train.csv"
    model_path = model_service.train_from_file(data_path)
    return {"status": "trained", "model_path": model_path}


async def _schedule_daily_training() -> None:
    """Background task that trains daily at 02:00 AM local time."""
    while True:
        now = datetime.now().astimezone()
        today_run = now.replace(hour=2, minute=0, second=0, microsecond=0)
        next_run = today_run if now < today_run else today_run + timedelta(days=1)
        wait_seconds = (next_run - now).total_seconds()
        logger.info(f"Next scheduled training at {next_run.isoformat()}")
        await asyncio.sleep(wait_seconds)
        try:
            data_path = os.getenv("TRAIN_DATA_PATH") or "data/train.csv"
            model_path = model_service.train_from_file(data_path)
            logger.info(f"Scheduled training completed. Saved model at {model_path}")
        except Exception as e:
            logger.exception(f"Scheduled training failed: {e}")
