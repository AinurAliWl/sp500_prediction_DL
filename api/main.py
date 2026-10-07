import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from threading import Lock
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from src.data_loader import PROJECT_ROOT
from src.predict import load_artifacts, predict, returns_to_prices

FiniteNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]
PositiveNumber = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    as_of: date
    last_price: PositiveNumber
    feature_columns: list[str] = Field(min_length=1)
    features: list[list[FiniteNumber]] = Field(min_length=1, max_length=10000)


class ForecastResponse(BaseModel):
    model: str
    as_of: date
    last_price: float
    horizon: int
    horizon_unit: str
    predicted_returns: list[float]
    predicted_index: list[float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    directory = Path(os.getenv("SP500_ARTIFACT_DIR", "models/returns_mlflow_v1"))
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    name = os.getenv("SP500_MODEL") or None
    model, feature_scaler, target_scaler, config = load_artifacts(directory, name)
    app.state.bundle = (model, feature_scaler, target_scaler, config)
    app.state.model_name = name or config["selected_model"]
    app.state.inference_lock = Lock()
    yield
    del app.state.bundle


app = FastAPI(
    title="S&P 500 Forecast API",
    version="0.1.0",
    description="Five-horizon forecasts from unscaled engineered features.",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "bundle"),
            "model": request.app.state.model_name}


@app.get("/metadata")
def metadata(request: Request):
    config = request.app.state.bundle[3]
    return {
        "model": request.app.state.model_name,
        "lookback": config["lookback"],
        "input_size": config["input_size"],
        "feature_columns": config["feature_columns"],
        "horizon": config["horizon"],
        "horizon_unit": "retained_trading_observations",
        "input_scale": "unscaled_engineered_features",
    }


@app.post("/predict", response_model=ForecastResponse)
def forecast(payload: ForecastRequest, request: Request):
    model, feature_scaler, target_scaler, config = request.app.state.bundle
    columns = config["feature_columns"]
    if payload.feature_columns != columns:
        raise HTTPException(422, "feature_columns must match /metadata exactly, including order.")
    if len(payload.features) != config["lookback"]:
        raise HTTPException(422, f"Expected exactly {config['lookback']} observations.")
    if any(len(row) != len(columns) for row in payload.features):
        raise HTTPException(422, f"Each observation must contain {len(columns)} features.")

    frame = pd.DataFrame(payload.features, columns=columns)
    X = feature_scaler.transform(frame)[None, :, :]
    if not np.isfinite(X).all() or np.abs(X).max() > np.finfo(np.float32).max:
        raise HTTPException(422, "Features are outside the supported numeric range.")
    with request.app.state.inference_lock:
        returns = predict(model, X, target_scaler)
    with np.errstate(over="ignore", invalid="ignore"):
        prices = returns_to_prices(np.array([payload.last_price]), returns)
    if not np.isfinite(returns).all() or not np.isfinite(prices).all():
        raise HTTPException(500, "Model produced non-finite predictions.")
    return ForecastResponse(
        model=request.app.state.model_name,
        as_of=payload.as_of,
        last_price=payload.last_price,
        horizon=config["horizon"],
        horizon_unit="retained_trading_observations",
        predicted_returns=returns[0].tolist(),
        predicted_index=prices[0].tolist(),
    )
