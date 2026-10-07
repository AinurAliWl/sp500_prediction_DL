import json

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler

from api.main import app
from src.models import LSTMModel, TransformerModel


@pytest.fixture(params=["lstm", "transformer"])
def api_case(request, tmp_path, monkeypatch):
    """Real serialization and startup with temporary, untrained models."""
    rng = np.random.default_rng(123)
    torch.manual_seed(123)
    columns = [f"feature_{i}" for i in range(19)]
    feature_scaler = StandardScaler().fit(pd.DataFrame(
        rng.normal(5, 3, (100, 19)), columns=columns))
    target_scaler = StandardScaler().fit(rng.normal(0.002, 0.01, (100, 1)))
    name = request.param
    parameters = {"input_size": 19, "num_layers": 2, "dropout": 0.2, "horizon": 5}
    cls = LSTMModel if name == "lstm" else TransformerModel
    if name == "lstm":
        parameters["hidden_size"] = 64
    else:
        parameters.update(d_model=64, nhead=4, lookback=60)
    original = cls(**parameters).eval()
    torch.save(original.state_dict(), tmp_path / f"{name}.pt")
    joblib.dump(feature_scaler, tmp_path / "feature_scaler.joblib")
    joblib.dump(target_scaler, tmp_path / "target_scaler.joblib")
    config = {"selected_model": name, "lookback": 60, "horizon": 5,
              "input_size": 19, "feature_columns": columns,
              "models": {name: parameters}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setenv("SP500_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.delenv("SP500_MODEL", raising=False)
    payload = {"as_of": "2023-12-29", "last_price": 4769.83,
               "feature_columns": columns,
               "features": rng.normal(4, 2, (60, 19)).tolist()}
    with TestClient(app) as client:
        yield client, payload, original, feature_scaler, target_scaler, name


@pytest.fixture
def synthetic_csv(tmp_path):
    """Deterministic source data with a distribution shift after row 300."""
    n = 500
    t = np.arange(n)
    shift = np.where(t >= 300, 30.0, 0.0)
    frame = pd.DataFrame({
        "SP500": 3000 + 2 * t + 10 * np.sin(t / 9),
        "DGS10": 2 + t / 1000 + shift / 100,
        "UNRATE": 4 + 0.1 * np.sin(t / 17),
        "CPIAUCSL": 250 + t / 20 + shift,
        "DCOILWTICO": 60 + 3 * np.sin(t / 11),
        "NASDAQCOM": 8000 + 4 * t + 50 * np.sin(t / 13),
    }, index=pd.bdate_range("2020-01-01", periods=n))
    path = tmp_path / "data.csv"
    frame.to_csv(path)
    return path
