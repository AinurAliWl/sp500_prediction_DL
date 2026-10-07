"""Create untrained artifacts for container checks, never forecasting quality."""
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import LSTMModel, TransformerModel


def main():
    directory = ROOT / "models/returns_mlflow_v1"
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise RuntimeError("CI bundle directory must be empty; existing artifacts are protected.")
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    columns = [f"feature_{i}" for i in range(19)]
    feature_scaler = StandardScaler().fit(pd.DataFrame(rng.normal(size=(100, 19)), columns=columns))
    target_scaler = StandardScaler().fit(rng.normal(0, 0.01, size=(100, 1)))
    parameters = {
        "lstm": {"input_size": 19, "hidden_size": 64, "num_layers": 2, "dropout": 0.2, "horizon": 5},
        "transformer": {"input_size": 19, "d_model": 64, "nhead": 4, "num_layers": 2,
                        "dropout": 0.2, "horizon": 5, "lookback": 60},
    }
    for name, cls in (("lstm", LSTMModel), ("transformer", TransformerModel)):
        torch.save(cls(**parameters[name]).state_dict(), directory / f"{name}.pt")
    joblib.dump(feature_scaler, directory / "feature_scaler.joblib")
    joblib.dump(target_scaler, directory / "target_scaler.joblib")
    config = {"selected_model": "lstm", "lookback": 60, "horizon": 5, "input_size": 19,
              "feature_columns": columns, "models": parameters, "purpose": "ci_untrained_smoke_check"}
    (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    payload = {"as_of": "2023-12-29", "last_price": 100.0, "feature_columns": columns,
               "features": rng.normal(size=(60, 19)).tolist()}
    (ROOT / "ci_predict_request.json").write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    print("Created untrained CI artifacts; do not use for real forecasts.")


if __name__ == "__main__":
    main()
