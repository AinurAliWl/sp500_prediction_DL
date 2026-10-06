import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from .data_loader import DEFAULT_DATA_PATH, PROJECT_ROOT, read_data
from .features import create_features
from .models import LSTMModel, TransformerModel


def returns_to_prices(last_prices, predicted_returns):
    return np.asarray(last_prices)[:, None] * np.cumprod(
        1 + np.asarray(predicted_returns), axis=1)


def predict(model, X, target_scaler, device="cpu", batch_size=256):
    if len(X) == 0 or batch_size < 1:
        raise ValueError("Input must be nonempty and batch_size positive.")
    model = model.to(device)
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.as_tensor(X[start:start+batch_size],
                                    dtype=torch.float32, device=device)
            outputs.append(model(batch).cpu().numpy())
    scaled = np.concatenate(outputs)
    return target_scaler.inverse_transform(
        scaled.reshape(-1, 1)).reshape(scaled.shape)


def load_artifacts(artifact_dir, model_name=None, device="cpu"):
    artifact_dir = Path(artifact_dir)
    config = json.loads((artifact_dir / "config.json").read_text())
    model_name = model_name or config["selected_model"]
    classes = {"lstm": LSTMModel, "transformer": TransformerModel}
    if model_name not in classes:
        raise ValueError("Model must be lstm or transformer.")
    model = classes[model_name](**config["models"][model_name])
    model.load_state_dict(torch.load(artifact_dir / f"{model_name}.pt",
                                    map_location=device, weights_only=True))
    model.to(device).eval()
    feature_scaler = joblib.load(artifact_dir / "feature_scaler.joblib")
    target_scaler = joblib.load(artifact_dir / "target_scaler.joblib")
    return model, feature_scaler, target_scaler, config


def forecast_latest(raw_data, artifact_dir, model_name=None, device="cpu"):
    model, feature_scaler, target_scaler, config = load_artifacts(
        artifact_dir, model_name, device)
    df = create_features(raw_data)
    lookback = config["lookback"]
    if len(df) < lookback:
        raise ValueError("Need enough raw history for features plus lookback.")
    if df.index[-1] != raw_data.index[-1]:
        raise ValueError("Latest row has invalid features; cannot silently forecast from an older date.")
    X = feature_scaler.transform(
        df[config["feature_columns"]].iloc[-lookback:])[None, :, :]
    returns = predict(model, X, target_scaler, device)
    last_price = float(df["SP500"].iloc[-1])
    prices = returns_to_prices(np.array([last_price]), returns)[0]
    return {"model": model_name or config["selected_model"],
            "as_of": str(df.index[-1]), "last_price": last_price,
            "predicted_returns": returns[0].tolist(),
            "predicted_index": prices.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path,
                        default=PROJECT_ROOT / "models" / "returns_v1")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", choices=["lstm", "transformer"])
    args = parser.parse_args()
    print(json.dumps(forecast_latest(read_data(args.data),
                                    args.artifacts, args.model), indent=2))


if __name__ == "__main__":
    main()
