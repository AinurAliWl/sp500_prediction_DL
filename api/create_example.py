"""Create a request from the latest engineered observations in the CSV."""
import argparse
import json
from pathlib import Path

from src.data_loader import PROJECT_ROOT, DEFAULT_DATA_PATH, read_data
from src.features import create_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "models/returns_mlflow_v1")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "examples/predict_request.json")
    args = parser.parse_args()
    config = json.loads((args.artifacts / "config.json").read_text())
    raw = read_data(args.data)
    df = create_features(raw)
    if len(df) < config["lookback"] or df.index[-1] != raw.index[-1]:
        raise ValueError("Not enough valid history, or the latest observation has invalid features.")
    payload = {
        "as_of": df.index[-1].date().isoformat(),
        "last_price": float(df["SP500"].iloc[-1]),
        "feature_columns": config["feature_columns"],
        "features": df[config["feature_columns"]].iloc[-config["lookback"]:].values.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Request saved: {args.output}")
    print(f"Shape: ({len(payload['features'])}, {len(payload['feature_columns'])})")


if __name__ == "__main__":
    main()
