import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data_loader import PROJECT_ROOT
from .predict import load_artifacts, predict, returns_to_prices

def calculate_metrics(
    actual,
    predicted,
):
    actual_flat = actual.reshape(-1)
    predicted_flat = predicted.reshape(-1)

    mae = mean_absolute_error(
        actual_flat,
        predicted_flat,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_flat,
            predicted_flat,
        )
    )

    mape = np.mean(
        np.abs(
            (
                actual_flat
                - predicted_flat
            )
            / actual_flat
        )
    ) * 100

    r2 = r2_score(
        actual_flat,
        predicted_flat,
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
    }

def evaluate_artifacts(artifact_dir):
    artifact_dir = Path(artifact_dir)
    with np.load(artifact_dir / "test_data.npz") as test:
        X = test["X"]
        actual = test["target_prices"]
        last_prices = test["last_prices"]
    predictions = {"Naive": np.repeat(last_prices[:, None], actual.shape[1], axis=1)}
    for name in ("lstm", "transformer"):
        model, _, target_scaler, _ = load_artifacts(artifact_dir, name)
        returns = predict(model, X, target_scaler)
        predictions[name.title()] = returns_to_prices(last_prices, returns)
    overall = pd.DataFrame({
        name: calculate_metrics(actual, values)
        for name, values in predictions.items()
    }).T
    rows = []
    for day in range(actual.shape[1]):
        for name, values in predictions.items():
            rows.append({"Model": name, "Day": day+1,
                **calculate_metrics(actual[:, day], values[:, day])})
    by_horizon = pd.DataFrame(rows)
    overall.to_csv(artifact_dir / "metrics_overall.csv", index_label="Model")
    by_horizon.to_csv(artifact_dir / "metrics_by_horizon.csv", index=False)
    print(overall.round(4).to_string())
    print(by_horizon.round(4).to_string(index=False))
    return overall, by_horizon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path,
                        default=PROJECT_ROOT / "models" / "returns_v1")
    args = parser.parse_args()
    evaluate_artifacts(args.artifacts)


if __name__ == "__main__":
    main()
