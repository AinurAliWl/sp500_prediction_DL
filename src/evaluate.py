import argparse
import json
import warnings

from mlflow import MlflowClient
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

def log_evaluation(artifact_dir, overall, by_horizon):
    config = json.loads((artifact_dir / "config.json").read_text())
    run_ids = config.get("mlflow_run_ids")
    if not run_ids:
        warnings.warn(
            "This artifact directory has no MLflow run IDs. "
            "Metrics were saved locally, but were not logged to MLflow."
        )
        return

    # Use the local project's database, even if the project folder was moved.
    tracking_uri = "sqlite:///" + (PROJECT_ROOT / "mlflow.db").as_posix()
    if not (PROJECT_ROOT / "mlflow.db").exists():
        raise FileNotFoundError(
            "MLflow database is missing. Local metric files were saved. "
            "Restore this training run's mlflow.db and retry."
        )
    client = MlflowClient(tracking_uri=tracking_uri)

    # Resolve both runs before writing any metrics; never create replacement runs.
    for name in ("lstm", "transformer"):
        run = client.get_run(run_ids[name])
        if run.info.lifecycle_stage != "active":
            raise ValueError(f"The {name} MLflow run is deleted; restore it first.")

    baseline = overall.loc["Naive"]
    for name in ("lstm", "transformer"):
        run_id = run_ids[name]
        label = name.title()
        metrics = {
            f"test_{key.lower()}": float(value)
            for key, value in overall.loc[label].items()
        }
        metrics.update({
            f"naive_test_{key.lower()}": float(value)
            for key, value in baseline.items()
        })
        for model_label, prefix in ((label, "test"), ("Naive", "naive_test")):
            for _, row in by_horizon[by_horizon["Model"] == model_label].iterrows():
                day = int(row["Day"])
                for key in ("MAE", "RMSE", "MAPE", "R2"):
                    metrics[f"{prefix}_day_{day}_{key.lower()}"] = float(row[key])
        baseline_rmse = float(baseline["RMSE"])
        if baseline_rmse > 0:
            metrics["test_rmse_improvement_pct"] = (
                100 * (baseline_rmse - float(overall.loc[label, "RMSE"]))
                / baseline_rmse
            )
        for key, value in metrics.items():
            client.log_metric(run_id, key, value)
        client.set_tag(run_id, "evaluation_scale", "index_level")
        client.set_tag(run_id, "mape_units", "percent")
        for filename in ("metrics_overall.csv", "metrics_by_horizon.csv"):
            client.log_artifact(run_id, str(artifact_dir / filename), "evaluation")
        client.log_artifact(run_id, __file__, "evaluation/source")
        print(f"Evaluation logged to MLflow ({name}): {run_id}")


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
    log_evaluation(artifact_dir, overall, by_horizon)
    return overall, by_horizon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path,
                        default=PROJECT_ROOT / "models" / "returns_v1")
    args = parser.parse_args()
    evaluate_artifacts(args.artifacts)


if __name__ == "__main__":
    main()
