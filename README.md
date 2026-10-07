# S&P 500 Forecasting with LSTM, Transformer and MLOps

A PyTorch forecasting project with experiment tracking, an HTTP inference API, automated tests, and Docker serving.

The goal is to build a reproducible workflow from historical market data to a running forecast service, while comparing neural models against a simple persistence baseline.

## Project Status

| Component                                     | Status                                                        |
| --------------------------------------------- | ------------------------------------------------------------- |
| FRED data preparation and feature engineering | Implemented                                                   |
| Chronological splits and train-only scaling   | Implemented                                                   |
| LSTM and Transformer training                 | Implemented                                                   |
| Validation-based model selection              | Implemented                                                   |
| Overall and per-horizon baseline evaluation   | Implemented                                                   |
| MLflow experiment tracking                    | Implemented and verified                                      |
| FastAPI inference service                     | Implemented and verified                                      |
| Automated tests                               | 29 passed                                                     |
| Docker serving                                | Implemented; healthy container and HTTP 200 forecast verified |
| GitHub Actions tests                          | Implemented and verified                                      |
| Docker checks in CI                           | Prepared; extended run not yet verified                       |

## Forecasting Task

Each model receives **60 historical observations with 19 engineered features** and predicts **five future simple returns**. The returns are compounded from the last observed S&P 500 value to produce index forecasts.

The Naive baseline repeats the last observed index value at all five horizons. Horizons correspond to retained trading observations, not calendar days.

The bundled data ends on **December 29, 2023**. Example forecasts describe that historical snapshot; starting the API or Docker container does not update the data.

## Data

The corrected dataset contains **1,509 observations**, covering **January 2, 2018 through December 29, 2023**, with six FRED series:

| Series     | Description            |
| ---------- | ---------------------- |
| SP500      | S&P 500 index          |
| NASDAQCOM  | NASDAQ Composite index |
| DGS10      | 10-year Treasury yield |
| UNRATE     | Unemployment rate      |
| CPIAUCSL   | Consumer Price Index   |
| DCOILWTICO | WTI crude oil price    |

The downloader sorts dates, forward-fills indicator values, retains dates with an observed S&P 500 value, and removes remaining incomplete rows. It does not backfill from future observations.

Features include percentage changes, moving-average ratios, rolling volatility, momentum, and changes in Treasury yields and oil prices. Rows without sufficient feature history are removed.

Chronological partitions use approximately 70% training, 15% validation, and 15% testing. Target windows remain entirely within their assigned partition; input windows may use earlier historical context. Both scalers are fitted on training data only.

| Partition  | Input shape       |
| ---------- | ----------------- |
| Training   | `(950, 60, 19)` |
| Validation | `(213, 60, 19)` |
| Testing    | `(214, 60, 19)` |

## Models

| Setting                 | LSTM             | Transformer        |
| ----------------------- | ---------------- | ------------------ |
| Layers                  | 2                | 2 encoder layers   |
| Hidden dimension        | 64               | 64                 |
| Attention heads         | —               | 4                  |
| Sequence representation | Last LSTM output | Last encoder token |
| Position encoding       | —               | Learned embeddings |
| Dropout                 | 0.2              | 0.2                |
| Output                  | Five returns     | Five returns       |

Training uses AdamW, learning rate `0.001`, weight decay `0.0001`, batch size `32`, MSE on scaled returns, and gradient clipping at `1.0`. Defaults allow up to 50 epochs with early stopping patience 7 and seed 42. Best validation weights are restored, and validation loss selects the default model.

## Latest Results

Results from the corrected-data run in `models/returns_mlflow_v1`, using default training settings. These match the earlier corrected-data run and are logged in MLflow. Metrics are calculated on reconstructed index values and pooled across all five horizons. MAE and RMSE are in index points; MAPE is a percentage.

| Model          |               MAE |              RMSE |         MAPE (%) |              R² |
| -------------- | ----------------: | ----------------: | ---------------: | ---------------: |
| Naive          |           46.6754 |           60.7450 |           1.0861 |           0.9192 |
| **LSTM** | **45.9707** | **60.1573** | **1.0711** | **0.9208** |
| Transformer    |           46.0685 |           60.3751 |           1.0726 |           0.9202 |

### RMSE by Horizon

| Horizon |   Naive |              LSTM |       Transformer |
| ------- | ------: | ----------------: | ----------------: |
| 1       | 33.9100 | **33.7987** |           34.0281 |
| 2       | 49.4342 | **49.1652** |           49.5732 |
| 3       | 60.8408 | **60.3145** |           61.0316 |
| 4       | 70.2853 |           69.5690 | **69.4541** |
| 5       | 78.8323 | **77.8282** |           77.8566 |

LSTM was selected by validation loss and reduced overall test RMSE by approximately **0.97%** relative to Naive. This is a modest improvement from one run, not evidence of a robust forecasting advantage. Both models reached their best validation loss in the first epoch and stopped after epoch 8.

Earlier notebook metrics came from a different experiment and a flawed data snapshot. They are superseded by these corrected-data results. Monte Carlo Dropout estimates from the earlier version are not part of the current modular pipeline.

## Local Setup

Run commands from the repository root. The verified Windows development environment uses Python 3.13.

```bash
git clone https://github.com/AinurAliWl/sp500_prediction_DL.git
cd sp500_prediction_DL
python -m venv .venv
```

Activate in Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

For PowerShell, use `.\.venv\Scripts\Activate.ps1`; for Linux or macOS, use `source .venv/bin/activate`.

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` contains the full development dependencies and currently has no version pins. `requirements-api.txt` pins the serving dependencies used by Docker. The Dockerfile installs CPU-only PyTorch separately. A full development dependency lock is future work.

## Training, Evaluation and Prediction

The corrected CSV is included in Git. To reproduce the documented snapshot, use that file. Downloading again may produce different values because source data can be revised.

Optional data download:

```bash
python download_data.py
```

This saves the processed CSV, an unfilled source CSV, and a backup of an existing processed CSV if no backup exists.

Train both models and save their artifacts:

```bash
python -m src.train --output-dir models/returns_mlflow_v1
```

Evaluate the saved test arrays and compare both models with Naive:

```bash
python -m src.evaluate --artifacts models/returns_mlflow_v1
```

Generate the latest forecast:

```bash
python -m src.predict --artifacts models/returns_mlflow_v1
```

Select Transformer explicitly:

```bash
python -m src.predict --artifacts models/returns_mlflow_v1 --model transformer
```

Always provide these artifact paths explicitly: current CLI defaults and the modular notebook still use `models/returns_v1`, while the API defaults to `models/returns_mlflow_v1`. Training into an existing directory overwrites its artifacts. Use a new output directory when preserving an experiment.

## MLflow Tracking

Training creates separate LSTM and Transformer runs in the `sp500-returns` experiment. It records model parameters, per-epoch losses, best validation loss and epoch, completed epochs, and a SHA-256 hash of the input CSV. Artifacts include weights, scalers, configuration, history, and source code.

Evaluation adds overall and per-horizon test metrics, baseline metrics, and result tables to the corresponding training runs. Runs created before MLflow integration have no stored run IDs and are evaluated locally without tracking.

Open the optional tracking UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --workers 1
```

Visit [http://127.0.0.1:5000](http://127.0.0.1:5000).

Useful metrics include `best_val_loss`, `test_rmse`, `test_mae`, `test_mape`, `test_r2`, `test_day_1_rmse` through `test_day_5_rmse`, and `test_rmse_improvement_pct`.

The local database is `mlflow.db`, and artifacts are in `mlartifacts/`. Both are ignored by Git. Artifact locations use absolute paths from the training machine; moving the project does not automatically update those paths.

Each MLflow run stores only its own model weights. The shared configuration may select LSTM even in a Transformer bundle. Explicitly select Transformer when using that downloaded bundle. Full evaluation requires both weights and the saved test arrays from the local training directory.

## FastAPI Service

The API loads the selected model and scalers once at startup. Requests do not train models or download data.

Model artifacts are ignored by Git. After cloning, train first or provide a compatible bundle before starting the service.

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

| Endpoint      | Method | Purpose                                            |
| ------------- | ------ | -------------------------------------------------- |
| `/health`   | GET    | Service status and model loading                   |
| `/metadata` | GET    | Input dimensions, feature order, model and horizon |
| `/predict`  | POST   | Five returns and reconstructed index values        |
| `/docs`     | GET    | Interactive documentation                          |

Generate an example request locally:

```bash
python -m api.create_example
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Expand **POST /predict**, click **Try it out**, paste the contents of `examples/predict_request.json` into **Request body**, and click **Execute**.

The request supplies an observation date, a positive last price, the exact saved feature order, and 60 rows of 19 **unscaled engineered features**. The API applies the saved scaler. The example generator computes these inputs from the local CSV using the project's feature logic.

Invalid dimensions, feature order, numeric values, or unexpected fields are rejected with HTTP 422. The caller is responsible for matching the supplied date and last price to the input history; the API echoes the date and does not validate historical chronology.

The verified LSTM request returned HTTP 200 and these index values:

```json
{
  "as_of": "2023-12-29",
  "last_price": 4769.83,
  "predicted_index": [
    4772.857269234657,
    4775.273284820318,
    4776.839799985885,
    4777.979859117269,
    4780.517556824684
  ]
}
```

This is an excerpt; the full response also contains the model name, horizon, horizon unit, and predicted returns. Tiny floating-point differences may occur across environments.

Set `SP500_ARTIFACT_DIR` to use another bundle and `SP500_MODEL` to select `lstm` or `transformer`. Relative artifact paths are resolved from the project root.

## Docker

**Docker serving is implemented and verified.** The local container reached `healthy`, and POST `/predict` returned HTTP 200 with the expected forecast.

Start Docker Desktop with Linux containers on Windows. Before building, ensure these files exist locally:

```text
models/returns_mlflow_v1/
├── lstm.pt
├── transformer.pt
├── feature_scaler.joblib
├── target_scaler.joblib
└── config.json
```

Build from the repository root:

```bash
docker build -t sp500-api:local .
```

The image uses Python 3.13, CPU-only PyTorch, pinned serving dependencies, a non-root user, and an HTTP health check. The `.dockerignore` includes only serving code, dependencies, and the required inference bundle.

Stop any local API already using port 8000, then run:

```bash
docker run --rm --name sp500-api -p 127.0.0.1:8000:8000 sp500-api:local
```

Open [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) or [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Use the same example request described above.

In a second terminal, check status, inspect logs, or stop the container:

```bash
docker ps
docker logs sp500-api
docker stop sp500-api
```

The verified status is `Up ... (healthy)`, with `127.0.0.1:8000->8000/tcp`. With `--rm`, stopping removes the container; the image remains available.

Docker does not require the local MLflow UI or local Uvicorn server. Weights and scalers are copied into the image at build time, so rebuild after changing the bundle. A fresh clone has no model artifacts and requires training or an existing bundle before this Docker build can succeed.

## Tests

```bash
python -m pytest -q
```

**29 tests passed** in the verified runs. Dependency deprecation and Transformer optimization warnings were emitted without failures.

Tests cover both model architectures, artifact serialization and loading, scaling, prediction consistency, return compounding, repeatable inference, invalid API requests, chronological target partitions, and train-only scaler fitting.

They use temporary untrained models and deterministic synthetic data. They do not require downloads, trained production artifacts, or running API and MLflow servers. They verify software behavior; forecasting quality is assessed separately against the baseline.

## Continuous Integration

The workflow is defined in `.github/workflows/ci.yml` and runs on pushes and pull requests to `main`. It can also be started manually from the GitHub Actions page.

The original automated test workflow completed successfully. The extended workflow adds Docker checks; its first successful GitHub run has not yet been confirmed.

### What the Extended Workflow Checks

1. Install Python 3.13, CPU PyTorch, serving dependencies, and test dependencies.
2. Check dependency compatibility and run the pytest suite.
3. Create deterministic, untrained model weights and scalers in the temporary CI environment.
4. Build the Docker image using that temporary inference bundle.
5. Start the container with LSTM, then Transformer.
6. Check `/health`, HTTP 200 from `/predict`, five finite forecasts, return compounding, and Docker health status.
7. Remove the container and show its logs if a check fails.

CI uses `scripts/create_ci_bundle.py` and `scripts/check_ci_api.py`. Temporary models check container behavior, not forecasting accuracy. The workflow does not train on FRED, use the local MLflow database, publish the image, or deploy the API.

**Do not run the CI bundle generator in your local trained-artifact directory.** It refuses to overwrite a nonempty directory. Your existing trained models remain the bundle used for local Docker serving.

After pushing the workflow and scripts, open [GitHub Actions](https://github.com/AinurAliWl/sp500_prediction_DL/actions), select the latest **Python tests** run, and inspect **Build Docker image** and **Check both models in Docker**. Mark Docker CI as verified only after those steps complete successfully.

## Repository Structure

```text
sp500_prediction_DL/
├── .github/workflows/ci.yml    # Automated tests and Docker checks
├── scripts/
│   ├── create_ci_bundle.py    # Untrained models for CI only
│   └── check_ci_api.py        # HTTP checks for container inference
├── api/                       # FastAPI service and example generator
├── src/                       # Data, features, models, training and inference
├── tests/                     # API and preprocessing tests
├── examples/predict_request.json
├── data/fred_economic_data.csv
├── prediction/                # Exploratory and modular notebooks
├── models/                    # Local training artifacts; ignored by Git
├── download_data.py
├── requirements.txt           # Development dependencies
├── requirements-api.txt       # Docker serving dependencies
├── Dockerfile
├── .dockerignore
├── .gitignore
├── pytest.ini
├── README.md
└── LICENSE
```

A complete local training directory additionally contains `history.json`, `test_data.npz`, `metrics_overall.csv`, and `metrics_by_horizon.csv`. Weights, scalers, feature order, and configuration must come from the same training run.

## Next Steps and Limitations

Automated tests have passed in GitHub Actions. Docker build and container checks are included in the prepared extended workflow; verification of its GitHub run is pending. Automated deployment is not configured.

The next engineering step is to confirm the extended workflow passes and keep its result visible in GitHub Actions. Model research and deployment can then be pursued separately.

Further model evaluation should include multiple seeds and walk-forward validation. The current improvement over persistence is small, overlapping forecast errors are not independent, and high R² on index levels does not establish strong predictive performance.

Forward filling avoids future-value backfilling, but revised macroeconomic values and observation dates are not a publication-aware historical dataset. Strict historical evaluation requires vintage data and release timing. Trading profitability and transaction costs are not evaluated.

## License

MIT License. See [LICENSE](LICENSE).
