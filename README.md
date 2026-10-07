
# S&P 500 Forecasting — LSTM, Transformer & MLOps

A modular PyTorch project that forecasts the S&P 500 over five future trading observations using market and macroeconomic data from FRED.

The project compares LSTM and Transformer models against a persistence baseline. Experiment tracking with MLflow, model serving with FastAPI, and automated checks with pytest are implemented. The next goal is to add containerization and continuous integration.

## Current Status

Implemented:

- FRED data download and chronological preprocessing.
- Feature engineering and train-only fitting of scalers.
- Chronological train, validation, and test partitions.
- LSTM and Transformer training with early stopping and best-weight restoration.
- Validation-based model selection.
- Overall and per-horizon evaluation against a Naive baseline.
- Saving and loading model weights, scalers, and configuration.
- Command-line inference from saved artifacts.
- MLflow tracking of training parameters, per-epoch losses, and best validation results.
- Logging model weights, scalers, configuration, and training history as MLflow artifacts.
- Logging overall and per-horizon test metrics, Naive baseline metrics, and evaluation tables to the corresponding training runs.
- FastAPI endpoints for health checks, input metadata, and five-horizon forecasts.
- Input validation and loading the selected model once at startup.
- Automated tests for both architectures, API inference, invalid inputs, chronological partitions, and scalers.

Docker and GitHub Actions are planned and are not implemented yet.

## How It Works

```text
FRED data → engineered features → chronological split → scaling
          → LSTM / Transformer → five returns → index forecasts
```

Each model receives 60 historical observations with 19 engineered features and predicts five future daily returns in one forward pass. The returns are compounded from the last observed index value to reconstruct the forecast.

The Naive baseline predicts the last observed index value at every horizon, equivalent to predicting zero returns. Horizons refer to retained trading observations, rather than calendar days.

## Data and Preprocessing

The current corrected snapshot contains **1,509 observations**, from **January 2, 2018 to December 29, 2023**.

| Indicator      | Description            |
| -------------- | ---------------------- |
| `SP500`      | S&P 500 index          |
| `NASDAQCOM`  | NASDAQ Composite index |
| `DGS10`      | 10-year Treasury yield |
| `UNRATE`     | Unemployment rate      |
| `CPIAUCSL`   | Consumer Price Index   |
| `DCOILWTICO` | WTI crude oil price    |

Dates are sorted before forward-filling missing indicator values. Only dates with an observed S&P 500 value are retained, and remaining incomplete rows are removed. The downloader does not backfill from future observations.

The 19 engineered features include indicator percentage changes, S&P 500 moving-average ratios, rolling return volatility, S&P 500 and NASDAQ momentum, and five-observation changes in Treasury yields and oil prices. Initial rows without sufficient rolling history are removed.

The feature table is partitioned chronologically at approximately 70% / 15% / 15%. Each target window stays within its partition; input windows may use earlier historical context. Feature and target scalers are fitted on training data only.

| Partition  | Input shape       |
| ---------- | ----------------- |
| Train      | `(950, 60, 19)` |
| Validation | `(213, 60, 19)` |
| Test       | `(214, 60, 19)` |

These sizes describe the current snapshot and default settings.

## Models and Training

| Setting                   | LSTM                        | Transformer                 |
| ------------------------- | --------------------------- | --------------------------- |
| Architecture              | 2-layer unidirectional LSTM | 2-layer Transformer encoder |
| Hidden dimension          | 64                          | 64                          |
| Attention heads           | —                          | 4                           |
| Sequence representation   | Last LSTM output            | Last encoder token          |
| Positional representation | —                          | Learned embeddings          |
| Output                    | 5 returns                   | 5 returns                   |
| Dropout                   | 0.2                         | 0.2                         |

Both models use LayerNorm and a feed-forward output head. Default training settings:

- AdamW with learning rate `0.001` and weight decay `0.0001`.
- Batch size `32` and MSE loss on scaled returns.
- Gradient clipping at `1.0`.
- Up to `50` epochs, with early stopping patience `7`.
- Restoration of the weights with the lowest validation loss.
- Random seed `42`.

The model with the lowest validation loss is selected for default inference. Test metrics do not determine selection.

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

## Installation

```bash
git clone https://github.com/AinurAliWl/sp500_prediction_DL.git
cd sp500_prediction_DL
python -m venv .venv
```

Activate the environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or in Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Or on Linux / macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

Run commands from the repository root.

### 1. Download Data

```bash
python download_data.py
```

The downloader writes `data/fred_economic_data.csv`, saves the unfilled source table as `data/fred_economic_data_raw.csv`, and preserves an existing processed CSV as a backup if no backup exists yet. Downloading requires internet access. Source revisions can change a newly downloaded snapshot.

### 2. Train

```bash
python -m src.train --output-dir models/returns_mlflow_v1
```

For a one-epoch smoke check, use a separate directory:

```bash
python -m src.train --epochs 1 --output-dir models/mlflow_smoke_test
```

### 3. Evaluate

```bash
python -m src.evaluate --artifacts models/returns_mlflow_v1
```

Evaluation logs metrics to the corresponding MLflow runs, prints overall and per-horizon metrics, and saves `metrics_overall.csv` and `metrics_by_horizon.csv` in the artifact directory.

### 4. Predict

```bash
python -m src.predict --artifacts models/returns_mlflow_v1
```

To select a model explicitly:

```bash
python -m src.predict --artifacts models/returns_mlflow_v1 --model transformer
```

The JSON output contains the model name, last observation date, last index value, five predicted returns, and five reconstructed index values. With this snapshot, forecasts start from **December 29, 2023**, not the current date.

Pass the corrected run's artifact directory explicitly: the current CLI defaults still point to `models/returns_v1`.

## Experiment Tracking with MLflow

Each training invocation creates separate LSTM and Transformer runs in the `sp500-returns` experiment. Runs record parameters, per-epoch training and validation losses, best validation loss, best epoch, and the number of completed epochs.

Saved artifacts include model weights, both scalers, configuration, training history, and the training script. The configuration stores the MLflow run IDs and a SHA-256 hash of the input CSV.

Evaluation adds test metrics and CSV tables to the original training runs without retraining or creating new runs.

### Open the Tracking UI

Run from the repository root:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --workers 1
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) and select `sp500-returns`. Keep the UI terminal running and use a second terminal for training and evaluation. The single-worker setting is used for compatibility with the current Windows environment.

### Compare Runs

| Metric                                          | Meaning                                                                |
| ----------------------------------------------- | ---------------------------------------------------------------------- |
| `best_val_loss`                               | Validation loss used for model selection                               |
| `test_mae`, `test_rmse`                     | Overall errors in index points                                         |
| `test_mape`                                   | Overall percentage error                                               |
| `test_r2`                                     | Overall R²                                                            |
| `test_day_1_rmse` through `test_day_5_rmse` | RMSE by forecast horizon                                               |
| `naive_test_rmse`                             | Persistence baseline RMSE                                              |
| `test_rmse_improvement_pct`                   | RMSE reduction relative to Naive; positive values indicate improvement |

Both model and baseline metrics are also recorded by horizon. Result tables are available under **Artifacts → evaluation**, and the model bundle is under **Artifacts → inference**.

Repeated training creates new records even when run names repeat. Metadata is stored in `mlflow.db`, and MLflow artifacts are stored in `mlartifacts/`; both are excluded from Git. Keep them locally to retain experiment history. Artifact directories created before MLflow integration have no run IDs, so their evaluation saves local tables without logging to MLflow.

## Model Serving with FastAPI

The API loads the validation-selected model and its scalers from `models/returns_mlflow_v1` once at startup. Requests do not retrain the model or download data.

Model artifacts are excluded from Git. After cloning, run training first or supply an existing compatible artifact bundle before starting the API.

### Start the API

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

| Endpoint      | Method | Purpose                                                      |
| ------------- | ------ | ------------------------------------------------------------ |
| `/health`   | GET    | Check service status and model loading                       |
| `/metadata` | GET    | Get model name, input dimensions, feature order, and horizon |
| `/predict`  | POST   | Generate five predicted returns and index values             |
| `/docs`     | GET    | Open interactive API documentation                           |

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Create and Send an Example Request

```bash
python -m api.create_example
```

This writes `examples/predict_request.json` from the latest valid observations in the local CSV. In `/docs`, expand **POST /predict**, select **Try it out**, paste the JSON file contents into **Request body**, and select **Execute**.

The request contains:

- `as_of`: date of the last input observation.
- `last_price`: positive S&P 500 value at that observation.
- `feature_columns`: feature names in exactly the saved order.
- `features`: 60 chronological rows with 19 unscaled engineered features per row.

The API applies the saved feature scaler itself. Do not send already scaled features or the six raw indicators. The caller is responsible for computing features with the project's feature logic and supplying a matching last price and observation date. The example generator does this using `src/features.py`.

Requests with invalid dimensions, feature order, numeric values, or fields are rejected with HTTP `422`.

### Verified Response

The local API returned HTTP `200` for the generated request, with predictions matching command-line inference:

```json
{
  "model": "lstm",
  "as_of": "2023-12-29",
  "last_price": 4769.83,
  "horizon": 5,
  "horizon_unit": "retained_trading_observations",
  "predicted_returns": [
    0.0006346516311168671,
    0.0005061982083134353,
    0.0003281015087850392,
    0.00023867588606663048,
    0.0005311177228577435
  ],
  "predicted_index": [
    4772.857269234657,
    4775.273284820318,
    4776.839799985885,
    4777.979859117269,
    4780.517556824684
  ]
}
```

The response echoes the supplied observation date. It does not assign calendar dates to future horizons.

To use another artifact directory or model, set `SP500_ARTIFACT_DIR` or `SP500_MODEL` before starting the server. Relative artifact paths are resolved from the project root. `SP500_MODEL` supports `lstm` and `transformer`.

## Automated Tests

Run from the repository root:

```bash
python -m pytest -q
```

**Latest local result: 29 tests passed.** The successful run emitted dependency and Transformer optimization warnings, but no test failures.

The suite checks:

- Health and metadata endpoints for both architectures.
- Serialization, loading, scaling, and API predictions against the original in-memory model.
- Index reconstruction by compounding predicted returns.
- Repeatable inference with dropout disabled.
- Rejection of malformed requests with HTTP `422`.
- Target windows staying within chronological partitions.
- Feature and target scalers fitted on training data only.
- Target scaling round trips.

Tests create temporary untrained models and deterministic synthetic data. They do not modify trained artifacts, require downloads, or need a running Uvicorn or MLflow server. These checks validate pipeline behavior, not forecasting accuracy.

## Saved Artifacts

```text
models/returns_mlflow_v1/
├── lstm.pt
├── transformer.pt
├── feature_scaler.joblib
├── target_scaler.joblib
├── config.json                  # Includes MLflow run IDs
├── history.json
├── test_data.npz
├── metrics_overall.csv
└── metrics_by_horizon.csv
```

Weights, scalers, feature order, and configuration must come from the same training run. Evaluation uses the test arrays saved during training. Training into an existing directory overwrites its artifacts; use a new directory to preserve another experiment.

## Project Structure

```text
sp500_prediction_DL/
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── create_example.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   └── test_data.py
├── examples/
│   └── predict_request.json
├── data/
│   └── fred_economic_data.csv
├── models/
├── prediction/
│   ├── model.ipynb
│   └── modular_experiment.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── features.py
│   ├── data_preparation.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── pytest.ini
├── download_data.py
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

Python modules implement the working pipeline. Notebooks remain available for exploration and analysis; the original notebook also contains historical experiments.

## Roadmap

Completed stages and the next planned steps:

1. **Completed — MLflow:** track training parameters, losses, evaluation metrics, and inference artifacts.
2. **Completed — FastAPI:** serve the saved model through `/health`, `/metadata`, and `/predict`, with input validation and interactive documentation.
3. **Completed — pytest:** verify preprocessing, artifact loading, inference, and API behavior; 29 tests pass locally.
4. **Docker:** package the API and its inference artifacts in a runnable container.
5. **GitHub Actions:** automatically run tests and build the Docker image on pushes and pull requests.
6. **Documentation:** add verified API examples and setup instructions as each stage is implemented.

Automated deployment is a possible later stage. Tests and Docker builds constitute CI, not CD.

## Limitations and Further Evaluation

- Results describe one seed and one chronological test period. Multi-seed experiments and walk-forward validation remain future work.
- The small improvement over persistence requires further evaluation. High R² on index levels should be interpreted alongside the baseline.
- Forecast windows overlap, so pooled errors are not independent observations.
- Forward filling prevents future-value backfilling, but macroeconomic observation dates and revised values do not represent information available at the time. Strict historical evaluation requires publication-aware, vintage data.
- The pipeline does not evaluate trading profitability or transaction costs.
- MLflow, FastAPI, and Uvicorn are pinned to `3.17.0`, `0.142.2`, and `0.54.0`. A complete dependency lock and cross-platform validation remain future work.

## License

Distributed under the MIT License. See [LICENSE](LICENSE).
