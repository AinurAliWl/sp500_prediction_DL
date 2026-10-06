# S&P 500 Forecasting — LSTM, Transformer & MLOps

A modular PyTorch project that forecasts the S&P 500 over five future trading observations using market and macroeconomic data from FRED.

The project compares LSTM and Transformer models against a persistence baseline. The next goal is to extend the working training and inference pipeline with experiment tracking, model serving, containerization, and continuous integration.

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

MLflow, FastAPI, Docker, automated tests, and GitHub Actions are planned and are not implemented yet.

## How It Works

```text
FRED data → engineered features → chronological split → scaling
          → LSTM / Transformer → five returns → index forecasts
```

Each model receives 60 historical observations with 19 engineered features and predicts five future daily returns in one forward pass. The returns are compounded from the last observed index value to reconstruct the forecast.

The Naive baseline predicts the last observed index value at every horizon, equivalent to predicting zero returns. Horizons refer to retained trading observations, rather than calendar days.

## Data and Preprocessing

The current corrected snapshot contains **1,509 observations**, from **January 2, 2018 to December 29, 2023**.

| Indicator | Description |
|---|---|
| `SP500` | S&P 500 index |
| `NASDAQCOM` | NASDAQ Composite index |
| `DGS10` | 10-year Treasury yield |
| `UNRATE` | Unemployment rate |
| `CPIAUCSL` | Consumer Price Index |
| `DCOILWTICO` | WTI crude oil price |

Dates are sorted before forward-filling missing indicator values. Only dates with an observed S&P 500 value are retained, and remaining incomplete rows are removed. The downloader does not backfill from future observations.

The 19 engineered features include indicator percentage changes, S&P 500 moving-average ratios, rolling return volatility, S&P 500 and NASDAQ momentum, and five-observation changes in Treasury yields and oil prices. Initial rows without sufficient rolling history are removed.

The feature table is partitioned chronologically at approximately 70% / 15% / 15%. Each target window stays within its partition; input windows may use earlier historical context. Feature and target scalers are fitted on training data only.

| Partition | Input shape |
|---|---|
| Train | `(950, 60, 19)` |
| Validation | `(213, 60, 19)` |
| Test | `(214, 60, 19)` |

These sizes describe the current snapshot and default settings.

## Models and Training

| Setting | LSTM | Transformer |
|---|---|---|
| Architecture | 2-layer unidirectional LSTM | 2-layer Transformer encoder |
| Hidden dimension | 64 | 64 |
| Attention heads | — | 4 |
| Sequence representation | Last LSTM output | Last encoder token |
| Positional representation | — | Learned embeddings |
| Output | 5 returns | 5 returns |
| Dropout | 0.2 | 0.2 |

Both models use LayerNorm and a feed-forward output head. Default training settings:

- AdamW with learning rate `0.001` and weight decay `0.0001`.
- Batch size `32` and MSE loss on scaled returns.
- Gradient clipping at `1.0`.
- Up to `50` epochs, with early stopping patience `7`.
- Restoration of the weights with the lowest validation loss.
- Random seed `42`.

The model with the lowest validation loss is selected for default inference. Test metrics do not determine selection.

## Latest Results

Results from the corrected-data run in `models/returns_clean_v1`, using default training settings. Metrics are calculated on reconstructed index values and pooled across all five horizons. MAE and RMSE are in index points; MAPE is a percentage.

| Model | MAE | RMSE | MAPE (%) | R² |
|---|---:|---:|---:|---:|
| Naive | 46.6754 | 60.7450 | 1.0861 | 0.9192 |
| **LSTM** | **45.9707** | **60.1573** | **1.0711** | **0.9208** |
| Transformer | 46.0685 | 60.3751 | 1.0726 | 0.9202 |

### RMSE by Horizon

| Horizon | Naive | LSTM | Transformer |
|---|---:|---:|---:|
| 1 | 33.9100 | **33.7987** | 34.0281 |
| 2 | 49.4342 | **49.1652** | 49.5732 |
| 3 | 60.8408 | **60.3145** | 61.0316 |
| 4 | 70.2853 | 69.5690 | **69.4541** |
| 5 | 78.8323 | **77.8282** | 77.8566 |

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
python -m src.train --output-dir models/returns_clean_v1
```

For a one-epoch smoke check, use a separate directory:

```bash
python -m src.train --epochs 1 --output-dir models/smoke_test
```

### 3. Evaluate

```bash
python -m src.evaluate --artifacts models/returns_clean_v1
```

Evaluation prints overall and per-horizon metrics and saves `metrics_overall.csv` and `metrics_by_horizon.csv` in the artifact directory.

### 4. Predict

```bash
python -m src.predict --artifacts models/returns_clean_v1
```

To select a model explicitly:

```bash
python -m src.predict --artifacts models/returns_clean_v1 --model transformer
```

The JSON output contains the model name, last observation date, last index value, five predicted returns, and five reconstructed index values. With this snapshot, forecasts start from **December 29, 2023**, not the current date.

Pass the corrected run's artifact directory explicitly: the current CLI defaults still point to `models/returns_v1`.

## Saved Artifacts

```text
models/returns_clean_v1/
├── lstm.pt
├── transformer.pt
├── feature_scaler.joblib
├── target_scaler.joblib
├── config.json
├── history.json
├── test_data.npz
├── metrics_overall.csv
└── metrics_by_horizon.csv
```

Weights, scalers, feature order, and configuration must come from the same training run. Evaluation uses the test arrays saved during training. Training into an existing directory overwrites its artifacts; use a new directory to preserve another experiment.

## Project Structure

```text
sp500_prediction_DL/
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
├── download_data.py
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

Python modules implement the working pipeline. Notebooks remain available for exploration and analysis; the original notebook also contains historical experiments.

## Roadmap

The next stages extend the existing project in this order:

1. **MLflow:** log training parameters, losses, evaluation metrics, and inference artifacts.
2. **FastAPI:** serve the saved model through `/health` and `/predict`, with input validation.
3. **pytest:** test preprocessing, artifact loading, inference, and API behavior.
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
- Dependency versions are not yet pinned; environment reproducibility remains a task for the MLOps upgrade.

## License

Distributed under the MIT License. See [LICENSE](LICENSE).
