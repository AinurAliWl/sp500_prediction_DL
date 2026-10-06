import numpy as np
from sklearn.preprocessing import StandardScaler

from .data_loader import read_data, DEFAULT_DATA_PATH, INDICATORS
from .features import create_features

def create_sequences(
    features,
    returns,
    prices,
    lookback,
    horizon,
):
    X = []
    y = []

    last_prices = []
    target_prices = []
    target_indices = []

    for i in range(
        lookback,
        len(features) - horizon + 1,
    ):
        X.append(
            features[i - lookback:i]
        )

        y.append(
            returns[i:i + horizon]
        )

        last_prices.append(
            prices[i - 1]
        )

        target_prices.append(
            prices[i:i + horizon]
        )

        target_indices.append(i)

    return (
        np.array(X),
        np.array(y),
        np.array(last_prices),
        np.array(target_prices),
        np.array(target_indices),
    )

def make_loader(X, y, shuffle=False, batch_size=32):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def prepare_data(path=DEFAULT_DATA_PATH, lookback=60, horizon=5,
                 train_ratio=0.70, val_ratio=0.15):
    '''Match the new notebook, including returns on the filtered row index.

    Horizons are future retained observations, not calendar days. A target
    window must be completely inside its assigned partition; past input
    context may come from an earlier partition.
    '''
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must be positive.")
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1
            and train_ratio + val_ratio < 1):
        raise ValueError("Invalid split ratios.")
    df = create_features(read_data(path))
    train_end = int(len(df) * train_ratio)
    val_end = int(len(df) * (train_ratio + val_ratio))
    feature_columns = [c for c in df.columns if c not in INDICATORS]
    feature_scaler = StandardScaler().fit(df.iloc[:train_end][feature_columns])
    features = feature_scaler.transform(df[feature_columns])
    prices = df["SP500"].to_numpy()
    returns = df["SP500"].pct_change(fill_method=None).to_numpy()
    if len(df) < lookback + horizon:
        raise ValueError("Not enough usable observations.")
    X, y, last_prices, target_prices, indices = create_sequences(
        features, returns, prices, lookback, horizon)
    masks = {
        "train": indices + horizon <= train_end,
        "val": (indices >= train_end) & (indices + horizon <= val_end),
        "test": indices >= val_end,
    }
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("A split has no complete sequences.")
    target_scaler = StandardScaler().fit(y[masks["train"]].reshape(-1, 1))
    result = {
        "df": df, "feature_columns": feature_columns,
        "feature_scaler": feature_scaler, "target_scaler": target_scaler,
        "train_end": train_end, "val_end": val_end,
        "lookback": lookback, "horizon": horizon,
    }
    for split, mask in masks.items():
        raw_y = y[mask]
        result[split] = {
            "X": X[mask],
            "y": raw_y,
            "y_scaled": target_scaler.transform(raw_y.reshape(-1, 1)).reshape(raw_y.shape),
            "last_prices": last_prices[mask],
            "target_prices": target_prices[mask],
            "target_indices": indices[mask],
        }
    return result
