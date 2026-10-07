import numpy as np

from src.data_preparation import prepare_data
from src.predict import returns_to_prices


def test_targets_stay_inside_chronological_partitions(synthetic_csv):
    data = prepare_data(synthetic_csv)
    horizon = data["horizon"]
    train = data["train"]["target_indices"]
    val = data["val"]["target_indices"]
    test = data["test"]["target_indices"]
    assert np.all(train + horizon <= data["train_end"])
    assert np.all(val >= data["train_end"])
    assert np.all(val + horizon <= data["val_end"])
    assert np.all(test >= data["val_end"])
    for split in ("train", "val", "test"):
        part = data[split]
        assert part["X"].shape[1:] == (60, 19)
        assert np.isfinite(part["X"]).all()
        assert np.isfinite(part["y_scaled"]).all()
        for index, actual in zip(part["target_indices"], part["target_prices"]):
            np.testing.assert_allclose(actual, data["df"]["SP500"].iloc[index:index+horizon])


def test_scalers_are_fitted_on_training_data_only(synthetic_csv):
    data = prepare_data(synthetic_csv)
    frame = data["df"][data["feature_columns"]]
    expected_mean = frame.iloc[:data["train_end"]].mean().to_numpy()
    np.testing.assert_allclose(data["feature_scaler"].mean_, expected_mean)
    assert not np.allclose(expected_mean, frame.mean().to_numpy())
    y = data["train"]["y"].reshape(-1)
    np.testing.assert_allclose(data["target_scaler"].mean_, [y.mean()])
    for split in ("train", "val", "test"):
        part = data[split]
        restored = data["target_scaler"].inverse_transform(part["y_scaled"].reshape(-1, 1))
        np.testing.assert_allclose(restored.reshape(part["y"].shape), part["y"])


def test_return_compounding_uses_each_previous_forecast():
    result = returns_to_prices(np.array([100.0, 200.0]),
                              np.array([[0.1, -0.1, 0.0], [0.0, 0.05, -0.2]]))
    np.testing.assert_allclose(result, [[110, 99, 99], [200, 210, 168]])
