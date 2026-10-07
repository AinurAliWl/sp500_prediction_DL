from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch


def test_health_and_metadata(api_case):
    client, payload, _, _, _, name = api_case
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "model_loaded": True, "model": name}
    response = client.get("/metadata")
    assert response.status_code == 200
    metadata = response.json()
    assert metadata["feature_columns"] == payload["feature_columns"]
    assert (metadata["lookback"], metadata["input_size"], metadata["horizon"]) == (60, 19, 5)
    assert metadata["input_scale"] == "unscaled_engineered_features"


def test_api_matches_original_model_and_manual_reconstruction(api_case):
    client, payload, original, feature_scaler, target_scaler, name = api_case
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    # Independent reference: original in-memory weights, explicit scaling,
    # direct forward pass, inverse scaling, and manual compounding.
    frame = pd.DataFrame(payload["features"], columns=payload["feature_columns"])
    X = feature_scaler.transform(frame)[None]
    with torch.no_grad():
        scaled = original(torch.tensor(X, dtype=torch.float32)).numpy()
    expected_returns = target_scaler.inverse_transform(scaled.reshape(-1, 1)).ravel()
    price = payload["last_price"]
    expected_prices = []
    for value in expected_returns:
        price *= 1 + float(value)
        expected_prices.append(price)
    np.testing.assert_allclose(body["predicted_returns"], expected_returns, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(body["predicted_index"], expected_prices, rtol=1e-6, atol=1e-3)
    assert len(body["predicted_index"]) == 5
    assert body["model"] == name
    assert body["as_of"] == payload["as_of"]
    # Dropout must be disabled during inference.
    again = client.post("/predict", json=payload)
    assert again.status_code == 200
    np.testing.assert_allclose(again.json()["predicted_returns"], body["predicted_returns"], atol=1e-8)


@pytest.mark.parametrize("case", [
    "short_history", "long_history", "short_row", "column_order",
    "negative_price", "zero_price", "missing_field", "extra_field",
    "string_number", "boolean_number", "invalid_date",
])
def test_invalid_requests_are_rejected(api_case, case):
    client, original_payload, *_ = api_case
    payload = deepcopy(original_payload)
    if case == "short_history":
        payload["features"].pop()
    elif case == "long_history":
        payload["features"].append(payload["features"][-1])
    elif case == "short_row":
        payload["features"][0].pop()
    elif case == "column_order":
        payload["feature_columns"].reverse()
    elif case == "negative_price":
        payload["last_price"] = -1
    elif case == "zero_price":
        payload["last_price"] = 0
    elif case == "missing_field":
        del payload["last_price"]
    elif case == "extra_field":
        payload["unexpected"] = 1
    elif case == "string_number":
        payload["features"][0][0] = "1.0"
    elif case == "boolean_number":
        payload["features"][0][0] = True
    elif case == "invalid_date":
        payload["as_of"] = "not-a-date"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, response.text
