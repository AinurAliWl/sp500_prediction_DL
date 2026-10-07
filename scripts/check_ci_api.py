"""Check both container architectures through the published HTTP API."""
import argparse
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "transformer"], required=True)
    args = parser.parse_args()
    url = "http://127.0.0.1:8000"
    deadline = time.monotonic() + 90
    while True:
        try:
            with urllib.request.urlopen(url + "/health", timeout=3) as response:
                health = json.load(response)
            assert health == {"status": "ok", "model_loaded": True, "model": args.model}, health
            break
        except (urllib.error.URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise RuntimeError("Container API did not become ready within 90 seconds.")
            time.sleep(2)
    payload = Path("ci_predict_request.json").read_bytes()
    request = urllib.request.Request(url + "/predict", data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=15) as response:
        assert response.status == 200
        result = json.load(response)
    assert result["model"] == args.model
    assert result["horizon"] == 5
    assert result["as_of"] == "2023-12-29"
    for key in ("predicted_returns", "predicted_index"):
        assert len(result[key]) == 5
        assert all(math.isfinite(x) for x in result[key])
    price = 100.0
    for daily_return, forecast in zip(result["predicted_returns"], result["predicted_index"]):
        price *= 1 + daily_return
        assert math.isclose(price, forecast, rel_tol=1e-5, abs_tol=1e-4)
    print(f"{args.model}: /health and /predict passed; five finite forecasts.")


if __name__ == "__main__":
    main()
