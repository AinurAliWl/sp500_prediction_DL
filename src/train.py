import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

from .data_loader import DEFAULT_DATA_PATH, PROJECT_ROOT
from .data_preparation import prepare_data, make_loader
from .models import LSTMModel, TransformerModel

def train_model(
    model,
    train_loader,
    val_loader,
    name,
    device="cpu",
    epochs=50,
    patience=7,
    learning_rate=1e-3,
):
    model = model.to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    best_loss = float("inf")
    best_state = None

    patience_counter = 0

    history = {
        "train": [],
        "val": [],
    }

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        train_loss = 0.0
        train_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            predictions = model(
                X_batch
            )

            loss = criterion(
                predictions,
                y_batch,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            batch_size = X_batch.size(0)

            train_loss += (
                loss.item()
                * batch_size
            )

            train_samples += batch_size

        train_loss /= train_samples

        model.eval()

        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                predictions = model(
                    X_batch
                )

                loss = criterion(
                    predictions,
                    y_batch,
                )

                batch_size = X_batch.size(0)

                val_loss += (
                    loss.item()
                    * batch_size
                )

                val_samples += batch_size

        val_loss /= val_samples

        history["train"].append(
            train_loss
        )

        history["val"].append(
            val_loss
        )

        print(
            f"{name} | "
            f"Epoch {epoch:02d} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss

            best_state = {
                key: value.cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch}"
            )
            break

    model.load_state_dict(
        best_state
    )

    model = model.to(device)

    return model, history, best_loss

def run_training(data_path=DEFAULT_DATA_PATH, output_dir=None, epochs=50,
                 patience=7, batch_size=32, learning_rate=1e-3,
                 lookback=60, horizon=5, seed=42, device=None):
    if epochs < 1 or patience < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Invalid training parameters.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir or PROJECT_ROOT / "models" / "returns_v1")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_data(data_path, lookback, horizon)
    train_loader = make_loader(data["train"]["X"], data["train"]["y_scaled"],
                               True, batch_size)
    val_loader = make_loader(data["val"]["X"], data["val"]["y_scaled"],
                             False, batch_size)
    for split in ("train", "val", "test"):
        print(f'{split}: {data[split]["X"].shape}')
    input_size = len(data["feature_columns"])
    config = {
        "seed": seed, "lookback": lookback, "horizon": horizon,
        "train_ratio": 0.70, "val_ratio": 0.15,
        "input_size": input_size, "feature_columns": data["feature_columns"],
        "epochs": epochs, "patience": patience, "batch_size": batch_size,
        "learning_rate": learning_rate, "weight_decay": 1e-4,
        "target": "daily_returns_on_filtered_rows",
        "train_end": data["train_end"], "val_end": data["val_end"],
        "data_start": str(data["df"].index[0]),
        "data_end": str(data["df"].index[-1]),
        "models": {
            "lstm": {"input_size": input_size, "hidden_size": 64,
                     "num_layers": 2, "dropout": 0.2, "horizon": horizon},
            "transformer": {"input_size": input_size, "d_model": 64,
                            "nhead": 4, "num_layers": 2, "dropout": 0.2,
                            "horizon": horizon, "lookback": lookback},
        },
    }
    histories, best_losses = {}, {}
    # Preserve the notebook's RNG order: initialize/train LSTM first,
    # then initialize/train Transformer.
    for name, cls in (("lstm", LSTMModel), ("transformer", TransformerModel)):
        model = cls(**config["models"][name])
        model, history, best_loss = train_model(
            model, train_loader, val_loader, name, device,
            epochs, patience, learning_rate)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                   output_dir / f"{name}.pt")
        histories[name] = history
        best_losses[name] = best_loss
    config["best_val_losses"] = best_losses
    config["selected_model"] = min(best_losses, key=best_losses.get)
    joblib.dump(data["feature_scaler"], output_dir / "feature_scaler.joblib")
    joblib.dump(data["target_scaler"], output_dir / "target_scaler.joblib")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    (output_dir / "history.json").write_text(json.dumps(histories, indent=2))
    # Freeze the exact test arrays so evaluation does not fit new scalers.
    test = data["test"]
    np.savez_compressed(output_dir / "test_data.npz", X=test["X"],
        last_prices=test["last_prices"], target_prices=test["target_prices"],
        target_dates=np.array([
            data["df"].index[i:i+horizon].astype(str).tolist()
            for i in test["target_indices"]]))
    print(f"Artifacts: {output_dir}")
    print(f'Validation-selected model: {config["selected_model"]}')
    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    args = parser.parse_args()
    run_training(data_path=args.data, output_dir=args.output_dir,
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        learning_rate=args.learning_rate, lookback=args.lookback,
        horizon=args.horizon, seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
