import torch
import torch.nn as nn

class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        horizon=5,
        lookback=60,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(
            hidden_size
        )

        self.head = nn.Sequential(
            nn.Linear(
                hidden_size,
                64,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                64,
                horizon,
            ),
        )

    def forward(self, x):
        output, _ = self.lstm(x)

        x = output[:, -1, :]
        x = self.norm(x)

        return self.head(x)

class TransformerModel(nn.Module):

    def __init__(
        self,
        input_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.2,
        horizon=5,
        lookback=60,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_size,
            d_model,
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                lookback,
                d_model,
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        self.head = nn.Sequential(
            nn.Linear(
                d_model,
                64,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                64,
                horizon,
            ),
        )

    def forward(self, x):
        x = self.input_projection(x)

        x = (
            x
            + self.position_embedding[
                :, :x.size(1)
            ]
        )

        x = self.encoder(x)

        x = x[:, -1, :]
        x = self.norm(x)

        return self.head(x)
