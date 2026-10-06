import numpy as np
import pandas as pd

def create_features(data):
    df = data.copy()

    # Returns
    for col in df.columns:
        df[f"{col}_return"] = df[col].pct_change(
            fill_method=None
        )

    # Moving averages as distance from current price
    for window in [5, 20, 50]:
        ma = df["SP500"].rolling(window).mean()

        df[f"SP500_MA{window}_ratio"] = (
            df["SP500"] / ma - 1
        )

    # Volatility
    df["SP500_Vol_5"] = (
        df["SP500_return"]
        .rolling(5)
        .std()
    )

    df["SP500_Vol_20"] = (
        df["SP500_return"]
        .rolling(20)
        .std()
    )

    # Momentum
    for window in [5, 20, 60]:
        df[f"SP500_Momentum_{window}"] = (
            df["SP500"]
            / df["SP500"].shift(window)
            - 1
        )

        df[f"NASDAQ_Momentum_{window}"] = (
            df["NASDAQCOM"]
            / df["NASDAQCOM"].shift(window)
            - 1
        )

    # Changes in macro variables
    df["Yield_Change_5"] = (
        df["DGS10"].pct_change(
            periods=5,
            fill_method=None,
        )
    )

    df["Oil_Change_5"] = (
        df["DCOILWTICO"].pct_change(
            periods=5,
            fill_method=None,
        )
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna()

    return df
