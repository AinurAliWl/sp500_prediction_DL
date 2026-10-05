import pandas as pd


def create_features(data):
    df = data.copy()

    for col in df.columns:
        df[f"{col}_return"] = df[col].pct_change()

    if "SP500" in df.columns:
        for window in [5, 20, 50]:
            df[f"SP500_MA{window}"] = (
                df["SP500"].rolling(window).mean()
            )

        df["SP500_Vol_20"] = (
            df["SP500_return"].rolling(20).std()
        )

    if "SP500" in df.columns and "DCOILWTICO" in df.columns:
        df["SP500_Oil_Ratio"] = (
            df["SP500"] / df["DCOILWTICO"]
        )

    if "DGS10" in df.columns and "UNRATE" in df.columns:
        df["Yield_Unemployment"] = (
            df["DGS10"] / (df["UNRATE"] + 1)
        )

    for col in ["SP500", "NASDAQCOM", "DCOILWTICO"]:
        if col in df.columns:
            df[f"{col}_Momentum_5"] = (
                df[col] / df[col].shift(5) - 1
            )

            df[f"{col}_Momentum_20"] = (
                df[col] / df[col].shift(20) - 1
            )

    return df