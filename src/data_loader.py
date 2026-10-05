from datetime import datetime

import pandas as pd
import pandas_datareader.data as web


START_DATE = datetime(2018, 1, 1)
END_DATE = datetime(2023, 12, 31)

INDICATORS = {
    "SP500": "S&P 500 Stock Index",
    "DGS10": "10-Year Treasury Yield",
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "Consumer Price Index",
    "DCOILWTICO": "Crude Oil Price",
    "NASDAQCOM": "NASDAQ Composite",
}


def load_fred_data():
    """Download economic data from FRED."""
    data_frames = []

    for code, description in INDICATORS.items():
        print(f"Loading {code}: {description}")

        data = web.DataReader(
            code,
            "fred",
            START_DATE,
            END_DATE,
        )

        data.columns = [code]
        data_frames.append(data)

    return pd.concat(data_frames, axis=1, sort=False)


def main():
    print("Downloading FRED data...")

    data = load_fred_data()

    print(f"\nShape: {data.shape}")
    print(
        f"Date range: "
        f"{data.index.min().date()} - "
        f"{data.index.max().date()}"
    )
    print(f"Columns: {list(data.columns)}")

    data.to_csv("data/fred_economic_data.csv")

    print("\nData saved to data/fred_economic_data.csv")


if __name__ == "__main__":
    main()