from pathlib import Path
import pandas as pd
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "fred_economic_data.csv"
INDICATORS = ["SP500", "DGS10", "UNRATE", "CPIAUCSL", "DCOILWTICO", "NASDAQCOM"]


def read_data(path=DEFAULT_DATA_PATH):
    '''Read the existing snapshot. Never download or fill it implicitly.'''
    data = pd.read_csv(path, index_col=0, parse_dates=True)
    if not data.index.is_monotonic_increasing:
        warnings.warn("CSV dates are unsorted. Inspect whether values were filled before sorting.")
    data = data.sort_index()
    missing = set(INDICATORS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if data.index.has_duplicates or data.index.hasnans:
        raise ValueError("Dates must be unique and valid.")
    data = data[INDICATORS].apply(pd.to_numeric, errors="raise")
    if (data["SP500"].dropna() <= 0).any():
        raise ValueError("SP500 must be positive.")
    if (data["SP500"].pct_change(fill_method=None).abs() > 0.30).any():
        warnings.warn("SP500 changes exceed 30%. Inspect source data before interpreting metrics.")
    return data
