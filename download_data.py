from pathlib import Path
import shutil

import pandas as pd
from pandas_datareader import data as web


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "fred_economic_data.csv"
BACKUP_PATH = ROOT / "data" / "fred_economic_data_backup.csv"

INDICATORS = [
    "SP500",
    "DGS10",
    "UNRATE",
    "CPIAUCSL",
    "DCOILWTICO",
    "NASDAQCOM",
]

START = "2018-01-01"
END = "2023-12-29"


def main():
    # Получаем исходные значения без заполнения пропусков.
    raw = web.DataReader(INDICATORS, "fred", START, END)
    raw = raw.sort_index()

    if raw.index.has_duplicates:
        raise ValueError("В данных обнаружены повторяющиеся даты.")

    if raw.empty or raw["SP500"].dropna().empty:
        raise ValueError("Не удалось получить данные S&P 500.")

    # Сохраняем исходные данные для проверки.
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(DATA_PATH.parent / "fred_economic_data_raw.csv")

    # Используем только даты с реальным наблюдением S&P 500.
    # Выходные и праздники не превращаем в торговые дни.
    trading_dates = raw.index[raw["SP500"].notna()]

    # Заполняем остальные показатели предыдущим известным значением.
    # Делаем это ДО выбора торговых дат, чтобы сохранить обновления,
    # которые могли приходиться на другие даты.
    data = raw.ffill().loc[trading_dates, INDICATORS]

    # Удаляем начальные строки, для которых ещё нет всех показателей.
    data = data.dropna()

    if data.empty:
        raise ValueError("После обработки не осталось данных.")

    returns = data["SP500"].pct_change(fill_method=None)
    if (returns.abs() > 0.30).any():
        raise ValueError(
            "Обнаружены изменения S&P 500 больше 30%. "
            "CSV не заменён: проверь исходные данные."
        )

    # Сохраняем старый CSV перед заменой.
    if DATA_PATH.exists() and not BACKUP_PATH.exists():
        shutil.copy2(DATA_PATH, BACKUP_PATH)

    data.index.name = "DATE"
    data.to_csv(DATA_PATH)

    print(f"Saved: {DATA_PATH}")
    print(f"Shape: {data.shape}")
    print(f"Dates: {data.index.min().date()} — {data.index.max().date()}")
    print(f"Missing values: {int(data.isna().sum().sum())}")
    print(f"Max absolute daily return: {returns.abs().max():.2%}")


if __name__ == "__main__":
    main()