"""Download and load the UCI Occupancy Detection dataset (Candanedo & Feldheim, 2016).

One office room in Mons, Belgium, February 2015, one reading per minute:
temperature (°C), relative humidity (%), light (lux), CO2 (ppm), humidity ratio,
and ground-truth occupancy (0/1, from time-stamped pictures).

Files, in time order:
  datatest.txt      2–4 Feb    (2,665 rows)
  datatraining.txt  4–10 Feb   (8,143 rows)
  datatest2.txt     11–18 Feb  (9,752 rows)

We train on the first two (earlier) and test on datatest2 (later), so the model is
always evaluated on days it has never seen.
"""
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from vsensor import RAW_DIR

URL = "https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip"
CHEAP = ["Temperature", "Humidity", "Light", "HumidityRatio"]   # sensors costing a few euros
TARGET_CO2 = "CO2"
TARGET_OCC = "Occupancy"


def download() -> None:
    if (RAW_DIR / "datatest2.txt").exists():
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive = RAW_DIR / "occupancy.zip"
    urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(RAW_DIR)


def _read(name: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / name, index_col=0, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour of day and weekday as smooth (sin/cos) features, plus short-term trends.

    Trends use only past values (diff over the previous 5 minutes), so nothing leaks
    from the future.
    """
    df = df.copy()
    minutes = df["date"].dt.hour * 60 + df["date"].dt.minute
    df["hour_sin"] = np.sin(2 * np.pi * minutes / 1440)
    df["hour_cos"] = np.cos(2 * np.pi * minutes / 1440)
    df["weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    for col in ["Temperature", "Humidity", "Light"]:
        df[f"{col}_d5"] = df[col].diff(5).fillna(0)
    # CO2 builds up while people are present and decays after they leave.
    # "Lights on" is a cheap proxy for presence; summing it over past windows
    # approximates how much CO2 has accumulated (a physics-inspired feature).
    lights_on = (df["Light"] > 200).astype(float)
    for minutes in (30, 60, 120, 240):
        df[f"lights_on_{minutes}m"] = lights_on.rolling(minutes, min_periods=1).sum()
    df["Humidity_d30"] = df["Humidity"].diff(30).fillna(0)
    return df


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) with time features added."""
    download()
    train = pd.concat([_read("datatest.txt"), _read("datatraining.txt")]).sort_values("date")
    test = _read("datatest2.txt")
    return add_time_features(train.reset_index(drop=True)), add_time_features(test)


def condition_group(df: pd.DataFrame) -> np.ndarray:
    """Which regime each minute is in, using only the cheap light sensor and the clock.

    'night empty'  00:00-06:00 with lights off: nobody there for hours, CO2 should sit near outdoor air
    'lights off'   other dark minutes (evenings, weekends): mostly empty, CO2 may still be decaying
    'lights on'    someone is probably there, or it is daylight
    """
    hour = df["date"].dt.hour
    off = df["Light"] < 50
    return np.where(off & (hour < 6), "night empty", np.where(off, "lights off", "lights on"))
