"""Shared helpers: data loading, output saving, plot style."""
import json
import os
import random

import matplotlib

matplotlib.use("Agg")  # write files, never open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C


def set_seeds(seed: int = C.RANDOM_SEED) -> None:
    """Fix every random number generator used in the analysis."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_output_dirs() -> None:
    C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_bitcoin() -> pd.DataFrame:
    """Daily Bitcoin network data from blockchain.com, indexed by date."""
    df = pd.read_csv(C.BITCOIN_FILE, parse_dates=["datetime"])
    return df.set_index("datetime").sort_index()


def load_surplus() -> pd.DataFrame:
    """KEPCO monthly surplus electricity by legal dong (2021-2023).

    The provider suppresses rows with fewer than five households: the
    household column then holds the text '5호미만제거' ("removed: fewer than
    5 households") and the surplus column is empty. These rows are dropped.
    """
    df = pd.read_csv(C.SURPLUS_FILE, encoding="utf-8-sig").rename(columns=C.SURPLUS_COLUMNS)
    df["households"] = pd.to_numeric(df["households"], errors="coerce")
    df["surplus_kwh"] = pd.to_numeric(df["surplus_kwh"], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["surplus_kwh", "households"]).copy()
    df.attrs["rows_suppressed"] = n_before - len(df)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    return df


def monthly_national_surplus(surplus: pd.DataFrame) -> pd.Series:
    """Total surplus electricity (kWh) for the whole country, per month."""
    return surplus.groupby("date")["surplus_kwh"].sum().sort_index()


def save_table(df: pd.DataFrame, name: str, index: bool = True, float_format: str = "%.4f") -> None:
    df.to_csv(C.TABLE_DIR / f"{name}.csv", index=index, float_format=float_format)


def update_numbers(**values) -> None:
    """Collect the numbers that appear in the manuscript text in one JSON file."""
    path = C.OUT_DIR / "numbers_in_text.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    for k, v in values.items():
        data[k] = v.item() if hasattr(v, "item") else v
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def savefig(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / f"{name}.png", dpi=C.FIG_DPI)
    plt.close(fig)


def sci_axis(ax, power: int) -> None:
    """Show the y axis as x 10^power, like the original figures."""
    import matplotlib.ticker as mticker

    fmt = mticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((power, power))
    ax.yaxis.set_major_formatter(fmt)
