"""Technical indicators used as model inputs (Table: development of independent variables).

Only daily closing prices are available, so the 14-day high and low used for
K% are the highest and lowest daily prices in the window.
"""
import numpy as np
import pandas as pd

import config as C


def add_indicators(df: pd.DataFrame, price_col: str = "market_price_usd") -> pd.DataFrame:
    out = df.copy()
    p = out[price_col]
    w = C.INDICATOR_WINDOW

    out["SMA_14"] = p.rolling(w).mean()

    weights = np.arange(1, w + 1)  # most recent day has the largest weight
    out["WMA_14"] = p.rolling(w).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    out["Momentum"] = p.diff(C.MOMENTUM_LAG)

    low, high = p.rolling(w).min(), p.rolling(w).max()
    out["K%"] = 100 * (p - low) / (high - low)
    out["D%"] = out["K%"].rolling(C.D_WINDOW).mean()

    delta = p.diff()
    avg_gain = delta.clip(lower=0).rolling(w).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(w).mean()
    out["RSI"] = 100 - 100 / (1 + avg_gain / avg_loss)
    return out
