"""Step 1. Data description.

Produces
  tables/table_surplus_stats.csv   (Table: descriptive statistics for surplus electricity)
  tables/table_bitcoin_stats.csv   (Table: descriptive statistics for Bitcoin data)
  figures/fig02_bitcoin_price_all.png
  figures/fig04_monthly_surplus.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

import common as M
import config as C
from features import add_indicators


def stats(series: pd.Series) -> dict:
    return {"Mean": series.mean(), "Max": series.max(), "Std. Dev.": series.std()}


def main() -> None:
    M.ensure_output_dirs()
    surplus = M.load_surplus()
    bitcoin = M.load_bitcoin()

    # --- Table: surplus electricity (one row per legal dong and month) ---
    t1 = pd.DataFrame(
        {
            "Electricity Power (kWh)": stats(surplus["surplus_kwh"]),
            "Total Households": stats(surplus["households"]),
        }
    ).T
    M.save_table(t1, "table_surplus_stats")

    # --- Table: Bitcoin data, over the price-modelling sample ---
    # The sample starts once all technical indicators are defined.
    sample = add_indicators(bitcoin.loc[C.PRICE_START:C.TEST_END, ["market_price_usd"]]).dropna()
    window = bitcoin.loc[sample.index.min():sample.index.max()]
    t2 = pd.DataFrame(
        {
            "Market Price (USD)": stats(window["market_price_usd"]),
            "Hash Rate (TH/s)": stats(window["hash_rate"]),
        }
    ).T
    M.save_table(t2, "table_bitcoin_stats")
    M.update_numbers(
        surplus_rows_used=len(surplus),
        surplus_rows_suppressed=surplus.attrs["rows_suppressed"],
        bitcoin_stats_start=str(window.index.min().date()),
        bitcoin_stats_end=str(window.index.max().date()),
    )

    # --- Figure 2: Bitcoin price over the whole period ---
    price = bitcoin.loc[:C.TEST_END, "market_price_usd"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(price.index, price.values, linewidth=0.8, label="Market price (USD)")
    ax.axvspan(pd.Timestamp(C.PRICE_START), pd.Timestamp(C.TEST_END), color="gray", alpha=0.2,
               label=f"Analysis period ({C.PRICE_START[:4]}-{C.TEST_END[:4]})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Market Price ($)")
    ax.legend(loc="upper left")
    ax.grid(True)
    M.savefig(fig, "fig02_bitcoin_price_all")

    # --- Figure 4: monthly national surplus electricity ---
    monthly = M.monthly_national_surplus(surplus)
    x = np.arange(len(monthly))
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = make_interp_spline(x, monthly.values, k=3)(x_smooth)
    dates_smooth = pd.to_datetime(np.linspace(monthly.index.min().value, monthly.index.max().value, 300))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dates_smooth, y_smooth, linewidth=2, label="Total surplus electricity (smoothed)")
    ax.plot(monthly.index, monthly.values, "o", markersize=4, color="C0", label="Monthly total")
    ticks = pd.date_range(monthly.index.min(), monthly.index.max(), freq="3MS")
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks.strftime("%Y-%m"), rotation=45)
    M.sci_axis(ax, 8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Surplus Electricity (kWh)")
    ax.legend()
    ax.grid(True)
    M.savefig(fig, "fig04_monthly_surplus")

    print("Step 1 done: descriptive tables, Figure 2, Figure 4")


if __name__ == "__main__":
    main()
