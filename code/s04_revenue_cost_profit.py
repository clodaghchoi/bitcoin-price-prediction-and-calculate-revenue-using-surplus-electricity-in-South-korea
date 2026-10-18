"""Step 4. Daily revenue, costs and profit.

Daily mining revenue = price x Bitcoins mined that day, for three price paths
(actual, Random Forest, LSTM) and two simulations. As in the original
analysis, revenue is summed over the 2023 days for which all three price paths
exist (the LSTM has no prediction for the first 14 days), and it is compared
with 12 months of cost. Each predicted price is matched with the day it predicts.

Monthly cost for M miners (Section 4.3)
  depreciation = M x 10,165 / 90;  labor = fixed per tier;
  cooling O&M  = M x 10,165 x yearly rate / 12;  networking = fixed + per-unit x M

Produces
  tables/daily_revenue_2023.csv
  tables/table_monthly_costs.csv
  tables/table_final_profit.csv        (Mid tier)
  tables/table_profit_all_tiers.csv
  figures/fig09-1_daily_revenue_sim1.png
  figures/fig09-2_daily_revenue_sim2.png
  figures/fig10_daily_revenue_rf.png
"""
import matplotlib.pyplot as plt
import pandas as pd

import common as M
import config as C

MODELS = {"actual": "Actual price", "random_forest": "Random Forest Regressor", "lstm": "LSTM"}
SIMS = {"sim1": "Simulation 1", "sim2": "Simulation 2"}


def monthly_cost(n_miners: int, tier: str) -> dict:
    capex = n_miners * C.MINER_UNIT_PRICE_USD
    items = {
        "Depreciation": capex / C.MINER_LIFETIME_MONTHS,
        "Labor": C.LABOR_MONTHLY_USD[tier],
        "Cooling O&M": capex * C.COOLING_RATE_PER_YEAR[tier] / 12,
        "Networking/site": C.NETWORKING_FIXED_USD[tier] + C.NETWORKING_PER_UNIT_USD[tier] * n_miners,
    }
    items["Monthly cost"] = sum(items.values())
    items["Per rig-month"] = items["Monthly cost"] / n_miners
    return items


def main() -> None:
    M.ensure_output_dirs()
    prices = pd.read_csv(C.TABLE_DIR / "predicted_prices_2023.csv", index_col="date", parse_dates=True)
    prices = prices.dropna(subset=["lstm"])                       # days with all three price paths
    daily = pd.read_csv(C.TABLE_DIR / "daily_btc_mined.csv", index_col="datetime", parse_dates=True)
    daily = daily.loc[prices.index]
    assert (abs(daily["market_price_usd"] - prices["actual"]) < 1e-6).all()

    rev = pd.DataFrame(index=prices.index)
    for m in MODELS:
        for s in SIMS:
            rev[f"rev_{m}_{s}"] = prices[m] * daily[f"btc_{s}"]
    M.save_table(rev, "daily_revenue_2023", float_format="%.2f")

    monthly = pd.read_csv(C.TABLE_DIR / "monthly_miners.csv", index_col=0)
    fleet = {"sim1": int(monthly["miners_sim1"].max()), "sim2": int(monthly["miners_sim2"].max())}
    costs = pd.DataFrame({(f"M={fleet[s]:,} ({SIMS[s]})", t): monthly_cost(fleet[s], t)
                          for s in SIMS for t in ["Low", "Mid", "High"]})
    M.save_table(costs, "table_monthly_costs", float_format="%.2f")

    rows = []
    for m, m_name in MODELS.items():
        for s, s_name in SIMS.items():
            revenue, base = rev[f"rev_{m}_{s}"].sum(), rev[f"rev_actual_{s}"].sum()
            for tier in ["Low", "Mid", "High"]:
                cost = C.COST_MONTHS * monthly_cost(fleet[s], tier)["Monthly cost"]
                rows.append({"Case": f"{m_name}, {s_name}", "Tier": tier, "Revenue (M USD)": revenue / 1e6,
                             "Revenue vs actual (%)": 100 * (revenue / base - 1), "Cost (M USD)": cost / 1e6,
                             "Profit (M USD)": (revenue - cost) / 1e6})
    allt = pd.DataFrame(rows)
    M.save_table(allt, "table_profit_all_tiers", index=False)
    final = allt[allt["Tier"] == C.COST_TIER_FOR_PROFIT].drop(columns="Tier").set_index("Case")
    M.save_table(final, "table_final_profit")
    lowest = final["Profit (M USD)"].idxmin()
    M.update_numbers(revenue_days=len(rev), revenue_first_day=str(rev.index.min().date()),
                     revenue_last_day=str(rev.index.max().date()),
                     annual_cost_mid_sim1_musd=float(final.loc["Actual price, Simulation 1", "Cost (M USD)"]),
                     annual_cost_mid_sim2_musd=float(final.loc["Actual price, Simulation 2", "Cost (M USD)"]),
                     lowest_profit_case=lowest, lowest_profit_musd=float(final.loc[lowest, "Profit (M USD)"]),
                     lowest_profit_per_month_musd=float(final.loc[lowest, "Profit (M USD)"] / 12))

    style = {"actual": dict(color="green", linewidth=1, label="Actual Price"),
             "random_forest": dict(color="red", linestyle="--", linewidth=1, label="Random Forest"),
             "lstm": dict(color="orange", linestyle="--", linewidth=1, label="LSTM")}
    for s, fname in [("sim1", "fig09-1_daily_revenue_sim1"), ("sim2", "fig09-2_daily_revenue_sim2")]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for m in MODELS:
            ax.plot(rev.index, rev[f"rev_{m}_{s}"], **style[m])
        M.sci_axis(ax, 6)
        ax.set_xlabel("Date")
        ax.set_ylabel("Mining Revenue per day (USD, $)")
        ax.legend()
        ax.grid(True)
        M.savefig(fig, fname)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rev.index, rev["rev_random_forest_sim1"], label="Random Forest, Simulation 1")
    ax.plot(rev.index, rev["rev_random_forest_sim2"], label="Random Forest, Simulation 2")
    M.sci_axis(ax, 6)
    ax.set_xlabel("Date")
    ax.set_ylabel("Mining Revenue per day (USD, $)")
    ax.legend()
    ax.grid(True)
    M.savefig(fig, "fig10_daily_revenue_rf")
    print("Step 4 done")
    print(final.round(2).to_string())


if __name__ == "__main__":
    main()
