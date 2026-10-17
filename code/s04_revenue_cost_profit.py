"""Step 4. Revenue, costs and profit over 2023, the profit curve, and sensitivity.

Daily mining revenue = price x Bitcoins mined that day (block subsidy and
transaction fees, with the fleet's hash power added to the network; see s03),
for three price paths (actual, Random Forest, LSTM) and the two simulations.
Revenue is summed over the 365 days of 2023.

Monthly cost for a fleet of M miners (Section 4.3)
  depreciation   = M x unit price / lifetime months
  labor          = fixed amount per tier
  cooling O&M    = M x unit price x yearly rate / 12
  networking     = fixed amount + per-unit amount x M
12-month cost = 12 x monthly cost. Profit = 2023 revenue - 12-month cost.

Also computed
  - profit over 2023 for every fleet size (profit curve) and the best fleet
    size in hindsight (an upper bound, not a usable rule);
  - how much transaction fees and the network adjustment change revenue;
  - revenue per miner after the April 2024 halving;
  - sensitivity to the operating hours and to the miner lifetime, and the
    break-even lifetime.

Produces
  tables/daily_revenue_2023.csv
  tables/table_monthly_costs.csv
  tables/table_final_profit.csv          (Mid tier)
  tables/table_profit_all_tiers.csv
  tables/profit_curve_2023.csv
  tables/table_sensitivity_hours.csv
  tables/table_sensitivity_lifetime.csv
  figures/fig09-1_daily_revenue_sim1.png
  figures/fig09-2_daily_revenue_sim2.png
  figures/fig10_daily_revenue_rf.png
  figures/fig12_profit_curve.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as M
import config as C
from s03_mining_simulation import (SIMS, btc_per_day, miners_available, monthly_cost,
                                   revenue_per_running_miner_month, running_miners, size_fleets)

MODELS = {"actual": "Actual price", "random_forest": "Random Forest Regressor", "lstm": "LSTM"}
SIM_SHORT = {"sim1": "Simulation 1", "sim2": "Simulation 2"}
TIERS = ["Low", "Mid", "High"]
LIFETIMES = [60, 72, 84, 90, 120]


def cost_items(n_miners: int, tier: str) -> dict:
    capex = n_miners * C.MINER_UNIT_PRICE_USD
    items = {
        "Depreciation": capex / C.MINER_LIFETIME_MONTHS,
        "Labor": C.LABOR_MONTHLY_USD[tier],
        "Cooling O&M": capex * C.COOLING_RATE_PER_YEAR[tier] / 12,
        "Networking/site": C.NETWORKING_FIXED_USD[tier] + C.NETWORKING_PER_UNIT_USD[tier] * n_miners,
    }
    items["Monthly cost"] = sum(items.values())
    assert abs(items["Monthly cost"] - monthly_cost(n_miners, tier)) < 1e-6
    items["Per rig-month"] = items["Monthly cost"] / n_miners
    return items


def revenue_2023(fleet: int, n: pd.Series, price: pd.Series, hours: float, **kw) -> float:
    days = price.index
    return float((price.values * btc_per_day(running_miners(n, fleet, days), days, hours, **kw)).sum())


def profit_2023(fleet: int, n: pd.Series, price: pd.Series, tier: str, hours: float,
                lifetime: float = C.MINER_LIFETIME_MONTHS) -> tuple:
    revenue = revenue_2023(fleet, n, price, hours) if fleet > 0 else 0.0
    cost = C.COST_MONTHS * monthly_cost(fleet, tier, lifetime) if fleet > 0 else 0.0
    return revenue, cost, revenue - cost


def best_in_hindsight(n: pd.Series, price: pd.Series, tier: str, hours: float) -> int:
    n23 = n.loc[C.SIMULATION_START:C.SIMULATION_END]
    hi = int(n23.max())
    f = lambda m: profit_2023(m, n, price, tier, hours)[2]
    best = max(range(0, hi + 1, 1000), key=f)
    fine = set(range(max(0, best - 1000), min(hi, best + 1000) + 1, 10)) | set(n23.tolist())
    return max(sorted(fine), key=f)   # the 2023 N_m values are where the profit curve bends


def break_even_lifetime(fleet: int, revenue: float, tier: str = "Mid") -> float:
    """Lifetime (months) at which 2023 profit is zero."""
    capex = fleet * C.MINER_UNIT_PRICE_USD
    other = monthly_cost(fleet, tier) - capex / C.MINER_LIFETIME_MONTHS
    return C.COST_MONTHS * capex / (revenue - C.COST_MONTHS * other)


def main() -> None:
    M.ensure_output_dirs()
    h = C.OPERATING_HOURS_PER_DAY
    prices = pd.read_csv(C.TABLE_DIR / "predicted_prices_2023.csv", index_col="date", parse_dates=True)
    assert len(prices) == 365
    n = miners_available(h)
    fleets = size_fleets(n, h)
    daily = pd.read_csv(C.TABLE_DIR / "daily_btc_mined.csv", index_col="datetime", parse_dates=True)
    assert prices.index.equals(daily.index)

    # --- daily revenue ---------------------------------------------------------
    rev = pd.DataFrame(index=daily.index)
    for m in MODELS:
        for s in SIMS:
            rev[f"rev_{m}_{s}"] = prices[m] * daily[f"btc_{s}"]
    M.save_table(rev, "daily_revenue_2023", float_format="%.2f")

    # --- monthly costs ---------------------------------------------------------
    costs = pd.DataFrame({(f"M={fleets[s]:,} ({SIM_SHORT[s]})", t): cost_items(fleets[s], t)
                          for s in SIMS for t in TIERS})
    M.save_table(costs, "table_monthly_costs", float_format="%.2f")

    # --- profit, all tiers -----------------------------------------------------
    rows = []
    for m, m_name in MODELS.items():
        for s in SIMS:
            for t in TIERS:
                r_, c_, p_ = profit_2023(fleets[s], n, prices[m], t, h)
                assert abs(r_ - rev[f"rev_{m}_{s}"].sum()) < 1e-6 * r_
                base = rev[f"rev_actual_{s}"].sum()
                rows.append({"Case": f"{m_name}, {SIM_SHORT[s]}", "Tier": t,
                             "Revenue (M USD)": r_ / 1e6, "Revenue vs actual (%)": 100 * (r_ / base - 1),
                             "Cost (M USD)": c_ / 1e6, "Profit (M USD)": p_ / 1e6})
    allt = pd.DataFrame(rows)
    M.save_table(allt, "table_profit_all_tiers", index=False)
    final = allt[allt["Tier"] == "Mid"].drop(columns="Tier").set_index("Case")
    M.save_table(final, "table_final_profit")

    # --- profit curve and best fleet in hindsight ------------------------------
    n23 = n.loc[C.SIMULATION_START:C.SIMULATION_END]
    grid = sorted(set(list(range(0, int(n23.max() * 1.06), 2000)) + n23.tolist() + [fleets["sim1"], fleets["sim2"]]))
    hind = best_in_hindsight(n, prices["actual"], "Mid", h)
    grid = sorted(set(grid + [hind]))
    curve = pd.DataFrame({t: [profit_2023(g, n, prices["actual"], t, h)[2] / 1e6 for g in grid] for t in TIERS},
                         index=pd.Index(grid, name="fleet_size"))
    M.save_table(curve, "profit_curve_2023")
    hind_profit = curve.loc[hind, "Mid"]

    # --- revenue components ------------------------------------------------------
    comp = {}
    for s in SIMS:
        full = revenue_2023(fleets[s], n, prices["actual"], h)
        comp[s] = {
            "fees_share_of_revenue": 1 - revenue_2023(fleets[s], n, prices["actual"], h, fees=False) / full,
            "network_adjustment_change": full / revenue_2023(fleets[s], n, prices["actual"], h, network=False) - 1,
        }
        days = prices.index
        h_fleet = running_miners(n, fleets[s], days) * C.MINER_HASH_RATE_THS * h / 24
        comp[s]["fleet_share_of_network_max"] = float((h_fleet / M.load_bitcoin().loc[days, "hash_rate"].values).max())
    r23 = revenue_per_running_miner_month(fleets["sim2"]).loc["2023"].mean()
    r23_after = revenue_per_running_miner_month(fleets["sim2"], h, C.BLOCK_REWARD_AFTER_2024_HALVING_BTC).loc["2023"].mean()

    # --- sensitivity to operating hours ---------------------------------------
    sens = []
    for label, hh in C.SENSITIVITY_HOURS.items():
        nn = miners_available(hh)
        ff = size_fleets(nn, hh)
        row = {"Capacity basis": label, "Hours per day": hh}
        for s in SIMS:
            row[f"{SIM_SHORT[s]} fleet"] = ff[s]
            row[f"{SIM_SHORT[s]} profit (M USD)"] = profit_2023(ff[s], nn, prices["actual"], "Mid", hh)[2] / 1e6
        hb = best_in_hindsight(nn, prices["actual"], "Mid", hh)
        row["Best in hindsight fleet"] = hb
        row["Best in hindsight profit (M USD)"] = profit_2023(hb, nn, prices["actual"], "Mid", hh)[2] / 1e6
        sens.append(row)
    sens = pd.DataFrame(sens)
    M.save_table(sens, "table_sensitivity_hours", index=False)

    # --- sensitivity to miner lifetime -----------------------------------------
    life = pd.DataFrame({SIM_SHORT[s]: [profit_2023(fleets[s], n, prices["actual"], "Mid", h, L)[2] / 1e6
                                        for L in LIFETIMES] for s in SIMS},
                        index=pd.Index(LIFETIMES, name="Lifetime (months)"))
    M.save_table(life, "table_sensitivity_lifetime")
    be = {s: break_even_lifetime(fleets[s], revenue_2023(fleets[s], n, prices["actual"], h)) for s in SIMS}

    lowest = final["Profit (M USD)"].idxmin()
    M.update_numbers(
        annual_cost_mid_sim1_musd=float(final.loc["Actual price, Simulation 1", "Cost (M USD)"]),
        annual_cost_mid_sim2_musd=float(final.loc["Actual price, Simulation 2", "Cost (M USD)"]),
        depreciation_per_miner_month_usd=C.MINER_UNIT_PRICE_USD / C.MINER_LIFETIME_MONTHS,
        depreciation_share_of_cost_mid={s: float(cost_items(fleets[s], "Mid")["Depreciation"]
                                                 / cost_items(fleets[s], "Mid")["Monthly cost"]) for s in SIMS},
        lowest_profit_case=lowest, lowest_profit_musd=float(final.loc[lowest, "Profit (M USD)"]),
        profit_range_all_cases_musd=[float(allt["Profit (M USD)"].min()), float(allt["Profit (M USD)"].max())],
        profit_range_mid_musd=[float(final["Profit (M USD)"].min()), float(final["Profit (M USD)"].max())],
        sim2_minus_sim1_mid_musd={m: float(final.loc[f"{MODELS[m]}, Simulation 2", "Profit (M USD)"]
                                           - final.loc[f"{MODELS[m]}, Simulation 1", "Profit (M USD)"]) for m in MODELS},
        best_in_hindsight_fleet=int(hind), best_in_hindsight_profit_musd=float(hind_profit),
        revenue_components=comp,
        revenue_per_running_miner_month_2023_usd=float(r23),
        revenue_per_running_miner_month_2023_after_halving_usd=float(r23_after),
        break_even_lifetime_months=be,
        revenue_days=len(daily),
    )

    # --- figures -----------------------------------------------------------------
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

    fig, ax = plt.subplots(figsize=(10, 6))
    for t, ls in zip(TIERS, [":", "-", "--"]):
        ax.plot(curve.index, curve[t], linestyle=ls, color="black" if t == "Mid" else "gray",
                label=f"{t} cost tier")
    ax.axhline(0, color="gray", linewidth=0.8)
    for s, mk, col in [("sim1", "o", "red"), ("sim2", "s", "purple")]:
        ax.plot(fleets[s], curve.loc[fleets[s], "Mid"], mk, color=col, markersize=9,
                label=f"{SIM_SHORT[s]} ({fleets[s]:,} units)")
    ax.plot(hind, hind_profit, "^", color="gray", markersize=9, label=f"Best in hindsight ({hind:,} units)")
    ax.set_xlabel("Number of miners purchased")
    ax.set_ylabel("Profit over 2023 (million USD)")
    ax.legend()
    ax.grid(True)
    M.savefig(fig, "fig12_profit_curve")

    print("Step 4 done: revenue, costs, profit")
    print(final.round(2).to_string())
    print(f"best in hindsight: {hind:,} miners, {hind_profit:.1f}M; break-even lifetime {be}")
    print(sens.round(2).to_string(index=False))
    print(life.round(1).to_string())


if __name__ == "__main__":
    main()
