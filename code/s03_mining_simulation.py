"""Step 3. Number of miners, fleet sizing, and Bitcoins mined.

Assumptions (all set in config.py)
  1. Usable energy in month m = national surplus kWh x (1 - transmission loss rate).
  2. Miners run at full power for h = OPERATING_HOURS_PER_DAY hours per day, the
     equivalent full-load hours of PV systems in Korea (3.48 h in 2023).
  3. Miners that the surplus can run in month m:
         N_m = floor( usable kWh in m / (MINER_POWER_KW x h x days in m) )
  4. Bitcoins mined on day d by R running miners:
         H_d   = R x 473 TH/s x h / 24          (fleet hash power averaged over the day)
         BTC_d = H_d / (H_net,d + H_d) x (144 x block reward + transaction fees_d)
     The fleet's hash power is added to the network hash rate H_net,d, because
     Bitcoin's difficulty adjusts (every 2,016 blocks, about two weeks) so that
     a miner's share of blocks equals its share of all hashes computed. Miners
     also receive the transaction fees of the blocks they find.
  5. The fleet size M is chosen at the start of 2023, using only data up to 2022:
       Simulation 1 (average-based sizing): M = mean of N_m over 2021-2022.
       Simulation 2 (profit-maximising sizing): M maximises the profit over the
       24 months of 2021-2022, with min(N_m, M) miners running in month m and
       2022 Bitcoin prices, network hash rate and fees. An extra miner earns
       revenue only in months in which it runs but costs money every month, so
       it is worth buying only if it runs in a large enough share of the months
       (about c/r, the critical ratio of the newsvendor problem).
  6. In month m of 2023, min(N_m, M) miners run.

Produces
  tables/monthly_miners.csv
  tables/daily_btc_mined.csv
  tables/table_simulation_settings.csv
  tables/revenue_per_miner_month.csv
  figures/fig07_monthly_miners.png
  figures/fig08_daily_btc_mined.png
"""
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as M
import config as C

SIMS = {"sim1": "Simulation 1 (average-based)", "sim2": "Simulation 2 (profit-maximising)"}


@lru_cache(maxsize=None)
def bitcoin() -> pd.DataFrame:
    return M.load_bitcoin()[["market_price_usd", "hash_rate", "total_transaction_fees"]]


def monthly_cost(n_miners: int, tier: str, lifetime_months: float = C.MINER_LIFETIME_MONTHS) -> float:
    """Monthly cost (USD) of a fleet of n miners; see s04 for the items."""
    capex = n_miners * C.MINER_UNIT_PRICE_USD
    return (capex / lifetime_months + C.LABOR_MONTHLY_USD[tier]
            + capex * C.COOLING_RATE_PER_YEAR[tier] / 12
            + C.NETWORKING_FIXED_USD[tier] + C.NETWORKING_PER_UNIT_USD[tier] * n_miners)


def miners_available(hours: float = C.OPERATING_HOURS_PER_DAY) -> pd.Series:
    total = M.monthly_national_surplus(M.load_surplus())
    usable = total * (1 - C.TRANSMISSION_LOSS_RATE)
    n = np.floor(usable / (C.MINER_POWER_KW * hours * total.index.days_in_month)).astype(int)
    n.name = "miners_available"
    return n


def btc_per_day(running, days: pd.DatetimeIndex, hours: float = C.OPERATING_HOURS_PER_DAY,
                block_reward: float = C.BLOCK_REWARD_BTC, fees: bool = True, network: bool = True) -> np.ndarray:
    """Bitcoins mined on each day by `running` miners (scalar or one value per day)."""
    b = bitcoin().loc[days]
    h_fleet = np.asarray(running, dtype=float) * C.MINER_HASH_RATE_THS * hours / 24
    share = h_fleet / (b["hash_rate"].values + (h_fleet if network else 0.0))
    per_day_btc = C.BLOCKS_PER_DAY * block_reward + (b["total_transaction_fees"].values if fees else 0.0)
    return share * per_day_btc


def running_miners(n: pd.Series, fleet: int, days: pd.DatetimeIndex) -> np.ndarray:
    return np.minimum(n.reindex(days.to_period("M").to_timestamp()).values, fleet)


def planning_profit(m: int, n_past: pd.Series, hours: float, tier: str = C.SIZING_COST_TIER) -> float:
    """Profit over the sizing months if every day had the average 2022 economics (prices, hash rate, fees)."""
    days = bitcoin().loc[C.SIZING_ECONOMICS_YEAR].index
    price = bitcoin().loc[days, "market_price_usd"].values
    revenue = 0.0
    for r, n_days in zip(np.minimum(n_past.values, m), n_past.index.days_in_month):
        revenue += (price * btc_per_day(r, days, hours)).mean() * n_days
    return revenue - len(n_past) * monthly_cost(m, tier)


def size_fleets(n: pd.Series, hours: float = C.OPERATING_HOURS_PER_DAY) -> dict:
    """Fleet sizes chosen with data up to 2022 only."""
    n_past = n.loc[C.SIZING_START:C.SIZING_END]
    avg = int(round(n_past.mean()))
    lo, hi = int(n_past.min()), int(n_past.max())
    coarse = range(lo, hi + 1, 1000)
    best = max(coarse, key=lambda m: planning_profit(m, n_past, hours))
    fine = range(max(lo, best - 1000), min(hi, best + 1000) + 1, 10)
    best = max(fine, key=lambda m: planning_profit(m, n_past, hours))
    if planning_profit(best, n_past, hours) <= 0:
        best = 0
    # revenue of one running miner per month in 2022 when the whole fleet runs (for the critical ratio)
    days = bitcoin().loc[C.SIZING_ECONOMICS_YEAR].index
    price = bitcoin().loc[days, "market_price_usd"].values
    r = (price * btc_per_day(best, days, hours)).mean() * 365 / 12 / best if best else float("nan")
    c = monthly_cost(best, C.SIZING_COST_TIER) / best if best else float("nan")
    return dict(sim1=avg, sim2=int(best), r_planning=r, unit_cost_planning=c, critical_ratio=c / r,
                share_months_running_sim2=float((n_past >= best).mean()),
                share_months_running_sim1=float((n_past >= avg).mean()))


def revenue_per_running_miner_month(fleet: int, hours: float = C.OPERATING_HOURS_PER_DAY,
                                    block_reward: float = C.BLOCK_REWARD_BTC) -> pd.Series:
    """USD per month earned by one miner of a fleet of `fleet` miners that all run (actual prices)."""
    b = bitcoin().loc["2021":"2023"]
    daily = b["market_price_usd"].values * btc_per_day(fleet, b.index, hours, block_reward) / fleet
    out = pd.Series(daily, index=b.index).groupby(b.index.to_period("M")).sum()
    out.index = out.index.to_timestamp()
    return out


def main() -> None:
    M.ensure_output_dirs()
    n = miners_available()
    fleets = size_fleets(n)
    monthly = pd.DataFrame({"miners_available": n})
    for s in SIMS:
        monthly[f"miners_{s}"] = np.minimum(n, fleets[s])
    M.save_table(monthly, "monthly_miners")

    days = bitcoin().loc[C.SIMULATION_START:C.SIMULATION_END].index
    daily = bitcoin().loc[days].copy()
    for s in SIMS:
        daily[f"miners_{s}"] = running_miners(n, fleets[s], days)
        daily[f"btc_{s}"] = btc_per_day(daily[f"miners_{s}"].values, days)
    M.save_table(daily, "daily_btc_mined", float_format="%.8g")

    r = revenue_per_running_miner_month(fleets["sim2"])
    r_year = r.groupby(r.index.year).mean()
    M.save_table(pd.DataFrame({"revenue_per_running_miner_month_usd": r}), "revenue_per_miner_month")

    settings = pd.DataFrame(
        {"Fleet size (miners)": [fleets["sim1"], fleets["sim2"]],
         "Rule": ["Mean of the monthly number of available miners, 2021-2022",
                  "Maximises 2021-2022 profit with 2022 prices, hash rate and fees"],
         "Share of 2021-2022 months in which the whole fleet runs":
             [fleets["share_months_running_sim1"], fleets["share_months_running_sim2"]]},
        index=list(SIMS.values()))
    M.save_table(settings, "table_simulation_settings")

    n23 = n.loc[C.SIMULATION_START:C.SIMULATION_END]
    M.update_numbers(
        operating_hours_per_day=C.OPERATING_HOURS_PER_DAY,
        capacity_factor=C.OPERATING_HOURS_PER_DAY / 24,
        pv_capacity_avg_2023_mw=C.PV_CAPACITY_AVG_2023_KW / 1000,
        pv_generation_2023_gwh=C.PV_GENERATION_2023_MWH / 1000,
        sensitivity_hours=C.SENSITIVITY_HOURS,
        miners_available_mean_2021_2023=float(n.mean()),
        miners_available_mean_2021_2022=float(n.loc[C.SIZING_START:C.SIZING_END].mean()),
        miners_available_min=int(n.min()), miners_available_min_month=n.idxmin().strftime("%Y-%m"),
        miners_available_max=int(n.max()), miners_available_max_month=n.idxmax().strftime("%Y-%m"),
        fleet_sim1=fleets["sim1"], fleet_sim2=fleets["sim2"],
        sizing_revenue_per_miner_month_2022=fleets["r_planning"],
        sizing_unit_cost_per_miner_month=fleets["unit_cost_planning"],
        sizing_critical_ratio=fleets["critical_ratio"],
        share_months_running_sim1=fleets["share_months_running_sim1"],
        share_months_running_sim2=fleets["share_months_running_sim2"],
        revenue_per_running_miner_month_by_year={str(k): float(v) for k, v in r_year.items()},
        months_2023_sim1_fully_used=int((n23 >= fleets["sim1"]).sum()),
        months_2023_sim2_fully_used=int((n23 >= fleets["sim2"]).sum()),
        btc_mined_2023_sim1=float(daily["btc_sim1"].sum()),
        btc_mined_2023_sim2=float(daily["btc_sim2"].sum()),
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n.index, n.values, marker="o", label="Number of units available per month")
    ax.axhline(fleets["sim1"], color="red", linestyle="--", label=f"Simulation 1: {fleets['sim1']:,} units")
    ax.axhline(fleets["sim2"], color="purple", linestyle=":", label=f"Simulation 2: {fleets['sim2']:,} units")
    ax.axvspan(pd.Timestamp(C.SIMULATION_START), n.index.max() + pd.offsets.MonthEnd(0), color="gray", alpha=0.12,
               label="Evaluation year (2023)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of units")
    ax.legend()
    ax.grid(True)
    M.savefig(fig, "fig07_monthly_miners")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(daily.index, daily["btc_sim1"], label="Daily Bitcoin Mined [Simulation 1]")
    ax.plot(daily.index, daily["btc_sim2"], label="Daily Bitcoin Mined [Simulation 2]")
    ax.set_xlabel("Date")
    ax.set_ylabel("No. of Bitcoin Mined per Day")
    ax.legend()
    ax.grid(True)
    M.savefig(fig, "fig08_daily_btc_mined")

    print(f"Step 3 done: {C.OPERATING_HOURS_PER_DAY:.2f} h/day; fleet Simulation 1 = {fleets['sim1']:,}, "
          f"Simulation 2 = {fleets['sim2']:,} (critical ratio {fleets['critical_ratio']:.2f})")


if __name__ == "__main__":
    main()
