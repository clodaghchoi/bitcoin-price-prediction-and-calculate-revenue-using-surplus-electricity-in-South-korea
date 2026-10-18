"""Step 3. Number of miners and expected number of Bitcoins mined.

Assumptions (all set in config.py; see Sections 3 and 4.2 of the paper)
  1. Usable energy in month m = national surplus kWh x (1 - transmission loss rate 3.59%).
  2. Miners run only during sunshine hours: SOLAR_HOURS_PER_DAY = 6.3 hours per day.
  3. Miners available in month m:
         N_m = floor( usable kWh in m / (5.676 kW x 6.3 h x days in m) )
  4. Simulation 1: N_m miners run in month m (maximum = largest N_m).
     Simulation 2: the fleet is the average of N_m over 2021-2023;
                   min(N_m, average) miners run in month m.
  5. Bitcoins mined per day:
         BTC_d = 6.25 x 144 x (6.3 / 24) x (running miners x 473 TH/s) / network hash rate_d
     The factor 6.3 / 24 is the share of the day during which the miners run.

Produces
  tables/monthly_miners.csv
  tables/daily_btc_mined.csv
  tables/table_simulation_settings.csv
  figures/fig07_monthly_miners.png
  figures/fig08_daily_btc_mined.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as M
import config as C


def monthly_miners() -> pd.DataFrame:
    total = M.monthly_national_surplus(M.load_surplus())
    out = pd.DataFrame({"surplus_kwh": total})
    out["usable_kwh"] = out["surplus_kwh"] * (1 - C.TRANSMISSION_LOSS_RATE)
    out["days"] = out.index.days_in_month
    out["miners_available"] = np.floor(
        out["usable_kwh"] / (C.MINER_POWER_KW * C.SOLAR_HOURS_PER_DAY * out["days"])).astype(int)
    fleet_avg = int(round(out["miners_available"].mean()))
    out["miners_sim1"] = out["miners_available"]
    out["miners_sim2"] = np.minimum(out["miners_available"], fleet_avg)
    out.attrs.update(fleet_sim1=int(out["miners_available"].max()), fleet_sim2=fleet_avg)
    return out


def daily_btc(monthly: pd.DataFrame) -> pd.DataFrame:
    b = M.load_bitcoin()
    d = b.loc[monthly.index.min():monthly.index.max() + pd.offsets.MonthEnd(0), ["market_price_usd", "hash_rate"]].copy()
    month = d.index.to_period("M").to_timestamp()
    for sim in ["sim1", "sim2"]:
        miners = monthly[f"miners_{sim}"].reindex(month).values
        d[f"miners_{sim}"] = miners
        d[f"btc_{sim}"] = (C.BLOCK_REWARD_BTC * C.BLOCKS_PER_DAY * (C.SOLAR_HOURS_PER_DAY / 24)
                           * miners * C.MINER_HASH_RATE_THS / d["hash_rate"])
    return d


def main() -> None:
    M.ensure_output_dirs()
    monthly = monthly_miners()
    f1, f2 = monthly.attrs["fleet_sim1"], monthly.attrs["fleet_sim2"]
    M.save_table(monthly, "monthly_miners")
    daily = daily_btc(monthly)
    M.save_table(daily, "daily_btc_mined", float_format="%.8g")

    settings = pd.DataFrame({"Number of miners": [f1, f2],
                             "Rule": ["Monthly number of available miners (maximum)",
                                      "Average monthly number of available miners over 2021-2023"]},
                            index=["Simulation 1", "Simulation 2"])
    M.save_table(settings, "table_simulation_settings")
    M.update_numbers(fleet_sim1_max=f1, fleet_sim2_average=f2,
                     miners_available_mean_exact=float(monthly["miners_available"].mean()),
                     peak_month=monthly["miners_available"].idxmax().strftime("%Y-%m"),
                     solar_hours_per_day=C.SOLAR_HOURS_PER_DAY, transmission_loss_rate=C.TRANSMISSION_LOSS_RATE)

    # Figure 7 (original style): miners available per month and their average
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(monthly.index, monthly["miners_available"], marker="o", label="Number of units available per month")
    ax.axhline(f2, color="red", linestyle="--", label=f"Average: {f2:,}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of units")
    ax.legend()
    ax.grid(True)
    M.savefig(fig, "fig07_monthly_miners")

    # Figure 8: Bitcoins mined per day, 2021-2023
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(daily.index, daily["btc_sim1"], label="Daily Bitcoin Mined [Simulation 1]")
    ax.plot(daily.index, daily["btc_sim2"], label="Daily Bitcoin Mined [Simulation 2]")
    ax.set_xlabel("Date")
    ax.set_ylabel("No. of Bitcoin Mined per Day")
    ax.legend()
    ax.grid(True)
    M.savefig(fig, "fig08_daily_btc_mined")
    print(f"Step 3 done: Simulation 1 maximum = {f1:,}, Simulation 2 average = {f2:,}")


if __name__ == "__main__":
    main()
