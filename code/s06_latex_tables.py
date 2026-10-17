"""Step 6. Write the result tables as LaTeX rows in the manuscript's format.

Produces
  tables/latex_tables.tex
"""
import pandas as pd

import common as M
import config as C


def fmt(x, d=0):
    return f"{x:,.{d}f}"


def main() -> None:
    T = C.TABLE_DIR
    out = []

    s = pd.read_csv(T / "table_surplus_stats.csv", index_col=0)
    out.append("% Table: Descriptive statistics for surplus electricity data (2021-2023)")
    out.append(f"Electricity Power (kWh) & {fmt(s.iloc[0,0])} & {fmt(s.iloc[0,1])} & {fmt(s.iloc[0,2])} \\\\")
    out.append(f"Total Households & {fmt(s.iloc[1,0],1)} & {fmt(s.iloc[1,1])} & {fmt(s.iloc[1,2],1)} \\\\")

    b = pd.read_csv(T / "table_bitcoin_stats.csv", index_col=0)
    out.append("\n% Table: Descriptive statistics for bitcoin data")
    out.append(f"Market Price (USD) & {fmt(b.iloc[0,0])} & {fmt(b.iloc[0,1])} & {fmt(b.iloc[0,2])} \\\\")
    out.append(f"Hash Rate (TH/s) & {fmt(b.iloc[1,0])} & {fmt(b.iloc[1,1])} & {fmt(b.iloc[1,2])} \\\\")

    p = pd.read_csv(T / "table_prediction_results.csv", index_col=0)
    out.append("\n% Table: Bitcoin price prediction results")
    for name, r in p.iterrows():
        out.append(f"{name} & {fmt(r.MAE)} & {fmt(r.MSE)} & {r.R2:.2f} \\\\")

    c = pd.read_csv(T / "table_monthly_costs.csv", header=[0, 1], index_col=0)
    groups = list(dict.fromkeys(c.columns.get_level_values(0)))
    out.append("\n% Table: Monthly costs (USD) for baseline")
    out.append("% header: " + " | ".join(groups))
    for item in c.index:
        d = 2 if item == "Per rig-month" else 0
        vals = " & ".join(fmt(v, d) for v in c.loc[item].values)
        label = f"\\textbf{{{item}}}" if item == "Monthly cost" else item
        out.append(f"{label} & {vals} \\\\")

    f = pd.read_csv(T / "table_final_profit.csv", index_col=0)
    out.append(f"\n% Table: Final profit comparison ({C.COST_TIER_FOR_PROFIT} cost tier, unit: million USD)")
    for case, r in f.iterrows():
        rev = fmt(r["Revenue (M USD)"])
        if not case.startswith("Actual"):
            rev += f" ({r['Revenue vs actual (%)']:+.1f}\\%)"
        out.append(f"{case} & {rev} & {r['Cost (M USD)']:.2f} & {r['Profit (M USD)']:.2f} \\\\")

    st = pd.read_csv(T / "table_simulation_settings.csv", index_col=0)
    out.append("\n% Table: Simulation settings")
    for name, r in st.iterrows():
        out.append(f"{name} & {fmt(r['Fleet size (miners)'])} & "
                   f"{100 * r['Share of 2021-2022 months in which the whole fleet runs']:.0f}\\% \\\\")

    lf = pd.read_csv(T / "table_sensitivity_lifetime.csv")
    out.append("\n% Table: Sensitivity to miner lifetime (actual price, Mid tier, million USD)")
    for _, r in lf.iterrows():
        out.append(f"{int(r['Lifetime (months)'])} & {r['Simulation 1']:.2f} & {r['Simulation 2']:.2f} \\\\")

    sh = pd.read_csv(T / "table_sensitivity_hours.csv")
    out.append("\n% Table: Sensitivity to operating hours (actual price, Mid tier, million USD)")
    for _, r in sh.iterrows():
        out.append(f"{r['Capacity basis']} & {r['Hours per day']:.2f} & "
                   f"{fmt(r['Simulation 1 fleet'])} & {r['Simulation 1 profit (M USD)']:.2f} & "
                   f"{fmt(r['Simulation 2 fleet'])} & {r['Simulation 2 profit (M USD)']:.2f} & "
                   f"{fmt(r['Best in hindsight fleet'])} & {r['Best in hindsight profit (M USD)']:.2f} \\\\")

    (T / "latex_tables.tex").write_text("\n".join(out) + "\n")
    print("Step 6 done: LaTeX table rows")


if __name__ == "__main__":
    main()
