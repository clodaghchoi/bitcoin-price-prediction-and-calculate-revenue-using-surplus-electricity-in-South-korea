# Replication package

**Paper:** Leveraging Surplus Electricity: Profitability of Bitcoin Mining as a National Strategy in South Korea
**Authors:** Yoonseul Choi, Jaehong Jeong, Jungsoon Choi
**Journal:** Energy Economics (manuscript ENEECO-D-25-01705)

This package contains the data and code that produce every table and figure
with numbers in the paper. One command runs the whole analysis, from the raw
data files to the final tables and figures.

## 1. Quick start

```bash
python -m pip install -r requirements.txt
python code/run_all.py
```

The run takes about 2 to 4 minutes on a normal laptop CPU (no GPU needed).
All results are written to `output/`. The folder is deleted and rebuilt on
every run, so old results never mix with new ones.

## 2. Folder structure

```
data/
  bitcoin_daily_blockchain_com.csv          Bitcoin network data (daily)
  kepco_surplus_electricity_2021_2023.csv   Surplus electricity (monthly, by legal dong)
code/
  config.py                    every assumption and parameter, with its source
  common.py                    data loading and helper functions
  features.py                  technical indicators (SMA, WMA, Momentum, K%, D%, RSI)
  s01_data_description.py      descriptive statistics, Figures 2 and 4
  s02_price_models.py          Random Forest and LSTM price prediction, Figure 5
  s03_mining_simulation.py     number of miners and Bitcoins mined, Figures 7 and 8
  s04_revenue_cost_profit.py   revenue, costs, profit, Figures 9 and 10
  s05_regional_map.py          Figure 3 (runs only if the map files are present)
  s06_latex_tables.py          result tables as LaTeX rows
  run_all.py                   runs steps 1-6 in order
  prepare_boundaries.py        (run once, not part of run_all) prepares the downloaded outlines, see Section 6
output/
  tables/    CSV tables, plus latex_tables.tex
  figures/   PNG figures (300 dpi)
  numbers_in_text.json   numbers quoted in the text of the paper
  environment_used.txt   Python and package versions of the last run
```

## 3. Where each result in the paper comes from

| Paper | Output file | Script |
|---|---|---|
| Table: descriptive statistics for surplus electricity | `tables/table_surplus_stats.csv` | s01 |
| Table: descriptive statistics for Bitcoin data | `tables/table_bitcoin_stats.csv` | s01 |
| Table: model hyperparameters (best Random Forest setting) | `tables/table_model_params.csv`, `tables/table_rf_grid_search.csv` | s02 |
| Table: Bitcoin price prediction results | `tables/table_prediction_results.csv` | s02 |
| Table: simulation settings (fleet sizes) | `tables/table_simulation_settings.csv` | s03 |
| Table: monthly costs | `tables/table_monthly_costs.csv` | s04 |
| Table: final profit comparison | `tables/table_final_profit.csv` (Mid tier), `tables/table_profit_all_tiers.csv` | s04 |
| Figure 2: Bitcoin price over the whole period | `figures/fig02_bitcoin_price_all.png` | s01 |
| Figure 3: regional map (needs the outlines, see Section 6) | `figures/fig03_regional_map.png` | s05 |
| Figure 4: monthly total surplus electricity | `figures/fig04_monthly_surplus.png` | s01 |
| Figure 5: predicted vs actual prices | `figures/fig05_price_predictions.png` | s02 |
| Figure 7: number of miners available per month | `figures/fig07_monthly_miners.png` | s03 |
| Figure 8: Bitcoins mined per day (2021-2023) | `figures/fig08_daily_btc_mined.png` | s03 |
| Figure 9: daily revenue, Simulations 1 and 2 | `figures/fig09-1_daily_revenue_sim1.png`, `figures/fig09-2_daily_revenue_sim2.png` | s04 |
| Figure 10: daily revenue, Random Forest | `figures/fig10_daily_revenue_rf.png` | s04 |
| Numbers in the text (fleet sizes, annual costs, lowest profit, etc.) | `numbers_in_text.json` | s01-s04 |

Figure 1 (LSTM cell diagram) and Figure 6 (photo of the miner) contain no data
and are not produced by the code.

## 4. Data

### 4.1 Bitcoin network data — `data/bitcoin_daily_blockchain_com.csv`

- Original source: blockchain.com charts (https://www.blockchain.com/explorer/charts), public data.
- Archived copy: Choi, Y. (2025). *Bitcoin_daily_data_090103_240904* [Data set]. IEEE DataPort.
  https://doi.org/10.21227/ea8x-3298
- Period in the file: 2009-01-03 to 2025-01-03, one row per day. The analysis uses 2016-01-01 to 2023-12-31.
- Columns used: `datetime`, `market_price_usd` (daily average price, USD), `hash_rate` (total network hash rate, TH/s).
  Other columns in the file are not used.

### 4.2 Surplus electricity — `data/kepco_surplus_electricity_2021_2023.csv`

- Source: Korea Electric Power Corporation (KEPCO), "한국전력공사_법정동별 상계거래 잉여전력량"
  (Surplus electricity of net-metering customers by legal dong), Public Data Portal,
  https://www.data.go.kr/data/15111708/fileData.do (version of August 2024).
- Archived copy: Choi, Y. (2025). *KEPCO_Amount of offsetting transaction surplus power
  by statutory district_2020_2023* [Data set]. IEEE DataPort. https://doi.org/10.21227/d74a-qf79
  (this package uses the 2021-2023 file only).
- The file is kept exactly as downloaded (UTF-8 with BOM, Korean column names).
  The code renames the columns to English when it reads the file:

  | Korean header | English name | Meaning |
  |---|---|---|
  | 시도 | province | province or metropolitan city |
  | 시군구 | district | city, county or district |
  | 법정동 | legal_dong | legal neighbourhood (smallest unit) |
  | 년도 | year | year |
  | 월 | month | month |
  | 전체호수 | households | number of households with solar panels under net metering |
  | 잉여전력량 | surplus_kwh | surplus electricity after net metering, kWh per month |

- 169,918 rows (2021-01 to 2023-12). In 15,748 rows the provider has removed
  the values for privacy: `전체호수` holds the text `5호미만제거`
  ("removed: fewer than 5 households") and `잉여전력량` is empty. These rows
  are dropped; 154,170 rows are used.

## 5. Method in brief

All values below are set in `code/config.py`.

**Price prediction (s02).** Six technical indicators computed from each day's
data predict the next day's price. Training rows: 2016-01-16 to 2022-12-31.
Test rows: 2023-01-01 to 2023-12-30, so the predicted days are 2023-01-02 to
2023-12-31. The Random Forest is tuned by grid search (5-fold unshuffled
cross-validation, MSE). The LSTM uses the previous 14 days of indicators,
taken only from the test period, so it has no prediction for the first 14
test days (as stated in the paper). Each predicted price is matched with the
day it predicts.

**Mining simulation (s03).**
1. Usable energy in a month = national surplus kWh × (1 − 3.59% transmission loss).
2. Miners run only during sunshine hours, 6.3 hours per day (Section 3 of the paper).
3. Miners available in month *m*: N_m = floor(usable kWh / (5.676 kW × 6.3 h × days in *m*)).
4. Simulation 1: N_m miners run in month *m*; the fleet is the largest N_m
   over 2021-2023 (166,887, June 2022).
   Simulation 2: the fleet is the average of N_m over 2021-2023 (111,269);
   min(N_m, 111,269) miners run in month *m*.
5. Bitcoins mined per day = 6.25 BTC × 144 blocks × (6.3 / 24) × running hash rate / network hash rate.
   The factor 6.3 / 24 is the share of the day during which the miners run.

**Revenue, cost, profit (s04).** Daily revenue = price × Bitcoins mined, for
the actual price, the Random Forest prediction and the LSTM prediction. Revenue
is summed over the 350 days of 2023 that have all three prices (2023-01-16 to
2023-12-31). Monthly cost = straight-line depreciation of the fleet (USD 10,165
per miner over 90 months) + labor + cooling O&M + networking/site, under
Low / Mid / High tiers. Profit = revenue − 12 months of cost. The paper reports
the Mid tier; all tiers are in `table_profit_all_tiers.csv`.

## 6. Figure 3 (district outlines, downloaded separately)

Figure 3 needs the outlines of the 250 districts (city, county or district;
시군구) of South Korea. They come from Statistics Korea:

> Statistics Korea (Ministry of Data and Statistics), Statistical Geographic
> Information Service (SGIS), "센서스용 행정구역 경계" (census administrative
> boundaries), 2023, 시군구 (district) level, SHP format, coordinate system UTM-K
> (EPSG:5179). https://sgis.mods.go.kr/view/pss/openDataIntrcn (자료제공 → 통계지역경계).
> File used: `bnd_all_00_2023_2Q.zip`, which contains `bnd_sigungu_00_2023_2Q`
> (base date 1 July 2023).

The SGIS terms of use allow free reuse only for material marked with the
Korea Open Government License (KOGL, 공공누리) Type 1, and the boundary files
carry no such mark, so **this package does not include the outlines**. They
are free to download. To make Figure 3:

1. Download the 2023 census administrative boundaries (`bnd_all_00_2023_2Q.zip`)
   from the address above.
2. From the package root, run
   ```bash
   python code/prepare_boundaries.py PATH/TO/bnd_all_00_2023_2Q.zip
   ```
   The zip file does not need to be unzipped. The script writes
   `data/geo/sigungu_boundaries.geojson`: one outline per district,
   simplified to about 100 m (enough for a national map). The province name is
   taken from the first two digits of the SGIS district code.
3. Run `python code/run_all.py` again (or only `python code/s05_regional_map.py`).

All 250 districts in the 2023 SGIS file match the districts in the KEPCO data.
Step 5 prints a warning if a district has data but no outline, or an outline
but no data, which would mean a different version of the outlines was downloaded.

Without the outlines, step 5 prints a message and is skipped. No other table
or figure depends on them.

## 7. Reproducibility notes

- All random number generators are fixed (seed 42). Running the code twice on
  the same machine gives identical output files.
- The LSTM is trained on one CPU thread, so its results do not depend on the
  number of CPU cores. We checked this by running the package on two cores
  and on one core: all output files were identical.
- The Random Forest results do not depend on the hardware. LSTM results can
  differ in the last digits on a different CPU or TensorFlow version, because
  floating-point operations may run in a different order. The exact numbers in
  the paper come from the environment listed in `requirements.txt`.
