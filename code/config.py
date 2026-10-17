"""All assumptions and parameters used in the paper, in one place.

Each value has a comment that says where it comes from. Change a value here
and re-run `python code/run_all.py` to see how the results change.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"

BITCOIN_FILE = DATA_DIR / "bitcoin_daily_blockchain_com.csv"
SURPLUS_FILE = DATA_DIR / "kepco_surplus_electricity_2021_2023.csv"
# Optional file, only needed for the regional map (Figure 3); see README Section 6.
DISTRICT_GEOJSON_FILE = DATA_DIR / "geo" / "sigungu_boundaries.geojson"

# English names for the Korean column headers of the KEPCO file
# (the raw file is kept exactly as downloaded).
SURPLUS_COLUMNS = {
    "시도": "province",
    "시군구": "district",
    "법정동": "legal_dong",
    "년도": "year",
    "월": "month",
    "전체호수": "households",
    "잉여전력량": "surplus_kwh",
}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Bitcoin price prediction (Section 4.1)
# ---------------------------------------------------------------------------
PRICE_START = "2016-01-01"     # price changes before 2016 are negligible (Section 3.1)
TRAIN_END = "2022-12-31"       # last TARGET date in the training set
TEST_START = "2023-01-01"      # first TARGET date in the test set
TEST_END = "2023-12-31"        # last TARGET date in the test set
INDICATOR_WINDOW = 14          # SMA, WMA, K%, RSI window (Table 3)
D_WINDOW = 3                   # D% = 3-day SMA of K% (Table 3)
MOMENTUM_LAG = 1               # N = 1 (Table 3)
FEATURES = ["SMA_14", "WMA_14", "Momentum", "K%", "D%", "RSI"]

RF_PARAM_GRID = {              # Table 4
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5, 10],
}
RF_CV_FOLDS = 5                # unshuffled K-fold = contiguous time blocks

LSTM_SEQUENCE_LENGTH = 14      # previous 14 days as input (Section 4.1)
LSTM_UNITS = 50                # Table 4
LSTM_DROPOUT = 0.2             # Table 4
LSTM_EPOCHS = 20               # Table 4
LSTM_BATCH_SIZE = 32           # Table 4

# ---------------------------------------------------------------------------
# Mining simulation (Sections 3 and 4.2)
# ---------------------------------------------------------------------------
TRANSMISSION_LOSS_RATE = 0.0359   # KEPCO 10-year average 2012-2021 (Kim, 2022)

# Operating hours per day = equivalent full-load hours of photovoltaic (PV)
# systems in Korea in 2023: annual PV generation / (average installed capacity x 365).
# Source: Korea Energy Agency, 2023 New and Renewable Energy Supply Statistics
# (published 2024), table 2, sheet "4.2 태양광", column 전국 (whole country).
PV_GENERATION_2023_MWH = 33_236_446.9        # row "2023년 발전량(MWh)"
PV_CAPACITY_END_2023_KW = 28_033_335.7       # row "2023년 누적 보급용량(kW)"
PV_NEW_CAPACITY_2023_KW = 3_682_247.4        # row "최근 3년간 신규 보급용량(kW)", '23
PV_CAPACITY_AVG_2023_KW = PV_CAPACITY_END_2023_KW - PV_NEW_CAPACITY_2023_KW / 2   # mean of start and end of 2023
OPERATING_HOURS_PER_DAY = PV_GENERATION_2023_MWH * 1000 / (PV_CAPACITY_AVG_2023_KW * 365)   # = 3.48 h
# Sensitivity: capacity at the end of 2023 (3.25 h) and at the start of 2023 (3.74 h)
SENSITIVITY_HOURS = {
    "End-of-year capacity": PV_GENERATION_2023_MWH * 1000 / (PV_CAPACITY_END_2023_KW * 365),
    "Average capacity (base case)": PV_GENERATION_2023_MWH * 1000 / (PV_CAPACITY_AVG_2023_KW * 365),
    "Start-of-year capacity": PV_GENERATION_2023_MWH * 1000 / ((PV_CAPACITY_END_2023_KW - PV_NEW_CAPACITY_2023_KW) * 365),
}

MINER_POWER_KW = 5.676            # Antminer S21 XP Hyd., 5,676 W (BITMAIN, 2024)
MINER_HASH_RATE_THS = 473         # Antminer S21 XP Hyd., 473 TH/s (BITMAIN, 2024)
BLOCK_REWARD_BTC = 6.25           # valid 2020-05-12 to 2024-04-20 (Table 6)
BLOCK_REWARD_AFTER_2024_HALVING_BTC = 3.125
BLOCKS_PER_DAY = 144              # one block per ~10 minutes

# Fleet sizing is decided at the start of 2023 with information available then:
SIZING_START = "2021-01-01"       # surplus electricity pattern used for sizing
SIZING_END = "2022-12-31"
SIZING_ECONOMICS_YEAR = "2022"    # most recent full year of Bitcoin price and hash rate
SIZING_COST_TIER = "Mid"

# The fleets are then evaluated over 2023:
SIMULATION_START = "2023-01-01"
SIMULATION_END = "2023-12-31"

# ---------------------------------------------------------------------------
# Costs (Section 4.3)
# ---------------------------------------------------------------------------
MINER_UNIT_PRICE_USD = 10_165     # BITMAIN list price
MINER_LIFETIME_MONTHS = 90        # 7.5 years, straight-line depreciation
LABOR_MONTHLY_USD = {"Low": 16_000, "Mid": 24_000, "High": 32_000}
COOLING_RATE_PER_YEAR = {"Low": 0.005, "Mid": 0.010, "High": 0.020}   # share of miner CAPEX per year
NETWORKING_FIXED_USD = {"Low": 1_000, "Mid": 2_000, "High": 5_000}     # per month
NETWORKING_PER_UNIT_USD = {"Low": 0.5, "Mid": 1.0, "High": 2.0}        # per miner per month
COST_TIER_FOR_PROFIT = "Mid"
COST_MONTHS = 12

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
FIG_DPI = 300
