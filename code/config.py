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
DATA_END = "2023-12-31"        # last day of Bitcoin data used
# Rows are indexed by the date t on which the indicators are known; the
# target is the price on day t+1. Periods as in the original analysis:
TRAIN_START, TRAIN_END = "2016-01-16", "2022-12-31"   # first row with all indicators .. end of 2022
TEST_START, TEST_END = "2023-01-01", "2023-12-30"     # 2023 (the target of 2023-12-30 is 2023-12-31)
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
SOLAR_HOURS_PER_DAY = 6.3         # annual average daily sunshine hours in Korea (Nematollahi and Kim, 2017)
MINER_POWER_KW = 5.676            # Antminer S21 XP Hyd., 5,676 W (BITMAIN, 2024)
MINER_HASH_RATE_THS = 473         # Antminer S21 XP Hyd., 473 TH/s (BITMAIN, 2024)
BLOCK_REWARD_BTC = 6.25           # valid 2020-05-12 to 2024-04-20 (Table 7)
BLOCKS_PER_DAY = 144              # one block per ~10 minutes

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
