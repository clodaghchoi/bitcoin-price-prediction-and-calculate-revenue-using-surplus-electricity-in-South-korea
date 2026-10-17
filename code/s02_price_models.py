"""Step 2. Bitcoin price prediction with Random Forest Regressor and LSTM.

Timing convention
  A row dated t holds the indicators computed from prices up to day t.
  Its target is the price on day t+1 (the "target date").
  Training set: target dates 2016 .. 2022-12-31.
  Test set:     target dates 2023-01-01 .. 2023-12-31 (365 days).
  The LSTM input for target date t+1 is the 14 rows dated t-13 .. t. For the
  first days of 2023 these rows lie in late 2022; they contain only prices
  known before the target date, so there is no look-ahead.

Produces
  tables/table_rf_grid_search.csv
  tables/table_model_params.csv        (best hyperparameters)
  tables/table_prediction_results.csv  (Table: Bitcoin price prediction results)
  tables/predicted_prices_2023.csv     (daily actual / RF / LSTM prices, used in step 4)
  figures/fig05_price_predictions.png
"""
import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")   # fewer floating-point differences across CPUs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler

import common as M
import config as C
from features import add_indicators


def build_dataset() -> pd.DataFrame:
    bitcoin = M.load_bitcoin()
    df = bitcoin.loc[C.PRICE_START:C.TEST_END, ["market_price_usd"]]
    gaps = df.index.to_series().diff().dropna()
    assert (gaps == pd.Timedelta(days=1)).all(), "daily price series has missing days"
    df = add_indicators(df)
    df["target"] = df["market_price_usd"].shift(-1)
    df["target_date"] = df.index + pd.Timedelta(days=1)
    df = df.dropna(subset=C.FEATURES + ["target"])
    return df


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def fit_random_forest(train, test):
    search = GridSearchCV(
        RandomForestRegressor(random_state=C.RANDOM_SEED),
        C.RF_PARAM_GRID,
        cv=KFold(n_splits=C.RF_CV_FOLDS, shuffle=False),
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(train[C.FEATURES], train["target"])
    cv = pd.DataFrame(search.cv_results_)[
        ["param_n_estimators", "param_max_depth", "param_min_samples_split", "mean_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")
    cv["mean_cv_mse"] = -cv.pop("mean_test_score")
    M.save_table(cv, "table_rf_grid_search", index=False)
    pred = search.best_estimator_.predict(test[C.FEATURES])
    return pred, search.best_params_


def fit_lstm(df, train_mask, test_mask):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(C.RANDOM_SEED)
    tf.config.experimental.enable_op_determinism()
    # One thread: the order of floating-point sums then does not depend on the
    # number of CPU cores, so results match across machines.
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    cols = C.FEATURES + ["target"]
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df.loc[train_mask, cols])   # fit on training rows only
    scaled = scaler.transform(df[cols])
    L = C.LSTM_SEQUENCE_LENGTH

    def windows(mask):
        rows = np.where(mask)[0]
        rows = rows[rows >= L - 1]
        X = np.stack([scaled[r - L + 1:r + 1, :-1] for r in rows])
        y = scaled[rows, -1]
        return X, y, rows

    X_tr, y_tr, _ = windows(train_mask.values)
    X_te, _, rows_te = windows(test_mask.values)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(L, len(C.FEATURES))),
        tf.keras.layers.LSTM(C.LSTM_UNITS, return_sequences=True),
        tf.keras.layers.Dropout(C.LSTM_DROPOUT),
        tf.keras.layers.LSTM(C.LSTM_UNITS),
        tf.keras.layers.Dropout(C.LSTM_DROPOUT),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_tr, y_tr, epochs=C.LSTM_EPOCHS, batch_size=C.LSTM_BATCH_SIZE, verbose=0)

    pred_scaled = model.predict(X_te, verbose=0).ravel()
    # undo the 0-1 scaling of the target column
    pred = pred_scaled * scaler.data_range_[-1] + scaler.data_min_[-1]
    assert len(rows_te) == test_mask.sum(), "LSTM must predict every test day"
    return pred


def main() -> None:
    M.ensure_output_dirs()
    M.set_seeds()
    df = build_dataset()
    train_mask = df["target_date"] <= pd.Timestamp(C.TRAIN_END)
    test_mask = (df["target_date"] >= pd.Timestamp(C.TEST_START)) & (df["target_date"] <= pd.Timestamp(C.TEST_END))
    train, test = df[train_mask], df[test_mask]

    rf_pred, best = fit_random_forest(train, test)
    lstm_pred = fit_lstm(df, train_mask, test_mask)

    pred = pd.DataFrame(
        {"actual": test["target"].values, "random_forest": rf_pred, "lstm": lstm_pred},
        index=pd.DatetimeIndex(test["target_date"].values, name="date"),
    )
    M.save_table(pred, "predicted_prices_2023")

    res = pd.DataFrame(
        {
            "Random Forest Regressor": metrics(pred["actual"], pred["random_forest"]),
            "LSTM": metrics(pred["actual"], pred["lstm"]),
        }
    ).T
    M.save_table(res, "table_prediction_results")
    params = pd.DataFrame(
        {"Setting": {
            "RF n_estimators (B)": best["n_estimators"],
            "RF max_depth": best["max_depth"],
            "RF min_samples_split": best["min_samples_split"],
            "RF CV scoring": "MSE, 5-fold unshuffled",
            "LSTM architecture": "LSTM(50, seq) -> Dropout(0.2) -> LSTM(50) -> Dropout(0.2) -> Dense(1)",
            "LSTM optimizer / loss": "Adam / MSE",
            "LSTM epochs / batch": f"{C.LSTM_EPOCHS} / {C.LSTM_BATCH_SIZE}",
            "Training target dates": f"{train['target_date'].min().date()} to {train['target_date'].max().date()}",
            "Test target dates": f"{test['target_date'].min().date()} to {test['target_date'].max().date()}",
        }}
    )
    M.save_table(params, "table_model_params")
    M.update_numbers(
        rf_best_params=best,
        train_rows=int(train_mask.sum()),
        test_rows=int(test_mask.sum()),
        train_first_target_date=str(train["target_date"].min().date()),
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pred.index, pred["actual"], color="green", linewidth=2, label="Actual Prices")
    ax.plot(pred.index, pred["random_forest"], color="red", linestyle="--", linewidth=2,
            label="Predicted Prices (Random Forest)")
    ax.plot(pred.index, pred["lstm"], color="orange", linestyle="--", linewidth=2, label="Predicted Prices (LSTM)")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Bitcoin Price (USD, $)", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    M.savefig(fig, "fig05_price_predictions")

    print("Step 2 done: price models")
    print(res.round(2).to_string())
    print("best RF params:", best)


if __name__ == "__main__":
    main()
