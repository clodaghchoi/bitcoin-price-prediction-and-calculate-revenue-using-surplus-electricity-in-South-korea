"""Step 2. Bitcoin price prediction with Random Forest Regressor and LSTM.

This follows the original analysis:
  - A row dated t holds the indicators computed from prices up to day t; its
    target is the price on day t+1.
  - Training rows: 2016-01-16 .. 2022-12-31. Test rows: 2023-01-01 .. 2023-12-30.
  - Random Forest: grid search (5-fold cross-validation, mean squared error).
  - LSTM: input = the previous 14 test rows, so the first 14 test rows have no
    LSTM prediction ("the initial 14 days are absent", Section 4.1).
Random number generators are fixed and the LSTM runs on one CPU thread, so the
results are the same on every run.

Produces
  tables/table_rf_grid_search.csv
  tables/table_model_params.csv
  tables/table_prediction_results.csv  (Table: Bitcoin price prediction results)
  tables/predicted_prices_2023.csv     (by target date; used in step 4)
  figures/fig05_price_predictions.png
"""
import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
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
    df = M.load_bitcoin().loc[C.PRICE_START:C.DATA_END, ["market_price_usd"]]
    df = add_indicators(df).dropna()
    df["target"] = df["market_price_usd"].shift(-1)
    return df


def metrics(y_true, y_pred) -> dict:
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mean_squared_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)}


def fit_random_forest(train, test):
    search = GridSearchCV(RandomForestRegressor(random_state=C.RANDOM_SEED), C.RF_PARAM_GRID,
                          cv=KFold(n_splits=C.RF_CV_FOLDS, shuffle=False),
                          scoring="neg_mean_squared_error", n_jobs=-1)
    search.fit(train[C.FEATURES], train["target"])
    cv = pd.DataFrame(search.cv_results_)[
        ["param_n_estimators", "param_max_depth", "param_min_samples_split", "mean_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")
    cv["mean_cv_mse"] = -cv.pop("mean_test_score")
    M.save_table(cv, "table_rf_grid_search", index=False)
    return search.best_estimator_.predict(test[C.FEATURES]), search.best_params_


def fit_lstm(train, test):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(C.RANDOM_SEED)
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    cols = C.FEATURES + ["target"]
    scaler = MinMaxScaler(feature_range=(0, 1))
    tr = scaler.fit_transform(train[cols])
    te = scaler.transform(test[cols])
    L = C.LSTM_SEQUENCE_LENGTH

    def sequences(a):
        X = np.stack([a[i:i + L, :-1] for i in range(len(a) - L)])
        return X, a[L:, -1]

    X_tr, y_tr = sequences(tr)
    X_te, _ = sequences(te)
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
    return pred_scaled * scaler.data_range_[-1] + scaler.data_min_[-1]   # predictions for test rows L, L+1, ...


def main() -> None:
    M.ensure_output_dirs()
    M.set_seeds()
    df = build_dataset()
    train = df.loc[C.TRAIN_START:C.TRAIN_END]
    test = df.loc[C.TEST_START:C.TEST_END]

    rf_pred, best = fit_random_forest(train, test)
    lstm_pred = fit_lstm(train, test)
    L = C.LSTM_SEQUENCE_LENGTH

    pred = pd.DataFrame({"feature_date": test.index, "actual": test["target"].values, "random_forest": rf_pred,
                         "lstm": np.r_[np.full(L, np.nan), lstm_pred]})
    pred.index = pd.DatetimeIndex(test.index + pd.Timedelta(days=1), name="date")   # target date
    M.save_table(pred, "predicted_prices_2023")

    lstm_rows = pred.dropna(subset=["lstm"])
    res = pd.DataFrame({"Random Forest Regressor": metrics(pred["actual"], pred["random_forest"]),
                        "LSTM": metrics(lstm_rows["actual"], lstm_rows["lstm"])}).T
    res["Days evaluated"] = [len(pred), len(lstm_rows)]
    M.save_table(res, "table_prediction_results")
    params = pd.DataFrame({"Setting": {
        "RF n_estimators (B)": best["n_estimators"], "RF max_depth": best["max_depth"],
        "RF min_samples_split": best["min_samples_split"], "RF CV scoring": "MSE, 5-fold unshuffled",
        "LSTM architecture": "LSTM(50, seq) -> Dropout(0.2) -> LSTM(50) -> Dropout(0.2) -> Dense(1)",
        "LSTM optimizer / loss": "Adam / MSE", "LSTM epochs / batch": f"{C.LSTM_EPOCHS} / {C.LSTM_BATCH_SIZE}",
        "Training rows": f"{train.index.min().date()} to {train.index.max().date()}",
        "Test rows": f"{test.index.min().date()} to {test.index.max().date()}"}})
    M.save_table(params, "table_model_params")
    M.update_numbers(rf_best_params=best, train_rows=len(train), test_rows=len(test),
                     lstm_prediction_days=len(lstm_rows))

    # Figure 5 (as in the original: x axis = date of the input row)
    x = pred["feature_date"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, pred["actual"], label="Actual Prices", color="green", linewidth=2)
    ax.plot(x, pred["random_forest"], label="Predicted Prices (Random Forest)", color="red", linestyle="--", linewidth=2)
    ax.plot(x, pred["lstm"], label="Predicted Prices (LSTM)", color="orange", linestyle="--", linewidth=2)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Bitcoin Price (USD, $)", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    M.savefig(fig, "fig05_price_predictions")

    print("Step 2 done: price models")
    print(res.round(4).to_string())
    print("best RF params:", best)


if __name__ == "__main__":
    main()
