from pathlib import Path
from typing import Any
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import torch
    from lightgbm import LGBMRegressor
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    st.error(
        "PyTorch and LightGBM are required for FX forecasting. Install the "
        "project dependencies with `pip install -r requirements.txt`, then "
        "restart Streamlit."
    )
    st.stop()


st.set_page_config(
    page_title="FX Rates Prediction",
    page_icon="💱",
    layout="wide",
)

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "cross_border_payments_100k.csv"
)
TIMESTAMP_COLUMN = "transaction_timestamp"
SOURCE_CURRENCY_COLUMN = "source_currency"
DESTINATION_CURRENCY_COLUMN = "destination_currency"
RATE_COLUMN = "fx_rate_inr_per_unit"
REQUIRED_COLUMNS = {
    TIMESTAMP_COLUMN,
    SOURCE_CURRENCY_COLUMN,
    DESTINATION_CURRENCY_COLUMN,
    RATE_COLUMN,
}
MIN_DAILY_OBSERVATIONS = 100
LGBM_LOOKBACK = 30


class LSTMForecaster(nn.Module):
    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.lstm(inputs)
        return self.output(sequence_output[:, -1, :]).squeeze(-1)


@st.cache_data(show_spinner=False)
def load_data(data_path: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    return pd.read_csv(data_path)


def prepare_daily_series(
    data: pd.DataFrame,
    source_currency: str,
    destination_currency: str,
) -> pd.Series:
    pair_data = data.loc[
        (data[SOURCE_CURRENCY_COLUMN] == source_currency)
        & (data[DESTINATION_CURRENCY_COLUMN] == destination_currency),
        [TIMESTAMP_COLUMN, RATE_COLUMN],
    ].copy()
    pair_data[TIMESTAMP_COLUMN] = pd.to_datetime(
        pair_data[TIMESTAMP_COLUMN],
        errors="coerce",
    )
    pair_data[RATE_COLUMN] = pd.to_numeric(pair_data[RATE_COLUMN], errors="coerce")
    pair_data = pair_data.dropna(subset=[TIMESTAMP_COLUMN, RATE_COLUMN])
    pair_data = pair_data[
        np.isfinite(pair_data[RATE_COLUMN]) & (pair_data[RATE_COLUMN] > 0)
    ]
    if pair_data.empty:
        return pd.Series(dtype=float, name=RATE_COLUMN)

    pair_data["date"] = pair_data[TIMESTAMP_COLUMN].dt.floor("D")
    daily_rates = (
        pair_data.groupby("date")[RATE_COLUMN]
        .mean()
        .sort_index()
        .resample("D")
        .mean()
        .ffill()
    )
    daily_rates.name = RATE_COLUMN
    return daily_rates.dropna()


def make_lightgbm_features(
    history: list[float],
    target_date: pd.Timestamp,
) -> dict[str, float]:
    features = {
        f"lag_{lag}": float(history[-lag])
        for lag in (1, 2, 3, 7, 14, LGBM_LOOKBACK)
    }
    for window in (7, 14, LGBM_LOOKBACK):
        recent_values = np.asarray(history[-window:], dtype=float)
        features[f"mean_{window}"] = float(recent_values.mean())
        features[f"std_{window}"] = float(recent_values.std())

    day_of_year = target_date.dayofyear
    features["day_of_week"] = float(target_date.dayofweek)
    features["month"] = float(target_date.month)
    features["year_sin"] = float(np.sin(2 * np.pi * day_of_year / 365.25))
    features["year_cos"] = float(np.cos(2 * np.pi * day_of_year / 365.25))
    return features


def predict_next_rate(
    history: list[float],
    target_date: pd.Timestamp,
    lstm_model: LSTMForecaster,
    lightgbm_model: LGBMRegressor,
    lookback: int,
    scaler_mean: float,
    scaler_scale: float,
) -> tuple[float, float, float]:
    scaled_window = (
        np.asarray(history[-lookback:], dtype=np.float32) - scaler_mean
    ) / scaler_scale
    sequence = torch.from_numpy(scaled_window).reshape(1, lookback, 1)
    lstm_model.eval()
    with torch.no_grad():
        lstm_scaled_prediction = float(lstm_model(sequence).item())
    lstm_prediction = lstm_scaled_prediction * scaler_scale + scaler_mean

    lightgbm_features = pd.DataFrame(
        [make_lightgbm_features(history, target_date)]
    )
    lightgbm_prediction = float(lightgbm_model.predict(lightgbm_features)[0])
    hybrid_prediction = (lstm_prediction + lightgbm_prediction) / 2
    return hybrid_prediction, lstm_prediction, lightgbm_prediction


def train_hybrid_model(
    daily_rates: pd.Series,
    lookback: int,
    epochs: int,
    learning_rate: float,
) -> dict[str, Any]:
    values = daily_rates.to_numpy(dtype=np.float64)
    split_index = int(len(values) * 0.8)
    if split_index <= max(lookback, LGBM_LOOKBACK) + 10:
        raise ValueError(
            "The selected pair needs more history for the training window. "
            "Choose a pair with at least 100 daily observations."
        )
    if split_index >= len(values):
        raise ValueError("The selected pair needs a non-empty validation period.")

    training_values = values[:split_index]
    scaler_mean = float(training_values.mean())
    scaler_scale = float(training_values.std())
    if scaler_scale < 1e-12:
        scaler_scale = 1.0
    scaled_values = (values - scaler_mean) / scaler_scale

    sequences = np.asarray(
        [
            scaled_values[index - lookback : index]
            for index in range(lookback, split_index)
        ],
        dtype=np.float32,
    )
    targets = np.asarray(
        scaled_values[lookback:split_index],
        dtype=np.float32,
    )
    torch.manual_seed(42)
    lstm_model = LSTMForecaster()
    training_data = TensorDataset(
        torch.from_numpy(sequences).unsqueeze(-1),
        torch.from_numpy(targets),
    )
    batches = DataLoader(
        training_data,
        batch_size=min(32, len(training_data)),
        shuffle=True,
    )
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    lstm_model.train()
    for _ in range(epochs):
        for batch_sequences, batch_targets in batches:
            optimizer.zero_grad()
            predictions = lstm_model(batch_sequences)
            loss = loss_function(predictions, batch_targets)
            loss.backward()
            optimizer.step()

    lightgbm_rows = [
        make_lightgbm_features(
            values[:index].tolist(),
            pd.Timestamp(daily_rates.index[index]),
        )
        for index in range(LGBM_LOOKBACK, split_index)
    ]
    lightgbm_targets = values[LGBM_LOOKBACK:split_index]
    lightgbm_model = LGBMRegressor(
        n_estimators=250,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=5,
        min_child_samples=10,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=2,
        verbosity=-1,
    )
    lightgbm_model.fit(pd.DataFrame(lightgbm_rows), lightgbm_targets)

    validation_history = values[:split_index].tolist()
    hybrid_predictions: list[float] = []
    lstm_predictions: list[float] = []
    lightgbm_predictions: list[float] = []
    validation_dates = daily_rates.index[split_index:]
    for target_date in validation_dates:
        hybrid, lstm_prediction, lightgbm_prediction = predict_next_rate(
            validation_history,
            pd.Timestamp(target_date),
            lstm_model,
            lightgbm_model,
            lookback,
            scaler_mean,
            scaler_scale,
        )
        hybrid_predictions.append(hybrid)
        lstm_predictions.append(lstm_prediction)
        lightgbm_predictions.append(lightgbm_prediction)
        validation_history.append(hybrid)

    validation_actual = values[split_index:]
    persistence_predictions = np.full(
        len(validation_actual),
        values[split_index - 1],
    )
    metrics = {
        "mae": mean_absolute_error(validation_actual, hybrid_predictions),
        "rmse": np.sqrt(mean_squared_error(validation_actual, hybrid_predictions)),
        "lstm_mae": mean_absolute_error(validation_actual, lstm_predictions),
        "lightgbm_mae": mean_absolute_error(
            validation_actual,
            lightgbm_predictions,
        ),
        "persistence_mae": mean_absolute_error(
            validation_actual,
            persistence_predictions,
        ),
        "validation_size": len(validation_actual),
    }
    return {
        "lstm_model": lstm_model,
        "lightgbm_model": lightgbm_model,
        "lookback": lookback,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "metrics": metrics,
        "validation_dates": validation_dates,
        "validation_actual": validation_actual,
        "validation_predictions": np.asarray(hybrid_predictions),
    }


def forecast_rates(
    daily_rates: pd.Series,
    training_result: dict[str, Any],
    horizon: int,
) -> pd.DataFrame:
    history = daily_rates.to_numpy(dtype=float).tolist()
    forecast_dates = pd.date_range(
        start=daily_rates.index[-1] + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
    )
    forecasts: list[float] = []
    for target_date in forecast_dates:
        prediction, _, _ = predict_next_rate(
            history,
            pd.Timestamp(target_date),
            training_result["lstm_model"],
            training_result["lightgbm_model"],
            training_result["lookback"],
            training_result["scaler_mean"],
            training_result["scaler_scale"],
        )
        forecasts.append(prediction)
        history.append(prediction)
    return pd.DataFrame(
        {
            "date": forecast_dates,
            "predicted_fx_rate": forecasts,
        }
    )


st.title("FX Rates Prediction")
st.markdown(
    "Forecast daily currency-pair rates with a **hybrid LSTM + LightGBM** "
    "model. The LSTM learns sequential patterns while LightGBM uses recent "
    "lags, rolling statistics, and calendar features."
)
st.caption(
    "The default source is the project payment dataset, aggregated to daily "
    "pair averages. The forecast preserves the dataset's rate units; it is "
    "an analytical estimate, not financial advice or a live market quote."
)

with st.sidebar:
    st.header("Data and model settings")
    uploaded_file = st.file_uploader(
        "Optional FX transaction CSV",
        type=["csv"],
        help=(
            "Required columns: transaction_timestamp, source_currency, "
            "destination_currency, fx_rate_inr_per_unit."
        ),
    )
    lookback = st.select_slider(
        "LSTM lookback (days)",
        options=[7, 14, 30],
        value=14,
    )
    epochs = st.slider("LSTM training epochs", 10, 100, 30, step=10)
    learning_rate = st.select_slider(
        "LSTM learning rate",
        options=[0.001, 0.003, 0.01],
        value=0.003,
        format_func=lambda value: f"{value:g}",
    )
    horizon = st.slider("Forecast horizon (days)", 7, 60, 30)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        st.stop()
    data_version = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    data_label = uploaded_file.name
else:
    if not DATA_PATH.is_file():
        st.error(f"Could not find the default FX dataset at `{DATA_PATH}`.")
        st.stop()
    try:
        modified_time = DATA_PATH.stat().st_mtime
        data = load_data(str(DATA_PATH), modified_time)
    except (OSError, pd.errors.ParserError) as error:
        st.error(f"Could not load the default FX dataset: {error}")
        st.stop()
    data_version = f"{DATA_PATH}:{modified_time}"
    data_label = DATA_PATH.name

missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
if missing_columns:
    st.error(
        "The FX dataset is missing required columns: "
        f"{', '.join(missing_columns)}"
    )
    st.stop()

data = data.copy()
data[TIMESTAMP_COLUMN] = pd.to_datetime(
    data[TIMESTAMP_COLUMN],
    errors="coerce",
)
data[RATE_COLUMN] = pd.to_numeric(data[RATE_COLUMN], errors="coerce")
for currency_column in (SOURCE_CURRENCY_COLUMN, DESTINATION_CURRENCY_COLUMN):
    data[currency_column] = data[currency_column].astype("string").str.strip().str.upper()
data = data.dropna(
    subset=[
        TIMESTAMP_COLUMN,
        SOURCE_CURRENCY_COLUMN,
        DESTINATION_CURRENCY_COLUMN,
        RATE_COLUMN,
    ]
)
data = data[
    np.isfinite(data[RATE_COLUMN])
    & (data[RATE_COLUMN] > 0)
    & (data[SOURCE_CURRENCY_COLUMN] != "")
    & (data[DESTINATION_CURRENCY_COLUMN] != "")
]
if data.empty:
    st.error("The dataset has no valid, positive FX rates to forecast.")
    st.stop()

currency_pairs = sorted(
    set(
        zip(
            data[SOURCE_CURRENCY_COLUMN].astype(str),
            data[DESTINATION_CURRENCY_COLUMN].astype(str),
        )
    )
)
st.caption(f"Data source: `{data_label}` · {len(data):,} valid rate observations")

pair = st.selectbox(
    "Currency pair",
    currency_pairs,
    format_func=lambda currencies: f"{currencies[0]} → {currencies[1]}",
)
source_currency, destination_currency = pair
daily_rates = prepare_daily_series(data, source_currency, destination_currency)
if len(daily_rates) < MIN_DAILY_OBSERVATIONS:
    st.warning(
        f"{source_currency}/{destination_currency} has {len(daily_rates)} daily "
        f"observations. Select a pair with at least {MIN_DAILY_OBSERVATIONS} "
        "daily observations to train and validate the model."
    )
    st.stop()

summary_columns = st.columns(3)
summary_columns[0].metric("Daily observations", f"{len(daily_rates):,}")
summary_columns[1].metric(
    "Latest dataset rate",
    f"{daily_rates.iloc[-1]:.6f}",
)
summary_columns[2].metric(
    "History covered",
    f"{daily_rates.index[0]:%Y-%m-%d} to {daily_rates.index[-1]:%Y-%m-%d}",
)

train_button = st.sidebar.button(
    "Train hybrid model",
    type="primary",
    use_container_width=True,
)
model_key = (
    f"{data_version}:{source_currency}:{destination_currency}:"
    f"{lookback}:{epochs}:{learning_rate}"
)
if train_button:
    with st.spinner("Training the LSTM and LightGBM on chronological history..."):
        st.session_state["fx_rates_training_result"] = train_hybrid_model(
            daily_rates,
            int(lookback),
            epochs,
            learning_rate,
        )
        st.session_state["fx_rates_model_key"] = model_key

training_result = st.session_state.get("fx_rates_training_result")
model_is_current = (
    training_result is not None
    and st.session_state.get("fx_rates_model_key") == model_key
)

st.subheader("Chronological holdout validation")
if model_is_current:
    metrics = training_result["metrics"]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Hybrid MAE", f"{metrics['mae']:.6f}")
    metric_columns[1].metric("Hybrid RMSE", f"{metrics['rmse']:.6f}")
    metric_columns[2].metric("LSTM MAE", f"{metrics['lstm_mae']:.6f}")
    metric_columns[3].metric("LightGBM MAE", f"{metrics['lightgbm_mae']:.6f}")
    st.caption(
        f"Validation uses the final 20% of dates ({metrics['validation_size']:,} "
        "days). Forecasts roll forward recursively from the training cutoff; "
        f"the persistence baseline MAE is {metrics['persistence_mae']:.6f}."
    )

    history_for_chart = daily_rates.tail(180).rename("Observed rate").to_frame()
    validation_frame = pd.DataFrame(
        {
            "Observed validation rate": training_result["validation_actual"],
            "Hybrid validation forecast": training_result[
                "validation_predictions"
            ],
        },
        index=training_result["validation_dates"],
    )
    chart_data = history_for_chart.join(validation_frame, how="outer")
    st.line_chart(
        chart_data,
        x_label="Date",
        y_label="FX rate (dataset units)",
    )

    forecast_frame = forecast_rates(daily_rates, training_result, horizon)
    st.subheader(f"{horizon}-day FX rate forecast")
    st.line_chart(
        forecast_frame.set_index("date"),
        y="predicted_fx_rate",
        x_label="Date",
        y_label="Predicted FX rate (dataset units)",
    )
    st.dataframe(forecast_frame, use_container_width=True, hide_index=True)
    st.download_button(
        "Download forecast CSV",
        data=forecast_frame.to_csv(index=False).encode("utf-8"),
        file_name=f"{source_currency}_{destination_currency}_fx_forecast.csv",
        mime="text/csv",
    )
elif training_result is not None:
    st.info("Data or model settings changed. Train the hybrid model again.")
else:
    st.info("Choose the currency pair and model settings, then train the model.")

st.caption(
    "Calendar dates without observations are forward-filled from the last "
    "available rate. Validation and future forecasts are recursive; uncertainty "
    "bands are not provided."
)