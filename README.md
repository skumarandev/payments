# payments

This repository targets ML models learning with Cross Border payments data.

## Run the Streamlit app

Install the dependencies and start the app from the repository root:

```bash
pip install -r requirements.txt
streamlit run Home.py
```

The **Payment Success Prediction** page trains a CatBoost classifier on
`data/cross_border_payments_100k.csv` and provides held-out evaluation metrics
and an interactive payment scoring form.

The **Settlement Delay Prediction** page trains a Random Forest regressor on
transactions with observed settlement delays. It reports held-out MAE, RMSE,
and R², and provides an interactive settlement-time estimate using
transaction-time inputs.

The **Payment Anomaly Detection** page uses an Isolation Forest to detect
unusual transaction patterns without training on anomaly labels. It reports
held-out metrics against the dataset's synthetic labels and provides an
interactive payment investigation form.

The **Payment Routing Optimisation** page compares payment rails and sending
banks using predicted success likelihood, fee, and settlement delay. It ranks
candidate routes according to reliability, cost, and speed preferences.

The **FX Rates Prediction** page aggregates transaction-level rates by
currency pair and date, trains a hybrid LSTM + LightGBM model, evaluates it on
a chronological holdout, and generates recursive daily forecasts. It can use
the bundled payment dataset or an uploaded CSV with transaction timestamp,
source currency, destination currency, and FX rate columns.