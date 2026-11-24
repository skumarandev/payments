from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(
    page_title="Payment Anomaly Detection",
    page_icon="🔎",
    layout="wide",
)

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "cross_border_payments_100k.csv"
)
ANOMALY_LABEL = "synthetic_anomaly_label"
TRANSACTION_ID = "transaction_id"
CATEGORICAL_FEATURES = [
    "source_country",
    "destination_country",
    "source_currency",
    "destination_currency",
    "payment_rail",
    "sending_bank",
    "beneficiary_bank",
    "compliance_status",
    "customer_segment",
    "channel",
]
NUMERIC_FEATURES = [
    "amount_source",
    "fx_rate_inr_per_unit",
    "intermediary_count",
    "transaction_hour",
    "day_of_week",
    "transaction_fee_usd_equiv",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


@st.cache_data(show_spinner=False)
def load_data(data_path: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    return pd.read_csv(data_path)


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    features = data[FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].fillna("Unknown").astype(str)
    return features


@st.cache_resource(show_spinner=False)
def train_model(
    features: pd.DataFrame,
    evaluation_labels: pd.Series,
    transaction_ids: pd.Series,
    n_estimators: int,
    contamination: str | float,
) -> dict[str, Any]:
    x_train, x_test = train_test_split(
        features,
        test_size=0.2,
        random_state=42,
    )
    y_test = evaluation_labels.loc[x_test.index].astype(int)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                SimpleImputer(strategy="median"),
                NUMERIC_FEATURES,
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "detector",
                IsolationForest(
                    n_estimators=n_estimators,
                    contamination=contamination,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x_train)

    predictions = model.predict(x_test)
    predicted_anomalies = predictions == -1
    anomaly_scores = -model.decision_function(x_test)
    metrics = {
        "precision": precision_score(
            y_test,
            predicted_anomalies,
            zero_division=0,
        ),
        "recall": recall_score(y_test, predicted_anomalies, zero_division=0),
        "f1": f1_score(y_test, predicted_anomalies, zero_division=0),
        "roc_auc": roc_auc_score(y_test, anomaly_scores),
        "test_size": len(y_test),
        "predicted_anomalies": int(predicted_anomalies.sum()),
    }

    review_rows = x_test.copy()
    review_rows.insert(
        0,
        TRANSACTION_ID,
        transaction_ids.loc[x_test.index].to_numpy(),
    )
    review_rows["actual_anomaly"] = y_test
    review_rows["predicted_anomaly"] = predicted_anomalies.astype(int)
    review_rows["anomaly_score"] = anomaly_scores
    review_rows = review_rows.sort_values("anomaly_score", ascending=False).head(20)

    return {
        "model": model,
        "metrics": metrics,
        "review_rows": review_rows,
    }


st.title("Payment Anomaly Detection")
st.markdown(
    "Use an **Isolation Forest** to identify transactions that differ from "
    "the broader payment population. The detector learns patterns without "
    "using anomaly labels."
)
st.caption(
    "Only transaction-time inputs are used. Synthetic anomaly labels are "
    "reserved for held-out evaluation and are never passed to model fitting."
)

if not DATA_PATH.is_file():
    st.error(f"Could not find the payment dataset at `{DATA_PATH}`.")
    st.stop()

try:
    data = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except (OSError, pd.errors.ParserError) as error:
    st.error(f"Could not load the payment dataset: {error}")
    st.stop()

missing_columns = sorted(
    set(FEATURES + [ANOMALY_LABEL, TRANSACTION_ID]) - set(data.columns)
)
if missing_columns:
    st.error(f"The dataset is missing required columns: {', '.join(missing_columns)}")
    st.stop()

data[ANOMALY_LABEL] = pd.to_numeric(data[ANOMALY_LABEL], errors="coerce")
data = data.dropna(subset=[ANOMALY_LABEL]).copy()
if set(data[ANOMALY_LABEL].unique()) != {0, 1}:
    st.error("The synthetic anomaly label must contain both 0 and 1 values.")
    st.stop()
data[ANOMALY_LABEL] = data[ANOMALY_LABEL].astype(int)

feature_data = prepare_features(data)
labels = data[ANOMALY_LABEL]
summary_columns = st.columns(3)
summary_columns[0].metric("Transactions", f"{len(data):,}")
summary_columns[1].metric(
    "Synthetic anomaly rate",
    f"{labels.mean():.2%}",
)
summary_columns[2].metric("Model input features", len(FEATURES))

with st.sidebar:
    st.header("Model settings")
    n_estimators = st.slider("Number of trees", 100, 500, 100, step=100)
    contamination_label = st.selectbox(
        "Expected anomaly rate",
        ["Automatic (auto)", "1%", "2.5%", "5%", "10%"],
        index=0,
        help=(
            "The expected outlier fraction sets the Isolation Forest decision "
            "threshold. Automatic mode uses scikit-learn's built-in threshold."
        ),
    )
    contamination: str | float = (
        "auto"
        if contamination_label == "Automatic (auto)"
        else float(contamination_label.rstrip("%")) / 100
    )
    train_button = st.button(
        "Train Isolation Forest",
        type="primary",
        use_container_width=True,
    )

model_key = (
    f"{DATA_PATH.stat().st_mtime}:{n_estimators}:{contamination_label}"
)
if train_button:
    with st.spinner("Fitting Isolation Forest and evaluating held-out transactions..."):
        st.session_state["payment_anomaly_training_result"] = train_model(
            feature_data,
            labels,
            data[TRANSACTION_ID],
            n_estimators,
            contamination,
        )
        st.session_state["payment_anomaly_model_key"] = model_key

training_result = st.session_state.get("payment_anomaly_training_result")
model_is_current = (
    training_result is not None
    and st.session_state.get("payment_anomaly_model_key") == model_key
)

if model_is_current:
    metrics = training_result["metrics"]
    st.subheader("Held-out evaluation")
    metric_columns = st.columns(5)
    metric_columns[0].metric("Precision", f"{metrics['precision']:.3f}")
    metric_columns[1].metric("Recall", f"{metrics['recall']:.3f}")
    metric_columns[2].metric("F1 score", f"{metrics['f1']:.3f}")
    metric_columns[3].metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
    metric_columns[4].metric(
        "Flagged transactions",
        f"{metrics['predicted_anomalies']:,} / {metrics['test_size']:,}",
    )
    st.caption(
        "The model is fitted on 80% of the transactions without labels. "
        "Metrics compare predictions on the held-out 20% with synthetic labels."
    )

    st.markdown("**Highest-scoring held-out transactions**")
    st.dataframe(
        training_result["review_rows"],
        use_container_width=True,
        hide_index=True,
    )
elif training_result is not None:
    st.info("Model settings changed. Train the model again before scoring payments.")
else:
    st.info("Choose model settings in the sidebar, then train the model to continue.")

st.subheader("Investigate a payment")
if model_is_current:
    st.write("Enter the information available when the payment is initiated.")
    form_values: dict[str, Any] = {}
    with st.form("payment_anomaly_prediction_form"):
        category_columns = st.columns(2)
        for index, feature in enumerate(CATEGORICAL_FEATURES):
            options = sorted(data[feature].dropna().astype(str).unique().tolist())
            if not options:
                options = ["Unknown"]
            with category_columns[index % 2]:
                form_values[feature] = st.selectbox(
                    feature.replace("_", " ").title(),
                    options,
                )

        number_columns = st.columns(3)
        integer_features = {"intermediary_count", "transaction_hour", "day_of_week"}
        for index, feature in enumerate(NUMERIC_FEATURES):
            values = pd.to_numeric(data[feature], errors="coerce").dropna()
            minimum = float(values.min())
            maximum = float(values.max())
            default = float(values.median())
            if minimum == maximum:
                minimum -= 1
                maximum += 1

            with number_columns[index % 3]:
                if feature in integer_features:
                    form_values[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        min_value=int(np.floor(minimum)),
                        max_value=int(np.ceil(maximum)),
                        value=int(round(default)),
                        step=1,
                    )
                else:
                    form_values[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        min_value=minimum,
                        max_value=maximum,
                        value=min(max(default, minimum), maximum),
                        step=max((maximum - minimum) / 100, 0.01),
                        format="%.4f" if "rate" in feature else "%.2f",
                    )

        predict_button = st.form_submit_button(
            "Check for anomaly",
            type="primary",
        )

    if predict_button:
        prediction_row = pd.DataFrame([form_values], columns=FEATURES)
        prediction_row = prepare_features(prediction_row)
        model = training_result["model"]
        is_anomaly = int(model.predict(prediction_row)[0]) == -1
        anomaly_score = float(-model.decision_function(prediction_row)[0])
        result_columns = st.columns(2)
        result_columns[0].metric(
            "Assessment",
            "Potential anomaly" if is_anomaly else "Typical transaction",
        )
        result_columns[1].metric("Anomaly score", f"{anomaly_score:.4f}")
        st.caption(
            "Higher anomaly scores indicate transactions that are more unusual. "
            "The score is not a probability."
        )
else:
    st.caption("Train the model to enable the payment anomaly investigation form.")