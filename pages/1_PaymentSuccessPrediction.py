from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    from catboost import CatBoostClassifier  # type: ignore[reportMissingImports]
except ImportError:
    st.error(
        "CatBoost is not installed. Install the project dependencies with "
        "`pip install -r requirements.txt`, then restart Streamlit."
    )
    st.stop()


st.set_page_config(
    page_title="Payment Success Prediction",
    page_icon="✅",
    layout="wide",
)

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "cross_border_payments_100k.csv"
)
TARGET = "payment_success"
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
    data: pd.DataFrame,
    iterations: int,
    depth: int,
    learning_rate: float,
) -> dict[str, Any]:
    labeled_data = data.dropna(subset=[TARGET])
    target = labeled_data[TARGET].astype(int)
    features = prepare_features(labeled_data)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        x_train,
        y_train,
        cat_features=CATEGORICAL_FEATURES,
        eval_set=(x_test, y_test),
        early_stopping_rounds=50,
    )

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
        "feature_importance": pd.DataFrame(
            {
                "Feature": FEATURES,
                "Importance": model.get_feature_importance(),
            }
        ).sort_values("Importance", ascending=False),
        "test_size": len(y_test),
    }
    return {"model": model, "metrics": metrics}


st.title("Payment Success Prediction")
st.markdown(
    "Train a **CatBoost classifier** to estimate whether a cross-border "
    "payment will succeed. CatBoost uses the country, bank, currency, payment "
    "rail and other categorical fields directly."
)
st.caption(
    "The model uses transaction-time information only. Transaction IDs, "
    "timestamps, settlement outcomes, SLA breaches and anomaly labels are "
    "excluded to prevent target leakage."
)

if not DATA_PATH.is_file():
    st.error(f"Could not find the payment dataset at `{DATA_PATH}`.")
    st.stop()

try:
    data = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except (OSError, pd.errors.ParserError) as error:
    st.error(f"Could not load the payment dataset: {error}")
    st.stop()

missing_columns = sorted(set(FEATURES + [TARGET]) - set(data.columns))
if missing_columns:
    st.error(f"The dataset is missing required columns: {', '.join(missing_columns)}")
    st.stop()

data = data.dropna(subset=[TARGET])
data[TARGET] = data[TARGET].astype(int)
if data[TARGET].nunique() != 2:
    st.error("The payment_success target must contain both 0 and 1 classes.")
    st.stop()

success_rate = data[TARGET].mean()
summary_columns = st.columns(3)
summary_columns[0].metric("Labeled transactions", f"{len(data):,}")
summary_columns[1].metric("Observed success rate", f"{success_rate:.1%}")
summary_columns[2].metric("Categorical features", len(CATEGORICAL_FEATURES))

with st.sidebar:
    st.header("Model settings")
    iterations = st.slider("Boosting iterations", 100, 1000, 500, step=100)
    depth = st.slider("Tree depth", 4, 10, 7)
    learning_rate = st.slider(
        "Learning rate",
        min_value=0.01,
        max_value=0.3,
        value=0.05,
        step=0.01,
        format="%.2f",
    )
    train_button = st.button(
        "Train CatBoost model",
        type="primary",
        use_container_width=True,
    )

model_key = (
    f"{DATA_PATH.stat().st_mtime}:{iterations}:{depth}:{learning_rate}"
)
if train_button:
    with st.spinner("Training CatBoost and evaluating on a held-out test set..."):
        st.session_state["payment_success_training_result"] = train_model(
            data,
            iterations,
            depth,
            learning_rate,
        )
        st.session_state["payment_success_model_key"] = model_key

training_result = st.session_state.get("payment_success_training_result")
model_is_current = (
    training_result is not None
    and st.session_state.get("payment_success_model_key") == model_key
)

if model_is_current:
    metrics = training_result["metrics"]
    st.subheader("Held-out evaluation")
    metric_columns = st.columns(5)
    for column, label, key in zip(
        metric_columns,
        ["ROC AUC", "Accuracy", "Precision", "Recall", "F1 score"],
        ["roc_auc", "accuracy", "precision", "recall", "f1"],
    ):
        column.metric(label, f"{metrics[key]:.3f}")

    results_left, results_right = st.columns(2)
    with results_left:
        st.markdown("**Confusion matrix**")
        st.dataframe(
            pd.DataFrame(
                metrics["confusion_matrix"],
                index=["Actual failure", "Actual success"],
                columns=["Predicted failure", "Predicted success"],
            ),
            use_container_width=True,
        )
    with results_right:
        st.markdown("**Feature importance**")
        st.bar_chart(
            metrics["feature_importance"].set_index("Feature"),
            x_label="Importance",
            y_label="Feature",
            horizontal=True,
        )
    st.caption(
        f"Evaluation uses a stratified 80/20 split. "
        f"Metrics are based on {metrics['test_size']:,} held-out transactions."
    )
elif training_result is not None:
    st.info(
        "Model settings changed. Train the model again before making predictions."
    )
else:
    st.info("Choose model settings in the sidebar, then train the model to continue.")

st.subheader("Score a payment")
if model_is_current:
    st.write("Enter the information available when the payment is initiated.")
    form_values: dict[str, Any] = {}
    with st.form("payment_prediction_form"):
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
        for index, feature in enumerate(NUMERIC_FEATURES):
            values = pd.to_numeric(data[feature], errors="coerce").dropna()
            minimum = float(values.min())
            maximum = float(values.max())
            default = float(values.median())
            if minimum == maximum:
                minimum -= 1
                maximum += 1

            if feature in {"intermediary_count", "transaction_hour", "day_of_week"}:
                with number_columns[index % 3]:
                    form_values[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        min_value=int(minimum),
                        max_value=int(maximum),
                        value=int(round(default)),
                        step=1,
                    )
            else:
                step = max((maximum - minimum) / 100, 0.01)
                with number_columns[index % 3]:
                    form_values[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        min_value=minimum,
                        max_value=maximum,
                        value=min(max(default, minimum), maximum),
                        step=step,
                        format="%.4f" if "rate" in feature else "%.2f",
                    )

        predict_button = st.form_submit_button(
            "Predict payment outcome",
            type="primary",
        )

    if predict_button:
        prediction_row = pd.DataFrame([form_values], columns=FEATURES)
        prediction_row = prepare_features(prediction_row)
        success_probability = float(
            training_result["model"].predict_proba(prediction_row)[0, 1]
        )
        outcome_column, success_column, failure_column = st.columns(3)
        outcome_column.metric(
            "Predicted outcome",
            "Likely to succeed" if success_probability >= 0.5 else "At risk of failure",
        )
        success_column.metric("Success probability", f"{success_probability:.1%}")
        failure_column.metric("Failure risk", f"{1 - success_probability:.1%}")
        st.caption("Outcome label uses a 0.5 probability threshold.")
else:
    st.caption("Train the model to enable the payment scoring form.")