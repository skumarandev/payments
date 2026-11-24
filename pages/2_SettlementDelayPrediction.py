from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(
    page_title="Settlement Delay Prediction",
    page_icon="⏱️",
    layout="wide",
)

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "cross_border_payments_100k.csv"
)
TARGET = "settlement_delay_hours"
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
INTEGER_FEATURES = {"intermediary_count", "transaction_hour", "day_of_week"}


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
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
) -> dict[str, Any]:
    labeled_data = data.dropna(subset=[TARGET]).copy()
    labeled_data[TARGET] = pd.to_numeric(labeled_data[TARGET], errors="coerce")
    labeled_data = labeled_data[np.isfinite(labeled_data[TARGET])]

    features = prepare_features(labeled_data)
    target = labeled_data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

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
                "regressor",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    transformed_names = model.named_steps["preprocessor"].get_feature_names_out()
    transformed_importance = pd.DataFrame(
        {
            "transformed_feature": transformed_names,
            "importance": model.named_steps["regressor"].feature_importances_,
        }
    )

    def source_feature(transformed_name: str) -> str:
        feature_name = transformed_name.split("__", maxsplit=1)[-1]
        for feature in FEATURES:
            if feature_name == feature or feature_name.startswith(f"{feature}_"):
                return feature
        return feature_name

    transformed_importance["Feature"] = transformed_importance[
        "transformed_feature"
    ].map(source_feature)
    feature_importance = (
        transformed_importance.groupby("Feature", as_index=False)["importance"]
        .sum()
        .rename(columns={"importance": "Importance"})
        .sort_values("Importance", ascending=False)
    )

    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "r2": r2_score(y_test, predictions),
        "feature_importance": feature_importance,
        "test_size": len(y_test),
    }
    return {"model": model, "metrics": metrics}


st.title("Settlement Delay Prediction")
st.markdown(
    "Train a **Random Forest Regressor** to estimate settlement time in hours "
    "for a cross-border payment from information available when it is initiated."
)
st.caption(
    "Only transactions with an observed settlement delay are used for training. "
    "Settlement outcomes, SLA fields, payment IDs, and timestamps are excluded "
    "from model inputs to prevent target leakage."
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

data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
data = data.dropna(subset=[TARGET])
data = data[np.isfinite(data[TARGET])]
if len(data) < 2:
    st.error("At least two transactions with valid settlement delays are required.")
    st.stop()

summary_columns = st.columns(3)
summary_columns[0].metric("Labeled transactions", f"{len(data):,}")
summary_columns[1].metric(
    "Median settlement delay",
    f"{data[TARGET].median():.2f} hours",
)
summary_columns[2].metric("Input features", len(FEATURES))

with st.sidebar:
    st.header("Model settings")
    n_estimators = st.slider("Number of trees", 100, 600, 100, step=100)
    max_depth_label = st.selectbox(
        "Maximum tree depth",
        ["8", "16", "24", "No limit"],
        index=0,
    )
    max_depth = int(max_depth_label) if max_depth_label != "No limit" else None
    min_samples_leaf = st.slider("Minimum samples per leaf", 1, 10, 2)
    train_button = st.button(
        "Train Random Forest",
        type="primary",
        use_container_width=True,
    )

model_key = (
    f"{DATA_PATH.stat().st_mtime}:{n_estimators}:{max_depth}:{min_samples_leaf}"
)
if train_button:
    with st.spinner("Training and evaluating on a held-out test set..."):
        st.session_state["settlement_delay_training_result"] = train_model(
            data,
            n_estimators,
            max_depth,
            min_samples_leaf,
        )
        st.session_state["settlement_delay_model_key"] = model_key

training_result = st.session_state.get("settlement_delay_training_result")
model_is_current = (
    training_result is not None
    and st.session_state.get("settlement_delay_model_key") == model_key
)

if model_is_current:
    metrics = training_result["metrics"]
    st.subheader("Held-out evaluation")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Mean absolute error", f"{metrics['mae']:.2f} hours")
    metric_columns[1].metric("Root mean squared error", f"{metrics['rmse']:.2f} hours")
    metric_columns[2].metric("R² score", f"{metrics['r2']:.3f}")
    metric_columns[3].metric("Test transactions", f"{metrics['test_size']:,}")

    st.markdown("**Feature importance**")
    st.bar_chart(
        metrics["feature_importance"].set_index("Feature"),
        x_label="Importance",
        y_label="Input feature",
        horizontal=True,
    )
    st.caption(
        "Evaluation uses a random 80/20 split. Errors are measured in hours; "
        "R² describes variance explained on the held-out transactions."
    )
elif training_result is not None:
    st.info("Model settings changed. Train the model again before making predictions.")
else:
    st.info("Choose model settings in the sidebar, then train the model to continue.")

st.subheader("Estimate settlement delay")
if model_is_current:
    st.write("Enter the information available when the payment is initiated.")
    form_values: dict[str, Any] = {}
    with st.form("settlement_delay_prediction_form"):
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

            with number_columns[index % 3]:
                if feature in INTEGER_FEATURES:
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
            "Predict settlement delay",
            type="primary",
        )

    if predict_button:
        prediction_row = pd.DataFrame([form_values], columns=FEATURES)
        prediction_row = prepare_features(prediction_row)
        delay_hours = float(training_result["model"].predict(prediction_row)[0])
        st.metric("Estimated settlement delay", f"{delay_hours:.2f} hours")
        st.caption(
            "This estimate is based on historical transactions with observed "
            "settlement delays and is not an SLA-breach probability."
        )
else:
    st.caption("Train the model to enable the settlement delay estimation form.")