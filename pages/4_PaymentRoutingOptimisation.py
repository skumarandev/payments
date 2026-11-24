from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Payment Routing Optimisation",
    page_icon="🧭",
    layout="wide",
)

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    st.error(
        "CatBoost is not installed. Install the project dependencies with "
        "`pip install -r requirements.txt`, then restart Streamlit."
    )
    st.stop()

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "cross_border_payments_100k.csv"
)
SUCCESS_TARGET = "payment_success"
FEE_TARGET = "transaction_fee_usd_equiv"
DELAY_TARGET = "settlement_delay_hours"
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
    "transaction_hour",
    "day_of_week",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
ROUTE_FEATURES = ["payment_rail", "sending_bank"]
OBJECTIVE_WEIGHTS = {
    "Balanced": {"success": 0.50, "cost": 0.25, "speed": 0.25},
    "Prioritise reliability": {"success": 0.70, "cost": 0.15, "speed": 0.15},
    "Minimise cost": {"success": 0.25, "cost": 0.60, "speed": 0.15},
    "Prioritise speed": {"success": 0.25, "cost": 0.15, "speed": 0.60},
}


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
def train_models(
    data: pd.DataFrame,
    iterations: int,
    depth: int,
) -> dict[str, Any]:
    features = prepare_features(data)

    success_target = data[SUCCESS_TARGET].astype(int)
    success_x_train, success_x_test, success_y_train, success_y_test = (
        train_test_split(
            features,
            success_target,
            test_size=0.2,
            random_state=42,
            stratify=success_target,
        )
    )
    success_model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    success_model.fit(
        success_x_train,
        success_y_train,
        cat_features=CATEGORICAL_FEATURES,
    )
    success_probabilities = success_model.predict_proba(success_x_test)[:, 1]

    fee_target = pd.to_numeric(data[FEE_TARGET], errors="coerce")
    fee_rows = fee_target.notna()
    fee_x_train, fee_x_test, fee_y_train, fee_y_test = train_test_split(
        features.loc[fee_rows],
        fee_target.loc[fee_rows],
        test_size=0.2,
        random_state=42,
    )
    fee_model = CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    fee_model.fit(
        fee_x_train,
        fee_y_train,
        cat_features=CATEGORICAL_FEATURES,
    )
    fee_predictions = fee_model.predict(fee_x_test)

    delay_target = pd.to_numeric(data[DELAY_TARGET], errors="coerce")
    delay_rows = delay_target.notna()
    delay_x_train, delay_x_test, delay_y_train, delay_y_test = train_test_split(
        features.loc[delay_rows],
        delay_target.loc[delay_rows],
        test_size=0.2,
        random_state=42,
    )
    delay_model = CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    delay_model.fit(
        delay_x_train,
        delay_y_train,
        cat_features=CATEGORICAL_FEATURES,
    )
    delay_predictions = delay_model.predict(delay_x_test)

    metrics = {
        "success_roc_auc": roc_auc_score(
            success_y_test,
            success_probabilities,
        ),
        "fee_mae": mean_absolute_error(fee_y_test, fee_predictions),
        "delay_mae": mean_absolute_error(delay_y_test, delay_predictions),
        "success_test_size": len(success_y_test),
        "fee_test_size": len(fee_y_test),
        "delay_test_size": len(delay_y_test),
    }
    return {
        "success_model": success_model,
        "fee_model": fee_model,
        "delay_model": delay_model,
        "metrics": metrics,
    }


def score_routes(
    candidates: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    scored = candidates.copy()
    success_range = scored["Predicted success probability"].max() - scored[
        "Predicted success probability"
    ].min()
    fee_range = scored["Predicted fee (USD equiv.)"].max() - scored[
        "Predicted fee (USD equiv.)"
    ].min()
    delay_range = scored["Predicted delay (hours)"].max() - scored[
        "Predicted delay (hours)"
    ].min()

    if success_range == 0:
        scored["success_score"] = 0.5
    else:
        scored["success_score"] = (
            scored["Predicted success probability"]
            - scored["Predicted success probability"].min()
        ) / success_range

    if fee_range == 0:
        scored["cost_score"] = 0.5
    else:
        scored["cost_score"] = 1 - (
            scored["Predicted fee (USD equiv.)"]
            - scored["Predicted fee (USD equiv.)"].min()
        ) / fee_range

    if delay_range == 0:
        scored["speed_score"] = 0.5
    else:
        scored["speed_score"] = 1 - (
            scored["Predicted delay (hours)"]
            - scored["Predicted delay (hours)"].min()
        ) / delay_range

    scored["Recommendation score"] = (
        weights["success"] * scored["success_score"]
        + weights["cost"] * scored["cost_score"]
        + weights["speed"] * scored["speed_score"]
    )
    return scored.sort_values("Recommendation score", ascending=False)


st.title("Payment Routing Optimisation")
st.markdown(
    "Compare payment rails and sending banks for a proposed transfer. "
    "The recommendation balances predicted success likelihood, transaction "
    "fees, and settlement speed."
)
st.caption(
    "The destination bank is fixed for the transfer. Payment rail and sending "
    "bank are evaluated as route choices; fee and settlement outcomes are "
    "targets, not model inputs."
)

if not DATA_PATH.is_file():
    st.error(f"Could not find the payment dataset at `{DATA_PATH}`.")
    st.stop()

try:
    data = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except (OSError, pd.errors.ParserError) as error:
    st.error(f"Could not load the payment dataset: {error}")
    st.stop()

required_columns = set(
    FEATURES + [SUCCESS_TARGET, FEE_TARGET, DELAY_TARGET]
)
missing_columns = sorted(required_columns - set(data.columns))
if missing_columns:
    st.error(f"The dataset is missing required columns: {', '.join(missing_columns)}")
    st.stop()

data[SUCCESS_TARGET] = pd.to_numeric(data[SUCCESS_TARGET], errors="coerce")
data[FEE_TARGET] = pd.to_numeric(data[FEE_TARGET], errors="coerce")
data[DELAY_TARGET] = pd.to_numeric(data[DELAY_TARGET], errors="coerce")
data = data.dropna(subset=[SUCCESS_TARGET, FEE_TARGET]).copy()
if data.empty or not set(data[SUCCESS_TARGET].unique()).issubset({0, 1}):
    st.error("The payment_success target must contain valid 0 and 1 values.")
    st.stop()
if data[SUCCESS_TARGET].nunique() != 2:
    st.error("The payment_success target must contain both 0 and 1 classes.")
    st.stop()
data[SUCCESS_TARGET] = data[SUCCESS_TARGET].astype(int)

with st.sidebar:
    st.header("Model settings")
    iterations = st.slider("Boosting iterations", 100, 600, 200, step=100)
    depth = st.slider("Tree depth", 4, 10, 6)
    train_button = st.button(
        "Train routing models",
        type="primary",
        use_container_width=True,
    )

model_key = f"{DATA_PATH.stat().st_mtime}:{iterations}:{depth}"
if train_button:
    with st.spinner("Training route success, fee, and delay models..."):
        st.session_state["routing_training_result"] = train_models(
            data,
            iterations,
            depth,
        )
        st.session_state["routing_model_key"] = model_key

training_result = st.session_state.get("routing_training_result")
model_is_current = (
    training_result is not None
    and st.session_state.get("routing_model_key") == model_key
)

if model_is_current:
    metrics = training_result["metrics"]
    st.subheader("Model validation")
    metric_columns = st.columns(3)
    metric_columns[0].metric(
        "Success ROC AUC",
        f"{metrics['success_roc_auc']:.3f}",
    )
    metric_columns[1].metric(
        "Fee MAE",
        f"${metrics['fee_mae']:.2f}",
    )
    metric_columns[2].metric(
        "Settlement delay MAE",
        f"{metrics['delay_mae']:.2f} hours",
    )
    st.caption(
        "Validation metrics use separate 80/20 splits. Settlement-delay "
        "validation includes only successful payments with an observed delay."
    )
elif training_result is not None:
    st.info("Model settings changed. Train the models again before optimising.")
else:
    st.info("Train the routing models in the sidebar to compare route options.")

st.subheader("Optimise a payment route")
if model_is_current:
    destinations = sorted(data["destination_country"].dropna().unique().tolist())
    destination_country = st.selectbox("Destination country", destinations)
    destination_data = data[data["destination_country"] == destination_country]
    beneficiary_banks = sorted(
        destination_data["beneficiary_bank"].dropna().unique().tolist()
    )
    destination_currency = (
        destination_data["destination_currency"].mode().iloc[0]
    )
    destination_fx_rates = pd.to_numeric(
        destination_data["fx_rate_inr_per_unit"],
        errors="coerce",
    ).dropna()

    with st.form("routing_optimisation_form"):
        beneficiary_bank = st.selectbox("Beneficiary bank", beneficiary_banks)
        input_columns = st.columns(3)
        amount_values = pd.to_numeric(data["amount_source"], errors="coerce").dropna()
        amount_min = float(amount_values.min())
        amount_max = float(amount_values.max())
        amount_default = float(amount_values.median())
        amount_range = amount_max - amount_min

        with input_columns[0]:
            amount_source = st.number_input(
                f"Amount (INR)",
                min_value=amount_min,
                max_value=amount_max,
                value=amount_default,
                step=max(amount_range / 100, 1.0),
                format="%.2f",
            )
        with input_columns[1]:
            fx_rate = st.number_input(
                f"FX rate (INR per {destination_currency})",
                min_value=float(destination_fx_rates.min()),
                max_value=float(destination_fx_rates.max()),
                value=float(destination_fx_rates.median()),
                step=max(
                    float(destination_fx_rates.max() - destination_fx_rates.min())
                    / 100,
                    0.000001,
                ),
                format="%.6f",
            )
        with input_columns[2]:
            optimization_objective = st.selectbox(
                "Optimisation objective",
                list(OBJECTIVE_WEIGHTS),
                index=0,
            )

        category_columns = st.columns(3)
        form_values: dict[str, Any] = {
            "source_country": data["source_country"].mode().iloc[0],
            "destination_country": destination_country,
            "source_currency": data["source_currency"].mode().iloc[0],
            "destination_currency": destination_currency,
            "beneficiary_bank": beneficiary_bank,
            "amount_source": amount_source,
            "fx_rate_inr_per_unit": fx_rate,
        }
        selectable_categories = [
            "compliance_status",
            "customer_segment",
            "channel",
        ]
        for index, feature in enumerate(selectable_categories):
            options = sorted(data[feature].dropna().astype(str).unique().tolist())
            with category_columns[index]:
                form_values[feature] = st.selectbox(
                    feature.replace("_", " ").title(),
                    options,
                )

        clock_columns = st.columns(2)
        for column, feature in zip(clock_columns, ["transaction_hour", "day_of_week"]):
            values = pd.to_numeric(data[feature], errors="coerce").dropna()
            with column:
                form_values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=int(values.min()),
                    max_value=int(values.max()),
                    value=int(round(values.median())),
                    step=1,
                )

        minimum_support = st.selectbox(
            "Minimum historical observations per route",
            [10, 25, 50, 100],
            index=0,
        )
        minimum_success_probability = st.slider(
            "Minimum predicted success probability",
            min_value=0.70,
            max_value=0.99,
            value=0.80,
            step=0.01,
            format="%.2f",
        )
        optimise_button = st.form_submit_button(
            "Compare routes",
            type="primary",
        )

    if optimise_button:
        base_features = pd.DataFrame([form_values])
        route_options = pd.MultiIndex.from_product(
            [
                sorted(data["payment_rail"].dropna().unique().tolist()),
                sorted(data["sending_bank"].dropna().unique().tolist()),
            ],
            names=ROUTE_FEATURES,
        ).to_frame(index=False)
        route_options["destination_country"] = destination_country
        route_options["beneficiary_bank"] = beneficiary_bank

        route_history = (
            data[
                (data["destination_country"] == destination_country)
                & (data["beneficiary_bank"] == beneficiary_bank)
            ]
            .groupby(ROUTE_FEATURES)
            .size()
            .rename("Historical observations")
            .reset_index()
        )
        route_options = route_options.merge(
            route_history,
            on=ROUTE_FEATURES,
            how="left",
        )
        route_options["Historical observations"] = (
            route_options["Historical observations"].fillna(0).astype(int)
        )
        route_options = route_options[
            route_options["Historical observations"] >= minimum_support
        ].copy()

        if route_options.empty:
            st.error(
                "No candidate routes meet the selected historical support "
                "threshold for this destination bank."
            )
        else:
            candidate_features = pd.concat(
                [
                    pd.concat(
                        [base_features] * len(route_options),
                        ignore_index=True,
                    ),
                    route_options[ROUTE_FEATURES].reset_index(drop=True),
                ],
                axis=1,
            )
            candidate_features = prepare_features(candidate_features)
            models = training_result
            route_options["Predicted success probability"] = models[
                "success_model"
            ].predict_proba(candidate_features)[:, 1]
            route_options["Predicted fee (USD equiv.)"] = models[
                "fee_model"
            ].predict(candidate_features)
            route_options["Predicted delay (hours)"] = models[
                "delay_model"
            ].predict(candidate_features)

            ranked_routes = score_routes(
                route_options,
                OBJECTIVE_WEIGHTS[optimization_objective],
            )
            eligible_routes = ranked_routes[
                ranked_routes["Predicted success probability"]
                >= minimum_success_probability
            ]

            if eligible_routes.empty:
                st.warning(
                    "No route meets the selected success threshold. "
                    "Showing the highest-scoring alternatives without a "
                    "recommendation."
                )
            else:
                best_route = eligible_routes.iloc[0]
                result_columns = st.columns(4)
                result_columns[0].metric(
                    "Recommended rail",
                    best_route["payment_rail"],
                )
                result_columns[1].metric(
                    "Sending bank",
                    best_route["sending_bank"],
                )
                result_columns[2].metric(
                    "Predicted success",
                    f"{best_route['Predicted success probability']:.1%}",
                )
                result_columns[3].metric(
                    "Predicted fee",
                    f"${best_route['Predicted fee (USD equiv.)']:.2f}",
                )
                st.metric(
                    "Predicted settlement delay if successful",
                    f"{best_route['Predicted delay (hours)']:.2f} hours",
                )

            st.markdown("**Ranked route alternatives**")
            st.dataframe(
                ranked_routes[
                    [
                        "payment_rail",
                        "sending_bank",
                        "Predicted success probability",
                        "Predicted fee (USD equiv.)",
                        "Predicted delay (hours)",
                        "Historical observations",
                        "Recommendation score",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Recommendation scores compare success, fee, and delay across "
                "candidate routes. The success threshold uses absolute predicted "
                "probability. Delay estimates are conditional on successful "
                "settlement."
            )
else:
    st.caption("Train the models to enable route comparison.")