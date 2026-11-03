import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models/churn_model.joblib"
METRICS_PATH = ROOT / "models/metrics.json"

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

with open(METRICS_PATH) as f:
    metrics = json.load(f)

st.title("📊 Customer Churn Prediction")

st.write(
    "Predict customer churn using machine learning "
    "and identify customers who may be at risk."
)

st.divider()

# Model performance
st.subheader("Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
col2.metric("Precision", f"{metrics['precision']:.1%}")
col3.metric("Recall", f"{metrics['recall']:.1%}")
col4.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

st.caption(f"Selected algorithm: {metrics['model']}")

st.divider()

tab1, tab2 = st.tabs([
    "Single Customer Prediction",
    "Batch CSV Prediction",
])

with tab1:

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )

        senior = st.selectbox(
            "Senior Citizen", ["No", "Yes"]
        )

        partner = st.selectbox(
            "Partner", ["No", "Yes"]
        )

        dependents = st.selectbox(
            "Dependents", ["No", "Yes"]
        )

        tenure = st.slider(
            "Tenure (Months)", 0, 72, 12
        )

        phone = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"],
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
        )

        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"],
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"],
        )

    with col2:

        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"],
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"],
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"],
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"],
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )

        paperless = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

        monthly = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=70.0,
        )

        total = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=840.0,
        )

    if st.button(
        "Predict Customer Churn",
        type="primary",
        use_container_width=True,
    ):

        # Keep related service selections consistent
        if phone == "No":
            multiple_lines = "No phone service"
        elif multiple_lines == "No phone service":
            multiple_lines = "No"

        internet_services = [
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
        ]

        if internet == "No":
            internet_services = ["No internet service"] * 6
        else:
            internet_services = [
                "No" if value == "No internet service" else value
                for value in internet_services
            ]

        (
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
        ) = internet_services

        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }])

        probability = model.predict_proba(customer)[0, 1]

        st.subheader("Prediction Result")

        st.metric(
            "Churn Probability",
            f"{probability:.1%}",
        )

        st.progress(float(probability))

        if probability >= 0.5:
            st.error("⚠️ Customer predicted to churn")
        else:
            st.success("✅ Customer predicted to stay")

        st.caption(
            "Predictions are estimates, not guarantees. "
            "The classification threshold is 50%."
        )

with tab2:

    st.subheader("Upload Customer CSV")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
    )

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)

        required_columns = list(
            model.feature_names_in_
        )

        missing = [
            col for col in required_columns
            if col not in data.columns
        ]

        if missing:
            st.error(
                "Missing required columns: "
                + ", ".join(missing)
            )

        else:
            prediction_data = data[required_columns].copy()

            for col in [
                "SeniorCitizen",
                "tenure",
                "MonthlyCharges",
                "TotalCharges",
            ]:
                prediction_data[col] = pd.to_numeric(
                    prediction_data[col],
                    errors="coerce",
                )

            predictions = model.predict(prediction_data)

            probabilities = model.predict_proba(
                prediction_data
            )[:, 1]

            results = data.copy()

            results["Predicted_Churn"] = [
                "Yes" if value == 1 else "No"
                for value in predictions
            ]

            results["Churn_Probability"] = probabilities

            st.dataframe(results)

            csv = results.to_csv(index=False)

            st.download_button(
                "Download Predictions",
                csv,
                "churn_predictions.csv",
                "text/csv",
            )