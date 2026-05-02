import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# PAGE SETTINGS
# -------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/churn_data.csv")

    # Convert TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df.dropna(inplace=True)

    # Drop customerID
    df.drop("customerID", axis=1, inplace=True)

    # Encode categorical columns
    encoder = LabelEncoder()
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = encoder.fit_transform(df[col])

    return df

df = load_data()

# -------------------------
# TRAIN MODEL
# -------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# -------------------------
# UI
# -------------------------
st.title("📊 Customer Churn Prediction (Kaggle Data)")

st.write("Enter customer details:")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure", 1, 72, 12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["No", "Yes"])

with col2:
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", ["Yes", "No"])
    backup = st.selectbox("Online Backup", ["Yes", "No"])
    protection = st.selectbox("Device Protection", ["Yes", "No"])
    support = st.selectbox("Tech Support", ["Yes", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])

monthly = st.slider("Monthly Charges", 20, 150, 70)
total = monthly * tenure

# -------------------------
# ENCODING INPUT
# -------------------------
input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone],
    "MultipleLines": [multiple],
    "InternetService": [internet],
    "OnlineSecurity": [security],
    "OnlineBackup": [backup],
    "DeviceProtection": [protection],
    "TechSupport": [support],
    "StreamingTV": ["No"],
    "StreamingMovies": ["No"],
    "Contract": [contract],
    "PaperlessBilling": [billing],
    "PaymentMethod": [payment],
    "MonthlyCharges": [monthly],
    "TotalCharges": [total]
})

# Encode input same way
encoder = LabelEncoder()
for col in input_data.columns:
    input_data[col] = encoder.fit_transform(input_data[col])

# Align columns
input_data = input_data[X.columns]

# -------------------------
# PREDICTION
# -------------------------
if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ Customer likely to churn")
    else:
        st.success("✅ Customer likely to stay")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.caption(f"Model Accuracy: {accuracy:.2%}")