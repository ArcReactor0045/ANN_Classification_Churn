

import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

.stButton>button {
    width: 100%;
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}

.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

.prediction-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

.success-box {
    background-color: #dcfce7;
    color: #166534;
}

.danger-box {
    background-color: #fee2e2;
    color: #991b1b;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Load Model
# ---------------------------
model = tf.keras.models.load_model("model.h5")

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ---------------------------
# Header
# ---------------------------
st.title("🏦 Customer Churn Prediction Dashboard")
st.markdown(
    "Predict whether a customer is likely to leave the bank using a trained Deep Learning model."
)

st.divider()

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("Customer Information")

geography = st.sidebar.selectbox(
    "🌍 Geography",
    onehot_encoder_geo.categories_[0]
)

gender = st.sidebar.selectbox(
    "👤 Gender",
    label_encoder_gender.classes_
)

credit_score = st.sidebar.number_input(
    "💳 Credit Score",
    min_value=0,
    max_value=900,
    value=650
)

age = st.sidebar.slider(
    "🎂 Age",
    18,
    92,
    35
)

tenure = st.sidebar.slider(
    "📅 Tenure (Years)",
    0,
    10,
    5
)

balance = st.sidebar.number_input(
    "💰 Balance",
    value=50000.0
)

num_of_products = st.sidebar.slider(
    "📦 Number of Products",
    1,
    4,
    2
)

has_cr_card = st.sidebar.selectbox(
    "💳 Has Credit Card",
    [0, 1]
)

is_active_member = st.sidebar.selectbox(
    "✅ Active Member",
    [0, 1]
)

estimated_salary = st.sidebar.number_input(
    "💵 Estimated Salary",
    value=100000.0
)

# ---------------------------
# Summary Cards
# ---------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Age", age)

with col2:
    st.metric("Credit Score", credit_score)

with col3:
    st.metric("Balance", f"${balance:,.0f}")

with col4:
    st.metric("Salary", f"${estimated_salary:,.0f}")

st.divider()

# ---------------------------
# Prediction Button
# ---------------------------
if st.button("🔮 Predict Churn Probability"):

    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    encoded_geo = onehot_encoder_geo.transform([[geography]])

    geo_df = pd.DataFrame(
        encoded_geo.toarray(),
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    input_data = pd.concat([input_data, geo_df], axis=1)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    pred_prob = float(prediction[0][0])

    st.subheader("Prediction Results")

    col1, col2 = st.columns([2, 1])

    with col1:

        st.progress(pred_prob)

        st.metric(
            "Churn Probability",
            f"{pred_prob*100:.2f}%"
        )

    with col2:

        if pred_prob >= 0.5:
            st.markdown(
                f"""
                <div class="prediction-box danger-box">
                    ⚠️ HIGH RISK<br>
                    Customer Likely to Churn
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="prediction-box success-box">
                    ✅ LOW RISK<br>
                    Customer Likely to Stay
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    st.subheader("Model Confidence")

    st.info(
        f"""
        Probability of Churn: **{pred_prob*100:.2f}%**

        Probability of Retention: **{(1-pred_prob)*100:.2f}%**
        """
    )