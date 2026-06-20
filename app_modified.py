import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pickle

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Design tokens / global styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --bg: #F6F7FA;
    --surface: #FFFFFF;
    --ink: #15202B;
    --muted: #6B7280;
    --line: #E4E7EC;
    --brand: #2F3C7E;
    --brand-dark: #232C61;
    --risk: #C7493A;
    --risk-light: #FBEAE7;
    --safe: #1E7A55;
    --safe-light: #E7F4ED;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg); }
#MainMenu, footer { visibility: hidden; }

div[data-testid="stAppViewContainer"] .main .block-container {
    padding-top: 3rem;
    max-width: 880px;
}

h1, h2, h3 { font-family: 'Fraunces', serif; color: var(--ink); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

.brand { display:flex; align-items:center; gap:10px; margin-bottom: 22px; }
.brand-mark { font-size:18px; color: var(--brand); }
.brand-name { font-family:'IBM Plex Mono', monospace; font-size:12px; font-weight:600;
              letter-spacing:0.1em; text-transform:uppercase; color: var(--ink); }

.form-section { font-size:11px; text-transform:uppercase; letter-spacing:0.08em;
                color: var(--muted); font-weight:600; margin: 18px 0 2px 0; }

/* Inputs */
div[data-testid="stNumberInput"] input,
div[data-baseweb="select"] > div {
    border-radius: 8px !important;
    border-color: var(--line) !important;
}
div[data-baseweb="slider"] [role="slider"] {
    background-color: var(--brand) !important;
    border-color: var(--brand) !important;
}
div[data-testid="stForm"] { border: none; padding: 0; }

/* Submit button */
div[data-testid="stFormSubmitButton"] button {
    width: 100%;
    background: var(--brand) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 0 !important;
    margin-top: 16px;
    transition: background 0.15s ease;
}
div[data-testid="stFormSubmitButton"] button:hover { background: var(--brand-dark) !important; }

/* Hero */
.page-wrap { max-width: 760px; }
.eyebrow { font-family:'IBM Plex Mono', monospace; font-size:12px; letter-spacing:0.12em;
           text-transform:uppercase; color: var(--brand); margin:0 0 10px 0; }
.hero-title { font-family:'Fraunces', serif; font-size:38px; font-weight:600; color:var(--ink);
              margin:0 0 8px 0; line-height:1.15; }
.hero-subtitle { color:var(--muted); font-size:15px; max-width:520px; margin: 0 0 32px 0; }

/* Result card */
.result-card { background: var(--surface); border:1px solid var(--line); border-radius:20px;
               padding:36px 40px; box-shadow: 0 1px 2px rgba(15,32,67,0.04), 0 12px 24px -12px rgba(15,32,67,0.08); }

.badge-risk, .badge-safe { display:inline-block; padding:6px 14px; border-radius:999px;
    font-family:'IBM Plex Mono', monospace; font-size:12px; font-weight:600; letter-spacing:0.08em; }
.badge-risk { background: var(--risk-light); color: var(--risk); }
.badge-safe { background: var(--safe-light); color: var(--safe); }

.prob-display { font-family:'Fraunces', serif; font-size:60px; font-weight:600; color:var(--ink);
                 line-height:1; margin: 16px 0 0 0; }
.prob-unit { font-size:26px; color:var(--muted); margin-left:4px; }
.prob-caption { color:var(--muted); font-size:13px; margin: 4px 0 0 0; text-transform:uppercase; letter-spacing:0.06em; }

.meter-track { position:relative; height:10px; border-radius:999px;
    background: linear-gradient(90deg, var(--safe) 0%, #E8B93F 50%, var(--risk) 100%); margin: 22px 0 8px 0; }
.meter-marker { position:absolute; top:50%; width:18px; height:18px; background:var(--ink);
    border:3px solid #fff; border-radius:50%; transform:translate(-50%,-50%); box-shadow:0 1px 4px rgba(0,0,0,0.35); }
.meter-labels { display:flex; justify-content:space-between; font-size:11px; color:var(--muted);
    text-transform:uppercase; letter-spacing:0.05em; }

.divider { height:1px; background: var(--line); margin: 28px 0 18px 0; }
.snapshot-title { font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted);
    font-weight:600; margin:0 0 12px 0; }
.snapshot-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(130px,1fr)); gap:14px; }
.chip { display:flex; flex-direction:column; gap:2px; padding:10px 12px; background: var(--bg);
    border:1px solid var(--line); border-radius:10px; }
.chip-label { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:0.05em; }
.chip-value { font-family:'IBM Plex Mono', monospace; font-size:14px; color:var(--ink); font-weight:600; }

/* Empty state */
.empty-state { text-align:center; padding:70px 20px; border:1px dashed var(--line); border-radius:16px;
    background: var(--surface); }
.empty-icon { font-size:26px; color: var(--brand); margin-bottom:10px; }
.empty-title { font-family:'Fraunces', serif; font-size:20px; color:var(--ink); margin:0 0 6px 0; }
.empty-text { color:var(--muted); font-size:14px; max-width:380px; margin:0 auto; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load model and preprocessing assets (cached so they only load once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('model.h5')
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, onehot_encoder_geo, label_encoder_gender, scaler

model, onehot_encoder_geo, label_encoder_gender, scaler = load_assets()

# ---------------------------------------------------------------------------
# Sidebar — customer profile form
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="brand-mark">◆</span>'
        '<span class="brand-name">Churn Intelligence</span></div>',
        unsafe_allow_html=True
    )

    with st.form("churn_profile_form"):
        st.markdown('<p class="form-section">Personal Details</p>', unsafe_allow_html=True)
        geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('Gender', label_encoder_gender.classes_)
        age = st.slider('Age', 18, 92)

        st.markdown('<p class="form-section">Account Details</p>', unsafe_allow_html=True)
        credit_score = st.number_input('Credit Score')
        balance = st.number_input('Balance')
        tenure = st.slider('Tenure', 0, 10)
        num_of_products = st.slider('Number of Products', 1, 4)

        st.markdown('<p class="form-section">Engagement</p>', unsafe_allow_html=True)
        has_cr_card = st.selectbox('Has Credit Card', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active_member = st.selectbox('Is Active Member', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        estimated_salary = st.number_input('Estimated Salary')

        submitted = st.form_submit_button("Run Prediction")

# ---------------------------------------------------------------------------
# Prediction (unchanged logic — only runs once the form is submitted)
# ---------------------------------------------------------------------------
if submitted:
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
    col_geo = pd.DataFrame(encoded_geo.toarray(), columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data, col_geo], axis=1)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    pred_prob = prediction[0][0]

    st.session_state['pred_prob'] = float(pred_prob)
    st.session_state['profile'] = {
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': f"{tenure} yrs",
        'Credit Score': f"{credit_score:,.0f}",
        'Balance': f"{balance:,.2f}",
        'Products': num_of_products,
        'Has Credit Card': 'Yes' if has_cr_card == 1 else 'No',
        'Active Member': 'Yes' if is_active_member == 1 else 'No',
        'Estimated Salary': f"{estimated_salary:,.2f}",
    }

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
st.markdown('<p class="eyebrow">Retention Analytics</p>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Customer Churn / Exit Prediction</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Fill in the customer profile in the sidebar and run a prediction '
    'to see the estimated likelihood that this customer will churn.</p>',
    unsafe_allow_html=True
)

if 'pred_prob' in st.session_state:
    pred_prob = st.session_state['pred_prob']
    profile = st.session_state['profile']
    pct = max(0.0, min(100.0, pred_prob * 100))

    if pred_prob >= 0.5:
        badge_class, badge_text = "badge-risk", "AT RISK OF CHURN"
    else:
        badge_class, badge_text = "badge-safe", "LIKELY TO STAY"

    chips = "".join(
        f'<div class="chip"><span class="chip-label">{label}</span>'
        f'<span class="chip-value">{value}</span></div>'
        for label, value in profile.items()
    )

    st.markdown(f"""
    <div class="result-card">
        <span class="{badge_class}">{badge_text}</span>
        <div class="prob-display">{pct:.1f}<span class="prob-unit">%</span></div>
        <p class="prob-caption">Estimated probability of churn</p>
        <div class="meter-track"><div class="meter-marker" style="left:{pct}%;"></div></div>
        <div class="meter-labels"><span>Low risk</span><span>High risk</span></div>
        <div class="divider"></div>
        <p class="snapshot-title">Profile Snapshot</p>
        <div class="snapshot-grid">{chips}</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">◆</div>
        <p class="empty-title">No prediction yet</p>
        <p class="empty-text">Complete the customer profile in the sidebar and select
        <strong>Run Prediction</strong> to see the retention outlook here.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

                                # My Code


# import streamlit as st
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
# import pickle

# model=tf.keras.models.load_model('model.h5')
# with open('onehot_encoder_geo.pkl','rb') as file:
#     onehot_encoder_geo=pickle.load(file)
# with open('label_encoder_gender.pkl','rb') as file:
#     label_encoder_gender=pickle.load(file)
# with open('scaler.pkl','rb') as file:
#     scaler=pickle.load(file)


#     ## Streamlit app


# st.title("Customer Churn / Exit Prediction")

# # User input

# geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
# gender = st.selectbox('Gender', label_encoder_gender.classes_)

# age = st.slider('Age', 18, 92)
# balance = st.number_input('Balance')

# credit_score = st.number_input('Credit Score')
# estimated_salary = st.number_input('Estimated Salary')

# tenure = st.slider('Tenure', 0, 10)
# num_of_products = st.slider('Number of Products', 1, 4)

# has_cr_card = st.selectbox('Has Credit Card', [0, 1])
# is_active_member = st.selectbox('Is Active Member', [0, 1])

# # Prepare the input data
# input_data = pd.DataFrame({
#     'CreditScore': [credit_score],
#     'Gender': [label_encoder_gender.transform([gender])[0]],
#     'Age': [age],
#     'Tenure': [tenure],
#     'Balance': [balance],
#     'NumOfProducts': [num_of_products],
#     'HasCrCard': [has_cr_card],
#     'IsActiveMember': [is_active_member],
#     'EstimatedSalary': [estimated_salary]
# })

# encoded_geo=onehot_encoder_geo.transform([[geography]])
# col_geo=pd.DataFrame(encoded_geo.toarray(),columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
# input_data=pd.concat([input_data,col_geo],axis=1)   

# input_scaled=scaler.transform(input_data)

# prediction=model.predict(input_scaled)
# pred_prob=prediction[0][0]

# st.write(f"The probability output was {pred_prob}")
# if pred_prob>=0.5:
#     st.write('The customer will most likely Churn / exit')

# else:
#     st.write('The customer will not Churn / exit')