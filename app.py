import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model, scaler and features
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
top_features = joblib.load('top_features.pkl')

# Page config
st.set_page_config(
    page_title="Bank Deposit Predictor",
    layout="centered"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');

    * { font-family: 'DM Sans', sans-serif; }

    /* Soft white background */
    .stApp {
        background: #f9fafb;
    }

    /* Header */
    .header {
        background: linear-gradient(135deg, #064e3b, #059669, #34d399);
        background-size: 200% 200%;
        animation: gradientShift 6s ease infinite;
        border-radius: 28px;
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 12px 40px rgba(5, 150, 105, 0.25);
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .header-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        animation: fadeInDown 0.8s ease;
    }

    .header-subtitle {
        font-size: 1rem;
        color: rgba(255,255,255,0.85);
        font-weight: 300;
        animation: fadeInUp 0.8s ease;
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-15px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Single input card */
    .card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 16px rgba(0,0,0,0.05);
        margin-bottom: 1.4rem;
        animation: fadeIn 0.6s ease;
    }

    .card-header {
        font-size: 0.8rem;
        font-weight: 700;
        color: #059669;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1.4rem;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid #d1fae5;
    }

    /* Input labels */
    label {
        color: #374151 !important;
        font-size: 0.87rem !important;
        font-weight: 500 !important;
    }

    /* Number inputs */
    .stNumberInput input {
        background-color: #f9fafb !important;
        border: 1.5px solid #e5e7eb !important;
        border-radius: 10px !important;
        color: #111827 !important;
        font-size: 0.95rem !important;
    }

    .stNumberInput input:focus {
        border-color: #059669 !important;
        box-shadow: 0 0 0 3px rgba(5,150,105,0.12) !important;
        background-color: white !important;
    }

    .stNumberInput button {
        background-color: #f3f4f6 !important;
        border: 1px solid #e5e7eb !important;
        color: #374151 !important;
        border-radius: 8px !important;
    }

    /* Dropdowns */
    div[data-baseweb="select"] > div {
        background-color: #f9fafb !important;
        border: 1.5px solid #e5e7eb !important;
        border-radius: 10px !important;
        color: #111827 !important;
    }

    ul[role="listbox"] {
        background-color: white !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.1) !important;
    }

    li[role="option"] {
        background-color: white !important;
        color: #111827 !important;
        font-size: 0.9rem !important;
    }

    li[role="option"]:hover {
        background-color: #ecfdf5 !important;
        color: #059669 !important;
    }

    /* Predict button - Pill style */
    .stButton > button {
        background: white !important;
        color: #111827 !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        border: 2px solid #111827 !important;
        border-radius: 999px !important;
        padding: 0.85rem 2rem !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.04em !important;
        box-shadow: none !important;
    }

    .stButton > button:hover {
        background: #111827 !important;
        color: white !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* Result cards */
    .result-yes {
        background: linear-gradient(135deg, #ecfdf5, #d1fae5);
        border: 2px solid #34d399;
        border-radius: 22px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-top: 1.5rem;
        animation: bounceIn 0.7s ease;
        box-shadow: 0 8px 32px rgba(52, 211, 153, 0.2);
    }

    .result-no {
        background: linear-gradient(135deg, #fff7ed, #ffedd5);
        border: 2px solid #fb923c;
        border-radius: 22px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-top: 1.5rem;
        animation: bounceIn 0.7s ease;
        box-shadow: 0 8px 32px rgba(251, 146, 60, 0.2);
    }

    @keyframes bounceIn {
        0% { opacity: 0; transform: scale(0.88); }
        60% { transform: scale(1.03); }
        100% { opacity: 1; transform: scale(1); }
    }

    .result-icon {
        font-size: 3.5rem;
        margin-bottom: 0.8rem;
        animation: pulse 2.5s infinite;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }

    .result-title-yes {
        font-size: 1.7rem;
        font-weight: 700;
        color: #065f46;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .result-title-no {
        font-size: 1.7rem;
        font-weight: 700;
        color: #9a3412;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .result-subtitle {
        font-size: 0.98rem;
        color: #6b7280;
        line-height: 1.7;
        font-weight: 400;
    }

    /* Stats */
    .stats-row {
        display: flex;
        gap: 1rem;
        margin-top: 1.5rem;
    }

    .stat-box {
        flex: 1;
        background: white;
        border-radius: 16px;
        padding: 1.2rem 0.8rem;
        text-align: center;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        animation: fadeIn 1s ease;
    }

    .stat-box:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    }

    .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .stat-label {
        font-size: 0.7rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-top: 0.3rem;
        font-weight: 500;
    }

    /* Probability bars */
    .prob-container {
        background: white;
        border-radius: 20px;
        padding: 1.8rem;
        margin-top: 1.2rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 16px rgba(0,0,0,0.05);
        animation: fadeIn 0.9s ease;
    }

    .prob-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #059669;
        margin-bottom: 1.4rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding-bottom: 0.7rem;
        border-bottom: 2px solid #d1fae5;
    }

    .prob-row { margin-bottom: 1.2rem; }

    .prob-label-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.88rem;
        color: #374151;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }

    .prob-bar-bg {
        background: #f3f4f6;
        border-radius: 999px;
        height: 12px;
        overflow: hidden;
    }

    .prob-bar-no {
        height: 12px;
        border-radius: 999px;
        background: linear-gradient(90deg, #fb923c, #fdba74);
        animation: growBar 1.2s ease forwards;
    }

    .prob-bar-yes {
        height: 12px;
        border-radius: 999px;
        background: linear-gradient(90deg, #059669, #34d399);
        animation: growBar 1.2s ease forwards;
    }

    @keyframes growBar {
        from { width: 0% !important; }
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
    <div class="header">
        <div class="header-title">🏦 Bank Deposit Predictor</div>
""", unsafe_allow_html=True)

# ── Single Input Card ─────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-header">👤 Client Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age", min_value=18, max_value=95, value=35)
    balance = st.number_input("💰 Account Balance ($)", min_value=-5000, max_value=100000, value=1000)
    duration = st.number_input("📞 Last Call Duration (sec)", min_value=0, max_value=5000, value=200)
    campaign = st.number_input("📣 Contacts This Campaign", min_value=1, max_value=50, value=2)
    day = st.number_input("📅 Last Contact Day", min_value=1, max_value=31, value=15)

with col2:
    housing = st.selectbox("🏠 Has Housing Loan?", ["yes", "no"])
    loan = st.selectbox("💳 Has Personal Loan?", ["yes", "no"])
    default = st.selectbox("⚠️ Has Credit in Default?", ["no", "yes"])
    previous = st.number_input("🔁 Previous Campaign Contacts", min_value=0, max_value=60, value=0)
    poutcome_success = st.selectbox("📋 Previous Campaign Outcome", ["success", "failure", "unknown"])

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict Button ────────────────────────────────────────────────
predict_btn = st.button("🔍 Predict Subscription", use_container_width=True)

if predict_btn:

    # Build input dictionary
    input_dict = {feature: 0 for feature in top_features}

    if 'age' in input_dict: input_dict['age'] = age
    if 'balance' in input_dict: input_dict['balance'] = balance
    if 'duration' in input_dict: input_dict['duration'] = duration
    if 'campaign' in input_dict: input_dict['campaign'] = campaign
    if 'day' in input_dict: input_dict['day'] = day
    if 'housing' in input_dict: input_dict['housing'] = 1 if housing == 'yes' else 0
    if 'loan' in input_dict: input_dict['loan'] = 1 if loan == 'yes' else 0
    if 'default' in input_dict: input_dict['default'] = 1 if default == 'yes' else 0
    if 'previous' in input_dict: input_dict['previous'] = previous
    if 'poutcome_success' in input_dict: input_dict['poutcome_success'] = 1 if poutcome_success == 'success' else 0
    if 'poutcome_unknown' in input_dict: input_dict['poutcome_unknown'] = 1 if poutcome_success == 'unknown' else 0
    if 'contact_unknown' in input_dict: input_dict['contact_unknown'] = 0
    if 'month_may' in input_dict: input_dict['month_may'] = 0
    if 'month_mar' in input_dict: input_dict['month_mar'] = 0
    if 'month_oct' in input_dict: input_dict['month_oct'] = 0
    if 'job_management' in input_dict: input_dict['job_management'] = 0
    if 'education_secondary' in input_dict: input_dict['education_secondary'] = 0

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    prob_no = round(probability[0] * 100, 1)
    prob_yes = round(probability[1] * 100, 1)

    # ── Result ────────────────────────────────────────────────────
    if prediction == 1:
        st.markdown(f"""
            <div class="result-yes">
                <div class="result-icon">🌿</div>
                <div class="result-title-yes">Likely to Subscribe!</div>
                <div class="result-subtitle">
                    This client has a <b>{prob_yes}%</b> probability of subscribing<br>to a term deposit.
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-no">
                <div class="result-icon">🔔</div>
                <div class="result-title-no">Unlikely to Subscribe</div>
                <div class="result-subtitle">
                    This client has only a <b>{prob_yes}%</b> probability of subscribing<br>to a term deposit.
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ── Stats ─────────────────────────────────────────────────────
    st.markdown(f"""
        <div class="stats-row">
            <div class="stat-box">
                <div class="stat-value" style="color:#059669">{prob_yes}%</div>
                <div class="stat-label">Subscribe Chance</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" style="color:#fb923c">{prob_no}%</div>
                <div class="stat-label">Not Subscribe</div>
        </div>
    """, unsafe_allow_html=True)

    # ── Probability Bars ──────────────────────────────────────────
    st.markdown(f"""
        <div class="prob-container">
            <div class="prob-title">📊 Probability Breakdown</div>
            <div class="prob-row">
                <div class="prob-label-row">
                    <span>🔔 Will NOT Subscribe</span>
                    <span>{prob_no}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-no" style="width:{prob_no}%"></div>
                </div>
            </div>
            <div class="prob-row">
                <div class="prob-label-row">
                    <span>🌿 Will Subscribe</span>
                    <span>{prob_yes}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-yes" style="width:{prob_yes}%"></div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    