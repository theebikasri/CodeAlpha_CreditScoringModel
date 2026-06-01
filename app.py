import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Page Config
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}
.big-title {
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:white;
}
.subtitle {
    text-align:center;
    color:#cbd5e1;
    margin-bottom:30px;
}
.metric-card {
    padding:15px;
    border-radius:15px;
    background-color:#1e293b;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    "<div class='big-title'>💳 Credit Risk Prediction System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Machine Learning Based Loan Risk Assessment</div>",
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("📊 Project Information")
st.sidebar.info("""
Algorithm: Random Forest

Dataset: Kaggle Credit Risk Dataset

Purpose:
Predict whether a loan applicant is a low-risk or high-risk borrower.
""")

# Input Layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Person Age", min_value=18)
    income = st.number_input("Annual Income", min_value=0)
    home = st.selectbox(
        "Home Ownership",
        ["RENT", "OWN", "MORTGAGE", "OTHER"]
    )
    emp_length = st.number_input(
        "Employment Length (Years)",
        min_value=0
    )
    loan_intent = st.selectbox(
        "Loan Intent",
        [
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION"
        ]
    )

with col2:
    loan_grade = st.selectbox(
        "Loan Grade",
        ["A", "B", "C", "D", "E", "F", "G"]
    )

    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=0
    )

    loan_int_rate = st.number_input(
        "Interest Rate"
    )

    loan_percent_income = st.number_input(
        "Loan Percent Income"
    )

    default_file = st.selectbox(
        "Default On File",
        ["N", "Y"]
    )

    cred_hist = st.number_input(
        "Credit History Length",
        min_value=0
    )

st.divider()

if st.button("🔍 Predict Risk", use_container_width=True):

    home_map = {
        "MORTGAGE": 0,
        "OTHER": 1,
        "OWN": 2,
        "RENT": 3
    }

    intent_map = {
        "DEBTCONSOLIDATION": 0,
        "EDUCATION": 1,
        "HOMEIMPROVEMENT": 2,
        "MEDICAL": 3,
        "PERSONAL": 4,
        "VENTURE": 5
    }

    grade_map = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6
    }

    default_map = {
        "N": 0,
        "Y": 1
    }

    features = np.array([[
        age,
        income,
        home_map[home],
        emp_length,
        intent_map[loan_intent],
        grade_map[loan_grade],
        loan_amnt,
        loan_int_rate,
        loan_percent_income,
        default_map[default_file],
        cred_hist
    ]])

    prediction = model.predict(features)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("❌ High Credit Risk")
        st.progress(85)
    else:
        st.success("✅ Low Credit Risk")
        st.progress(25)

st.divider()

st.caption("Developed using Python, Streamlit and Random Forest Machine Learning")