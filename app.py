import streamlit as st
import pickle
import numpy as np
import base64

# Page Config
st.set_page_config(
    page_title="Credit Risk Prediction System",
    page_icon="💳",
    layout="centered"
)

# Background Image
def set_background():
    with open("background.png", "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        [data-testid="stSidebar"] {{
            background-color: rgba(0,0,0,0.75);
        }}

        .main-title {{
            text-align: center;
            color: white;
            font-size: 80px;
            font-weight: bold;
        }}

        .sub-title {{
            text-align: center;
            color: white;
            font-size: 34px;
            margin-bottom: 30px;
        }}

        label {{
            font-size: 24px !important;
            font-weight: bold !important;
            color: white !important;
        }}

        .stButton > button {{
            width: 100%;
            height: 70px;
            font-size: 32px;
            font-weight: bold;
            border-radius: 12px;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

# Load Model
model = pickle.load(open("model.pkl", "rb"))

# Header
st.markdown(
    "<div class='main-title'>💳 Credit Risk Prediction System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Machine Learning Based Loan Risk Assessment</div>",
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("📊 Model Performance")

st.sidebar.success("Accuracy : 93.52%")
st.sidebar.success("Precision : 92.10%")
st.sidebar.success("Recall : 90.21%")
st.sidebar.success("F1 Score : 91.14%")
st.sidebar.success("ROC-AUC : 94.50%")

st.sidebar.markdown("---")

st.sidebar.info("""
Algorithm Used:
Random Forest Classifier

Dataset:
Credit Risk Dataset (Kaggle)

Purpose:
Predict whether an applicant is a low-risk or high-risk borrower.
""")

# Form
left_space, main_col, right_space = st.columns([1,8,1])

with main_col:
        st.markdown("<br>", unsafe_allow_html=True)

age = st.number_input(
    "Person Age",
    min_value=18,
    max_value=100,
    value=30
)

income = st.number_input(
    "Annual Income",
    min_value=0,
    value=50000
)

home = st.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

emp_length = st.number_input(
    "Employment Length (Years)",
    min_value=0,
    value=2
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

loan_grade = st.selectbox(
    "Loan Grade",
    ["A", "B", "C", "D", "E", "F", "G"]
)

loan_amnt = st.number_input(
    "Loan Amount",
    min_value=0,
    value=10000
)

loan_int_rate = st.number_input(
    "Interest Rate",
    value=10.0
)

loan_percent_income = st.number_input(
    "Loan Percent Income",
    value=0.20
)

default_file = st.selectbox(
    "Default On File",
    ["N", "Y"]
)

cred_hist = st.number_input(
    "Credit History Length",
    min_value=0,
    value=5
)

st.markdown("<br>", unsafe_allow_html=True)

predict = st.button("🔍 Predict Credit Risk")

if predict:

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

    st.markdown("## Prediction Result")

    if prediction[0] == 1:
        st.error("❌ High Credit Risk")
    else:
        st.success("✅ Low Credit Risk")

st.markdown("---")

st.caption(
    "Developed using Python, Streamlit and Random Forest Machine Learning"
)