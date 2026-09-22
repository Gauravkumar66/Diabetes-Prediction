import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp {
        background: #0f1117;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(135deg, #182848, #304a78);
        padding: 35px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 25px;
        border: 1px solid #3d557d;
    }

    .hero h1 {
        color: #ffffff !important;
        font-size: 42px !important;
        font-weight: 700 !important;
        margin: 0 !important;
    }

    .hero p {
        color: #dbe7ff !important;
        font-size: 18px !important;
        margin-top: 10px !important;
    }

    .info-box {
        background: #1e293b !important;
        padding: 25px !important;
        border-radius: 16px !important;
        border: 1px solid #334155 !important;
        margin-bottom: 25px !important;
    }

    .info-box h3 {
        color: #ffffff !important;
        font-size: 22px !important;
        margin-bottom: 12px !important;
    }

    .info-box p {
        color: #d1d5db !important;
        font-size: 16px !important;
        line-height: 1.7 !important;
    }

    .section-title {
        color: #ffffff !important;
        font-size: 25px !important;
        font-weight: 600 !important;
        margin-top: 20px !important;
        margin-bottom: 15px !important;
    }

    label {
        color: #e5e7eb !important;
        font-weight: 500 !important;
    }

    .stNumberInput input {
        color: #ffffff !important;
        background-color: #1e293b !important;
        border: 1px solid #475569 !important;
    }

    .stNumberInput input:focus {
        border-color: #60a5fa !important;
        box-shadow: 0 0 0 1px #60a5fa !important;
    }

    .stButton > button {
        width: 100%;
        height: 55px;
        border-radius: 12px;
        background: #2563eb !important;
        color: #ffffff !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        border: none !important;
    }

    .stButton > button:hover {
        background: #1d4ed8 !important;
        color: #ffffff !important;
    }

    .result-positive {
        background: #3f1d1d !important;
        border: 1px solid #ef4444 !important;
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        margin-top: 25px;
    }

    .result-positive h2 {
        color: #fca5a5 !important;
        margin: 0 !important;
    }

    .result-negative {
        background: #143524 !important;
        border: 1px solid #22c55e !important;
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        margin-top: 25px;
    }

    .result-negative h2 {
        color: #86efac !important;
        margin: 0 !important;
    }

    .probability-box {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        padding: 20px;
        border-radius: 16px;
        margin-top: 20px;
    }

    .probability-box h3 {
        color: #ffffff !important;
        margin-top: 0 !important;
    }

    .footer {
        text-align: center;
        color: #94a3b8 !important;
        margin-top: 45px;
        padding-top: 20px;
        border-top: 1px solid #334155;
        font-size: 14px;
    }

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    .stAlert {
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        return joblib.load("model/diabetes_model.pkl")
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


model = load_model()


st.markdown("""
<div class="hero">
    <h1>🩺 Diabetes Prediction System</h1>
    <p>Machine Learning Based Diabetes Risk Prediction using SVM</p>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="info-box">
    <h3>About This Application</h3>
    <p>
        This application uses a trained Support Vector Machine (SVM)
        machine learning model to predict whether a person is likely
        to have diabetes based on medical measurements.
    </p>
    <p>
        Enter the patient's information below and click
        <strong>Predict Diabetes</strong> to generate a prediction.
    </p>
</div>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="section-title">📋 Patient Information</div>',
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)


with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        max_value=300,
        value=120,
        step=1
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=200,
        value=70,
        step=1
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20,
        step=1
    )


with col2:
    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=900,
        value=80,
        step=1
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        step=0.1
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )


st.write("")


if st.button("🔍 Predict Diabetes", use_container_width=True):

    if model is None:
        st.error(
            "Model file not found. Make sure diabetes_model.pkl "
            "is in the same folder as app.py."
        )

    else:
        input_data = pd.DataFrame(
            [[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree,
                age
            ]],
            columns=[
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age"
            ]
        )

        try:
            prediction = model.predict(input_data)[0]

            probability = None

            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(input_data)[0][1]

            if prediction == 1:
                st.markdown("""
                <div class="result-positive">
                    <h2>⚠️ Diabetes Prediction: POSITIVE</h2>
                </div>
                """, unsafe_allow_html=True)

                st.warning(
                    "The model predicts that the patient may have diabetes. "
                    "This result is a machine-learning prediction and is not "
                    "a medical diagnosis."
                )

            else:
                st.markdown("""
                <div class="result-negative">
                    <h2>✅ Diabetes Prediction: NEGATIVE</h2>
                </div>
                """, unsafe_allow_html=True)

                st.success(
                    "The model predicts that the patient is unlikely "
                    "to have diabetes."
                )

            if probability is not None:
                probability_percentage = probability * 100

                st.markdown("""
                <div class="probability-box">
                    <h3>📊 Prediction Probability</h3>
                </div>
                """, unsafe_allow_html=True)

                st.progress(float(probability))

                st.markdown(
                    f"### Estimated Diabetes Probability: {probability_percentage:.2f}%"
                )

            st.markdown(
                '<div class="section-title">📄 Patient Input Summary</div>',
                unsafe_allow_html=True
            )

            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")


st.markdown("""
<div class="footer">
    Diabetes Prediction System | SVM Machine Learning Project<br>
    For educational purposes only
</div>
""", unsafe_allow_html=True)