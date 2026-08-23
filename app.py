import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)


# ============================================================
# LOAD MODEL, SCALER AND COLUMNS
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load("LOR_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")

    return model, scaler, columns


# Load files
model, scaler, columns = load_model()


# ============================================================
# HEADER
# ============================================================

st.title("❤️ Heart Disease Prediction System")

st.write(
    "Enter the patient's medical information below "
    "to predict the possibility of heart disease."
)

st.divider()


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.subheader("👤 Patient Information")

col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50,
        step=1
    )

    sex = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=120,
        step=1
    )

    cholesterol = st.number_input(
        "Cholesterol",
        min_value=0,
        max_value=600,
        value=200,
        step=1
    )


with col2:

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar",
        [
            "Normal (≤ 120 mg/dl)",
            "High (> 120 mg/dl)"
        ]
    )

    max_hr = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150,
        step=1
    )

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=-5.0,
        max_value=10.0,
        value=0.0,
        step=0.1
    )


# ============================================================
# HEART INFORMATION
# ============================================================

st.subheader("🫀 Heart Information")


chest_pain = st.selectbox(
    "Chest Pain Type",
    [
        "ATA",
        "NAP",
        "TA",
        "ASY"
    ]
)


resting_ecg = st.selectbox(
    "Resting ECG",
    [
        "Normal",
        "ST",
        "LVH"
    ]
)


exercise_angina = st.selectbox(
    "Exercise-Induced Angina",
    [
        "Yes",
        "No"
    ]
)


st_slope = st.selectbox(
    "ST Slope",
    [
        "Up",
        "Flat",
        "Down"
    ]
)


st.divider()


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔍 Predict Heart Disease",
    use_container_width=True
):

    # --------------------------------------------------------
    # CREATE INPUT DATA
    # --------------------------------------------------------

    input_data = {

        "Age": age,

        "RestingBP": resting_bp,

        "Cholesterol": cholesterol,

        "FastingBS":
            1 if fasting_bs == "High (> 120 mg/dl)" else 0,

        "MaxHR": max_hr,

        "Oldpeak": oldpeak,

        "Sex_M":
            1 if sex == "Male" else 0,

        "ChestPainType_ATA":
            1 if chest_pain == "ATA" else 0,

        "ChestPainType_NAP":
            1 if chest_pain == "NAP" else 0,

        "ChestPainType_TA":
            1 if chest_pain == "TA" else 0,

        "RestingECG_Normal":
            1 if resting_ecg == "Normal" else 0,

        "RestingECG_ST":
            1 if resting_ecg == "ST" else 0,

        "ExerciseAngina_Y":
            1 if exercise_angina == "Yes" else 0,

        "ST_Slope_Flat":
            1 if st_slope == "Flat" else 0,

        "ST_Slope_Up":
            1 if st_slope == "Up" else 0
    }


    # --------------------------------------------------------
    # CONVERT TO DATAFRAME
    # --------------------------------------------------------

    input_df = pd.DataFrame([input_data])


    # --------------------------------------------------------
    # MAKE SURE FEATURE ORDER IS SAME AS TRAINING
    # --------------------------------------------------------

    input_df = input_df.reindex(
        columns=columns,
        fill_value=0
    )


    # --------------------------------------------------------
    # SCALE INPUT
    # --------------------------------------------------------

    input_scaled = scaler.transform(input_df)


    # --------------------------------------------------------
    # MAKE PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(input_scaled)[0]


    # --------------------------------------------------------
    # GET PROBABILITY
    # --------------------------------------------------------

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(input_scaled)[0][1]

    else:

        probability = None


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.subheader("📊 Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ Higher possibility of Heart Disease"
        )

    else:

        st.success(
            "✅ Lower possibility of Heart Disease"
        )


    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    if probability is not None:

        st.metric(
            "Heart Disease Probability",
            f"{probability * 100:.2f}%"
        )

        st.progress(float(probability))


    # --------------------------------------------------------
    # INPUT SUMMARY
    # --------------------------------------------------------

    with st.expander("View Patient Information"):

        st.write(
            f"**Age:** {age}"
        )

        st.write(
            f"**Sex:** {sex}"
        )

        st.write(
            f"**Resting Blood Pressure:** {resting_bp}"
        )

        st.write(
            f"**Cholesterol:** {cholesterol}"
        )

        st.write(
            f"**Fasting Blood Sugar:** {fasting_bs}"
        )

        st.write(
            f"**Maximum Heart Rate:** {max_hr}"
        )

        st.write(
            f"**Oldpeak:** {oldpeak}"
        )

        st.write(
            f"**Chest Pain Type:** {chest_pain}"
        )

        st.write(
            f"**Resting ECG:** {resting_ecg}"
        )

        st.write(
            f"**Exercise Angina:** {exercise_angina}"
        )

        st.write(
            f"**ST Slope:** {st_slope}"
        )


    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.warning(
        "⚠️ This application is developed for educational "
        "and project purposes only. It is not a medical "
        "diagnosis and should not replace professional "
        "medical advice."
    )