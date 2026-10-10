import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import uuid
import os
from datetime import datetime



# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# =========================================
# LOAD MODELS
# =========================================

ann_model = tf.keras.models.load_model(
    "ann_model.h5",
    compile=False
)

with open("random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("xgboost_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

with open("ensemble_model.pkl", "rb") as f:
    ensemble_model = pickle.load(f)

with open("hybrid_meta_model.pkl", "rb") as f:
    hybrid_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# =========================================
# CSV FILE
# =========================================

DATA_FILE = "patient_records.csv"

if not os.path.exists(DATA_FILE):

    columns = [
        "patient_id",
        "date",
        "model_used",
        "age",
        "gender",
        "bmi",
        "smoking",
        "hypertension",
        "diabetes",
        "family_history",
        "obesity",
        "alcohol",
        "tobacco",
        "ldl",
        "hdl",
        "bp",
        "chest_pain",
        "breathlessness",
        "exercise_pain",
        "prediction"
    ]

    pd.DataFrame(columns=columns).to_csv(
        DATA_FILE,
        index=False
    )

# =========================================
# CUSTOM CSS
# =========================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

h1, h2, h3 {
    color: white;
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 20px;
}

.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    color: white;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# TITLE
# =========================================

st.title("❤️ Heart Disease Prediction Dashboard")

st.markdown(
    "### AI Based Heart Disease Detection System"
)

st.markdown("---")

# =========================================
# SIDEBAR
# =========================================

menu = st.sidebar.selectbox(
    "Choose Option",
    [
        "New Prediction",
        "Retrieve Patient Record",
        "Delete Patient Record"
    ]
)

# =========================================================
# RETRIEVE PATIENT RECORD
# =========================================================

if menu == "Retrieve Patient Record":

    st.subheader("🔎 Retrieve Patient Details")

    df = pd.read_csv(DATA_FILE)

    if len(df) == 0:

        st.warning("No records available yet")

    else:

        patient_ids = df["patient_id"].tolist()

        selected_id = st.selectbox(
            "Select Patient ID",
            patient_ids
        )

        if st.button("Get Details"):

            patient = df[
                df["patient_id"] == selected_id
            ]

            if len(patient) > 0:

                patient = patient.iloc[0]

                st.success("Patient Record Found ✅")

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown("### 👤 Patient Information")

                    st.write(
                        f"🆔 Patient ID: {patient['patient_id']}"
                    )

                    st.write(
                        f"📅 Date: {patient['date']}"
                    )

                    st.write(
                        f"🤖 Model Used: {patient['model_used']}"
                    )

                    st.write(
                        f"👤 Age: {patient['age']}"
                    )

                    st.write(
                        f"⚧ Gender: {patient['gender']}"
                    )

                    st.write(
                        f"⚖ BMI: {patient['bmi']}"
                    )

                    st.write(
                        f"🩸 Blood Pressure: {patient['bp']}"
                    )

                with col2:

                    st.markdown("### 📊 Health Report")

                    st.write(
                        f"🚬 Smoking: {patient['smoking']}"
                    )

                    st.write(
                        f"🧪 LDL: {patient['ldl']}"
                    )

                    st.write(
                        f"💙 HDL: {patient['hdl']}"
                    )

                    st.write(
                        f"⚠ Chest Pain: {patient['chest_pain']}"
                    )

                    st.write(
                        f"😮 Breathlessness: {patient['breathlessness']}"
                    )

                    st.write(
                        f"📋 Prediction: {patient['prediction']}"
                    )


# =========================================================
# NEW PREDICTION
# =========================================================
# =========================================================
# DELETE PATIENT RECORD
# =========================================================

elif menu == "Delete Patient Record":

    st.subheader("🗑 Delete Patient Record")

    df = pd.read_csv(DATA_FILE)

    if len(df) == 0:

        st.warning("No records available")

    else:

        patient_ids = df["patient_id"].tolist()

        selected_id = st.selectbox(
            "Select Patient ID to Delete",
            patient_ids
        )

        patient = df[
            df["patient_id"] == selected_id
        ]

        if len(patient) > 0:

            patient = patient.iloc[0]

            st.markdown("### 👤 Patient Details")

            st.write(
                f"🆔 Patient ID: {patient['patient_id']}"
            )

            st.write(
                f"👤 Age: {patient['age']}"
            )

            st.write(
                f"⚧ Gender: {patient['gender']}"
            )

            st.write(
                f"🤖 Model Used: {patient['model_used']}"
            )

            st.write(
                f"📋 Prediction: {patient['prediction']}"
            )

        if st.button("❌ Delete Record"):

            df = df[
                df["patient_id"] != selected_id
            ]

            df.to_csv(
                DATA_FILE,
                index=False
            )

            st.success(
                "✅ Record Deleted Successfully"
            )
else:

    # =========================================
    # SIDEBAR MODEL SELECTION
    # =========================================

    st.sidebar.title("Choose Model")

    model_choice = st.sidebar.selectbox(
        "Select Prediction Model",
        [
            "ANN",
            "Random Forest",
            "XGBoost",
            "Ensemble",
            "Hybrid Meta Model"
        ]
    )

    # =========================================
    # INPUT SECTION
    # =========================================

    st.header("🧾 Enter Patient Details")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            1,
            100,
            30
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        bmi = st.number_input(
            "BMI",
            10.0,
            50.0,
            22.0
        )

        smoking = st.selectbox(
            "Smoking",
            ["Yes", "No"]
        )

        hypertension = st.selectbox(
            "Hypertension",
            ["Yes", "No"]
        )

        diabetes = st.selectbox(
            "Diabetes",
            ["Yes", "No"]
        )

        family_history = st.selectbox(
            "Family History",
            ["Yes", "No"]
        )

        obesity = st.selectbox(
            "Obesity",
            ["Yes", "No"]
        )

    with col2:

        alcohol = st.selectbox(
            "Alcohol Consumption",
            ["Yes", "No"]
        )

        tobacco = st.selectbox(
            "Tobacco Use",
            ["Yes", "No"]
        )

        ldl = st.number_input(
            "LDL Cholesterol",
            0.0,
            300.0,
            100.0
        )

        hdl = st.number_input(
            "HDL Cholesterol",
            0.0,
            150.0,
            50.0
        )

        bp = st.number_input(
            "Blood Pressure",
            50.0,
            250.0,
            120.0
        )

        chest_pain = st.selectbox(
            "Chest Pain",
            ["Yes", "No"]
        )

        breathlessness = st.selectbox(
            "Breathlessness",
            ["Yes", "No"]
        )

        exercise_pain = st.selectbox(
            "Exercise Pain",
            ["Yes", "No"]
        )

    # =========================================
    # ENCODING
    # =========================================

    gender_value = 1 if gender == "Male" else 0

    smoking_value = 1 if smoking == "Yes" else 0
    hypertension_value = 1 if hypertension == "Yes" else 0
    diabetes_value = 1 if diabetes == "Yes" else 0
    family_history_value = 1 if family_history == "Yes" else 0
    obesity_value = 1 if obesity == "Yes" else 0
    alcohol_value = 1 if alcohol == "Yes" else 0
    tobacco_value = 1 if tobacco == "Yes" else 0
    chest_pain_value = 1 if chest_pain == "Yes" else 0
    breathlessness_value = 1 if breathlessness == "Yes" else 0
    exercise_pain_value = 1 if exercise_pain == "Yes" else 0

    # =========================================
    # INPUT ARRAY
    # =========================================

    input_data = np.array([[

        age,
        gender_value,
        bmi,
        smoking_value,
        hypertension_value,
        diabetes_value,
        family_history_value,
        obesity_value,
        alcohol_value,
        tobacco_value,
        ldl,
        hdl,
        bp,
        chest_pain_value,
        breathlessness_value,
        exercise_pain_value

    ]])

    # =========================================
    # SCALING
    # =========================================

    scaled_data = scaler.transform(input_data)

    # =========================================
    # PREDICTION
    # =========================================

    if st.button("🔍 Predict Risk"):

        if model_choice == "ANN":

            prediction = ann_model.predict(
                scaled_data
            )

            result = 1 if prediction[0][0] > 0.5 else 0

        elif model_choice == "Random Forest":

            result = rf_model.predict(
                scaled_data
            )[0]

        elif model_choice == "XGBoost":

            result = xgb_model.predict(
                scaled_data
            )[0]

        elif model_choice == "Ensemble":

            result = ensemble_model.predict(
                scaled_data
            )[0]

        elif model_choice == "Hybrid Meta Model":

            result = hybrid_model.predict(
                scaled_data
            )[0]

        # =========================================
        # RESULT SECTION
        # =========================================

        st.markdown("---")

        st.subheader("📊 Prediction Result")

        # =========================================
        # HEALTH ANALYSIS
        # =========================================

        st.markdown("### 🧠 AI Health Analysis")

        reasons = []

        if age > 55:
            reasons.append(
                "High age increases heart disease probability"
            )

        if bmi > 30:
            reasons.append(
                "High BMI indicates obesity risk"
            )

        if ldl > 160:
            reasons.append(
                "High LDL cholesterol detected"
            )

        if bp > 140:
            reasons.append(
                "Blood pressure is higher than normal"
            )

        if smoking == "Yes":
            reasons.append(
                "Smoking increases heart disease risk"
            )

        if diabetes == "Yes":
            reasons.append(
                "Diabetes increases cardiovascular risk"
            )

        if chest_pain == "Yes":
            reasons.append(
                "Chest pain symptom detected"
            )

        for reason in reasons:
            st.warning(reason)

        # =========================================
        # HIGH RISK
        # =========================================

        if result == 1:

            prediction_text = "Coronary Artery Disease "

            st.error(
                "⚠️ High Risk of Coronary Artery Disease"
            )

            st.markdown("---")

            st.subheader(
                "🏃 Recommended Exercises"
            )

            ex1, ex2, ex3 = st.columns(3)

            with ex1:

                st.image(
                    "https://images.unsplash.com/photo-1518611012118-696072aa579a",
                    use_container_width=True
                )

                st.markdown("### 🚶 Yoga")

                st.write(
                    "Yoga helps reduce stress and blood pressure."
                )

            with ex2:

                st.image(
                    "https://images.unsplash.com/photo-1517836357463-d25dfeac3438",
                    use_container_width=True
                )

                st.markdown("### 🧘 Exercise ")

                st.write(
                    "Daily exercise reduces heart related issues."
                )

            with ex3:

                st.image(
                    "https://images.unsplash.com/photo-1541625602330-2277a4c46182",
                    use_container_width=True
                )

                st.markdown("### 🚴 Cycling")

                st.write(
                    "Cycling strengthens cardiovascular health."
                )

            # =========================================
            # FOOD RECOMMENDATIONS
            # =========================================

            st.markdown("---")

            st.subheader(
                "🥗 Recommended Foods"
            )

            food1, food2, food3 = st.columns(3)

            with food1:

                st.image(
                    "https://images.unsplash.com/photo-1540189549336-e6e99c3679fe",
                    use_container_width=True
                )

                st.markdown(
                    "### 🥦 Green Vegetables"
                )

                st.write(
                    "Rich in fiber and vitamins."
                )

            with food2:

                st.image(
                    "https://images.unsplash.com/photo-1512621776951-a57141f2eefd",
                    use_container_width=True
                )

                st.markdown(
                    "### 🥗 Healthy Salad"
                )

                st.write(
                    "Helps reduce cholesterol levels."
                )

            with food3:

                st.image(
                    "https://images.unsplash.com/photo-1490645935967-10de6ba17061",
                    use_container_width=True
                )

                st.markdown("### 🍎 Fruits")

                st.write(
                    "Fruits improve heart and immune system."
                )

            # =========================================
            # EXTRA RECOMMENDATIONS
            # =========================================

            st.markdown("---")

            st.subheader(
                "💡 Additional Recommendations"
            )

            st.info(
                "✔ Reduce oily and junk food"
            )

            st.info(
                "✔ Avoid smoking and alcohol"
            )

            st.info(
                "✔ Sleep at least 7-8 hours daily"
            )

            st.info(
                "✔ Drink more water"
            )

            st.info(
                "✔ Regular health checkups are recommended"
            )

        # =========================================
        # LOW RISK
        # =========================================

        else:

            prediction_text = "Low Risk"

            st.success(
                "✅ Low Risk of Heart Disease"
            )



            st.success(
                "Maintain healthy diet and regular exercise"
            )

            st.success(
                "Continue regular walking and hydration"
            )

            st.success(
                "Avoid stress and maintain sleep cycle"
            )

        # =========================================
        # SAVE RECORD
        # =========================================

        # =========================================
        # CREATE SMART UNIQUE PATIENT ID
        # =========================================

        # Model Short Names

        model_short = {

            "ANN": "ANN",
            "Random Forest": "RF",
            "XGBoost": "XGB",
            "Ensemble": "ENS",
            "Hybrid Meta Model": "HYB"

        }

        # Risk Type

        risk_type = "H" if result == 1 else "L"



        # Random Unique Code

        unique_code = str(uuid.uuid4())[:4].upper()

        # Final Patient ID

        patient_id = (
            f"{model_short[model_choice]}-"
            f"{risk_type}"
            f"{unique_code}"
        )

        save_data = {

            "patient_id": patient_id,

            "date": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "model_used": model_choice,

            "age": age,
            "gender": gender,
            "bmi": bmi,
            "smoking": smoking,
            "hypertension": hypertension,
            "diabetes": diabetes,
            "family_history": family_history,
            "obesity": obesity,
            "alcohol": alcohol,
            "tobacco": tobacco,
            "ldl": ldl,
            "hdl": hdl,
            "bp": bp,
            "chest_pain": chest_pain,
            "breathlessness": breathlessness,
            "exercise_pain": exercise_pain,

            "prediction": prediction_text
        }

        save_df = pd.DataFrame([save_data])

        save_df.to_csv(
            DATA_FILE,
            mode='a',
            header=False,
            index=False
        )

        st.markdown("---")

        st.success(
            "✅ Record Saved Successfully"
        )

        st.info(f"""
        🆔 Your Patient ID is:

        # {patient_id}

        Save this ID to retrieve records later.
        """)

# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.markdown("""
💡 This system provides AI-based prediction for educational purposes only.
""")