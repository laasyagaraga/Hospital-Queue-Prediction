import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------
# Page Configuration
# ------------------------------------

st.set_page_config(
    page_title="Hospital Queue Prediction",
    page_icon="🏥",
    layout="wide"
)

# ------------------------------------
# Load Model and Dataset
# ------------------------------------

model = joblib.load("hospital_model.pkl")
encoders = joblib.load("label_encoders.pkl")
data = pd.read_csv("dataset.csv")

# ------------------------------------
# Prediction History
# ------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------------------
# Sidebar
# ------------------------------------

st.sidebar.title("🏥 Hospital Queue Predictor")
st.sidebar.markdown("---")

st.sidebar.info("""
### About

This application predicts hospital waiting time using a Machine Learning model.

### Features

- ⏱ Waiting Time Prediction
- 👥 Crowd Level
- 🚑 Queue Status
- 💡 Smart Recommendation
- 📊 Analytics Dashboard
""")

st.sidebar.markdown("---")
st.sidebar.success("Developed by Laasya")

# ------------------------------------
# Title
# ------------------------------------

st.title("🏥 Smart Hospital Queue & Patient Flow Intelligence System")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🩺 Prediction", "📊 Analytics", "📜 History", "ℹ About"]
)

# ==========================================================
# TAB 1 - PREDICTION
# ==========================================================

with tab1:

    st.markdown("""
    Welcome to the **Smart Hospital Queue & Patient Flow Intelligence System**.

    This application uses a **Random Forest Machine Learning model**
    to estimate hospital waiting time and provides analytics dashboards.
    """)

    st.subheader("📝 Enter Hospital Details")

    st.info(
        "Fill in the hospital details below and click **Predict Waiting Time** "
        "to estimate the expected waiting time."
    )

    # ------------------------------------
    # Inputs
    # ------------------------------------

    hospital = st.selectbox(
        "Hospital",
        ["Apollo", "AIIMS", "Care", "Yashoda"]
    )

    department = st.selectbox(
        "Department",
        [
            "General",
            "Cardiology",
            "Orthopedics",
            "Dermatology",
            "Neurology"
        ]
    )

    day = st.selectbox(
        "Day",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )

    time = st.slider(
        "Time (24 Hour Format)",
        8,
        18,
        10
    )

    patients = st.number_input(
        "Patients Waiting",
        min_value=1,
        max_value=100,
        value=20
    )

    doctors = st.number_input(
        "Doctors Available",
        min_value=1,
        max_value=10,
        value=4
    )

    emergency = st.number_input(
        "Emergency Cases",
        min_value=0,
        max_value=20,
        value=2
    )

    holiday = st.selectbox(
        "Holiday",
        ["No", "Yes"]
    )

    st.markdown("---")

    # ------------------------------------
    # Prediction
    # ------------------------------------

    if st.button("🔍 Predict Waiting Time"):

        sample = pd.DataFrame(
            [[
                encoders["hospital"].transform([hospital])[0],
                encoders["department"].transform([department])[0],
                encoders["day"].transform([day])[0],
                time,
                patients,
                doctors,
                emergency,
                encoders["holiday"].transform([holiday])[0]
            ]],
            columns=[
                "Hospital",
                "Department",
                "Day",
                "Time",
                "Patients",
                "DoctorsAvailable",
                "EmergencyCases",
                "Holiday"
            ]
        )

        waiting_time = model.predict(sample)[0]

        # Save prediction history
        st.session_state.history.append({
            "Hospital": hospital,
            "Department": department,
            "Day": day,
            "Time": time,
            "Patients": patients,
            "Doctors": doctors,
            "Emergency Cases": emergency,
            "Holiday": holiday,
            "Predicted Wait (min)": round(waiting_time, 2)
        })

        # ------------------------------------
        # Crowd / Queue Classification
        # ------------------------------------

        if waiting_time < 30:
            crowd = "🟢 Low"
            queue = "Normal"
            recommendation = "✅ Good time to visit."

        elif waiting_time < 60:
            crowd = "🟡 Medium"
            queue = "Busy"
            recommendation = "⚠ Moderate waiting expected."

        else:
            crowd = "🔴 High"
            queue = "Very Busy"
            recommendation = "❌ Heavy crowd. Visit later if possible."

        # ------------------------------------
        # Prediction Result
        # ------------------------------------

        st.success(
            f"🏥 Estimated Waiting Time: {waiting_time:.2f} minutes"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "⏱ Waiting Time",
            f"{waiting_time:.2f} min"
        )

        c2.metric(
            "👥 Crowd Level",
            crowd
        )

        c3.metric(
            "🚑 Queue Status",
            queue
        )

        st.subheader("💡 Recommendation")

        if waiting_time < 30:
            st.success(recommendation)

        elif waiting_time < 60:
            st.warning(recommendation)

        else:
            st.error(recommendation)


# ==========================================================
# TAB 2 - ANALYTICS
# ==========================================================

with tab2:

    st.header("📊 Hospital Analytics Dashboard")

    selected_hospital = st.selectbox(
        "🏥 Select Hospital",
        ["All"] + sorted(data["Hospital"].unique())
    )

    if selected_hospital == "All":
        filtered_data = data
    else:
        filtered_data = data[
            data["Hospital"] == selected_hospital
        ]

    # ------------------------------------
    # Dashboard Metrics
    # ------------------------------------

    total_hospitals = filtered_data["Hospital"].nunique()
    total_departments = filtered_data["Department"].nunique()
    avg_wait = round(filtered_data["WaitTime"].mean(), 2)
    total_records = len(filtered_data)

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("🏥 Hospitals", total_hospitals)
    m2.metric("🩺 Departments", total_departments)
    m3.metric("⏱ Avg Wait Time", f"{avg_wait} min")
    m4.metric("📋 Records", total_records)

    # ------------------------------------
    # Charts
    # ------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        hospital_avg = (
            filtered_data
            .groupby("Hospital")["WaitTime"]
            .mean()
        )

        fig, ax = plt.subplots(figsize=(5, 4))

        hospital_avg.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Average Waiting Time by Hospital")
        ax.set_ylabel("Minutes")

        st.pyplot(fig)
        plt.close(fig)

    with col2:

        dept_avg = (
            filtered_data
            .groupby("Department")["WaitTime"]
            .mean()
        )

        fig, ax = plt.subplots(figsize=(5, 4))

        dept_avg.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Average Waiting Time by Department")
        ax.set_ylabel("Minutes")

        st.pyplot(fig)
        plt.close(fig)

    col3, col4 = st.columns(2)

    with col3:

        days_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]

        day_avg = (
            filtered_data
            .groupby("Day")["WaitTime"]
            .mean()
            .reindex(days_order)
        )

        day_avg.index = [
            "Mon",
            "Tue",
            "Wed",
            "Thu",
            "Fri",
            "Sat",
            "Sun"
        ]

        fig, ax = plt.subplots(figsize=(5, 4))

        day_avg.plot(
            kind="line",
            marker="o",
            linewidth=3,
            ax=ax
        )

        ax.set_title("Average Waiting Time by Day")
        ax.set_xlabel("Day")
        ax.set_ylabel("Minutes")
        ax.grid(True)

        st.pyplot(fig)
        plt.close(fig)

    with col4:

        patient_count = (
            filtered_data["Department"]
            .value_counts()
        )

        fig, ax = plt.subplots(figsize=(5, 4))

        patient_count.plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=ax
        )

        ax.set_ylabel("")
        ax.set_title("Patients Distribution")

        st.pyplot(fig)
        plt.close(fig)

    # ------------------------------------
    # Project Insights
    # ------------------------------------

    st.divider()
    st.header("📌 Project Insights")

    highest_hospital = (
        filtered_data
        .groupby("Hospital")["WaitTime"]
        .mean()
        .idxmax()
    )

    highest_department = (
        filtered_data
        .groupby("Department")["WaitTime"]
        .mean()
        .idxmax()
    )

    busiest_day = (
        filtered_data
        .groupby("Day")["WaitTime"]
        .mean()
        .idxmax()
    )

    avg_wait = round(
        filtered_data["WaitTime"].mean(),
        2
    )

    st.info(f"""
🏥 **Hospital with Highest Average Waiting Time:** {highest_hospital}

🩺 **Department with Highest Average Waiting Time:** {highest_department}

📅 **Busiest Day:** {busiest_day}

⏱ **Average Waiting Time:** {avg_wait} minutes
""")


# ==========================================================
# TAB 3 - HISTORY
# ==========================================================

with tab3:

    st.header("📜 Prediction History")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

        if st.button("🗑 Clear History"):

            st.session_state.history = []

            st.rerun()

    else:

        st.info(
            "No prediction history available yet. "
            "Make a prediction from the Prediction tab."
        )


# ==========================================================
# TAB 4 - ABOUT
# ==========================================================

with tab4:

    st.header("📖 About This Project")

    st.markdown("""
    ### 🏥 Smart Hospital Queue & Patient Flow Intelligence System

    This project uses **Machine Learning (Random Forest Regression)**
    to predict hospital waiting time based on patient flow and
    hospital conditions.

    ### 🎯 Objectives

    - Predict expected waiting time
    - Estimate crowd level
    - Help patients understand expected waiting conditions
    - Visualize hospital analytics

    ### 🛠 Technologies Used

    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - Matplotlib
    - Joblib

    ### ⚙️ How It Works

    1. User enters hospital and patient-flow details.
    2. The input values are processed.
    3. A Random Forest Regression model predicts the expected waiting time.
    4. The application displays the estimated waiting time.
    5. The system classifies the crowd and queue status.
    6. Analytics are generated from the hospital dataset.

    ### 🚀 Future Scope

    - Live hospital data integration
    - Appointment booking
    - Real-time queue tracking
    - Mobile application support
    """)

    st.markdown("---")

    st.caption(
        "© 2026 Smart Hospital Queue & Patient Flow Intelligence System "
        "| Developed by Laasya"
    )