# 🏥 Smart Hospital Queue & Patient Flow Intelligence System

A **Streamlit-based Machine Learning application** that estimates hospital waiting time using a **Random Forest Regression model** and provides queue indicators, prediction history, and hospital analytics.

## 🎯 Project Objective

The system is designed to estimate expected hospital waiting time based on patient-flow and hospital-condition inputs such as:

- Hospital
- Department
- Day
- Time
- Patients waiting
- Doctors available
- Emergency cases
- Holiday status

The application also converts the predicted waiting time into a simple **crowd level, queue status, and recommendation**.

## ✨ Features

### ⏱ Waiting Time Prediction
Predicts the estimated hospital waiting time using a trained Random Forest Regression model.

### 👥 Crowd & Queue Classification
Classifies the predicted waiting condition into:

- 🟢 Low
- 🟡 Medium
- 🔴 High

and provides a corresponding queue status.

### 💡 Smart Recommendation
Provides a simple recommendation based on the predicted waiting time.

### 📊 Analytics Dashboard
Includes descriptive analytics such as:

- Average waiting time by hospital
- Average waiting time by department
- Average waiting time by day
- Patient distribution
- Overall hospital and department statistics

### 📜 Prediction History
Stores predictions made during the current Streamlit session and displays them in a table.

## 🛠 Tech Stack

- **Python**
- **Streamlit**
- **Pandas**
- **Scikit-learn**
- **Matplotlib**
- **Joblib**

## ⚙️ How It Works

```text
User Input
    ↓
Data Preprocessing
    ↓
Label Encoding
    ↓
Random Forest Regression Model
    ↓
Estimated Waiting Time
    ↓
Crowd Level + Queue Status
    ↓
Recommendation