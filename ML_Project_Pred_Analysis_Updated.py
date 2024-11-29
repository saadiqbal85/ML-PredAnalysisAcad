#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Generate Synthetic Data
np.random.seed(42)
n_samples = 1000
data = {
    "Age": np.random.randint(13, 19, n_samples),
    "Gender": np.random.choice(["Male", "Female"], n_samples),
    "Ethnicity": np.random.choice(["Caucasian", "African American", "Asian", "Other"], n_samples),
    "Parental_Education": np.random.randint(1, 5, n_samples),
    "Parental_Support": np.random.randint(1, 10, n_samples),
    "Weekly_Study_Hours": np.random.randint(1, 20, n_samples),
    "Absences": np.random.randint(0, 15, n_samples),
    "Sports": np.random.choice([0, 1], n_samples),
    "Music": np.random.choice([0, 1], n_samples),
    "Volunteering": np.random.choice([0, 1], n_samples),
    "Past_GPA": np.round(np.random.uniform(2.0, 4.0, n_samples), 2)
}

# Generate Target Variables
data["Current_GPA"] = (
    0.3 * data["Weekly_Study_Hours"] -
    0.1 * data["Absences"] +
    0.2 * data["Parental_Support"] +
    0.15 * data["Past_GPA"] +
    0.1 * (data["Sports"] + data["Music"] + data["Volunteering"]) +
    np.random.normal(0, 0.3, n_samples)
)
data["Current_GPA"] = np.clip(data["Current_GPA"], 0, 4)
data["At_Risk"] = (data["Current_GPA"] < 2.5).astype(int)

df = pd.DataFrame(data)

# Convert Categorical Columns
df = pd.get_dummies(df, columns=["Gender", "Ethnicity"], drop_first=True)

# Split Data
X = df.drop(columns=["Current_GPA", "At_Risk"])
y_gpa = df["Current_GPA"]
y_risk = df["At_Risk"]
X_train, X_test, y_train_gpa, y_test_gpa = train_test_split(X, y_gpa, test_size=0.2, random_state=42)
_, _, y_train_risk, y_test_risk = train_test_split(X, y_risk, test_size=0.2, random_state=42)

# Train Models
reg_model = LinearRegression()
reg_model.fit(X_train, y_train_gpa)

clf_model = LogisticRegression(max_iter=1000)
clf_model.fit(X_train, y_train_risk)

# Streamlit App
st.title("Student Performance Predictor")

# Sidebar Inputs
st.sidebar.header("Input Features")
age = st.sidebar.slider("Age", 13, 19, 16)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
ethnicity = st.sidebar.selectbox("Ethnicity", ["Caucasian", "African American", "Asian", "Other"])
parental_education = st.sidebar.selectbox("Parental Education", [1, 2, 3, 4])
parental_support = st.sidebar.slider("Parental Support (1-10)", 1, 10, 5)
study_hours = st.sidebar.slider("Weekly Study Hours", 1, 20, 10)
absences = st.sidebar.slider("Number of Absences", 0, 15, 3)
sports = st.sidebar.selectbox("Participates in Sports?", [0, 1])
music = st.sidebar.selectbox("Participates in Music?", [0, 1])
volunteering = st.sidebar.selectbox("Participates in Volunteering?", [0, 1])
past_gpa = st.sidebar.slider("Past GPA", 0.0, 4.0, 3.0)

# Prepare Input Data
input_data = {
    "Age": [age],
    "Parental_Education": [parental_education],
    "Parental_Support": [parental_support],
    "Weekly_Study_Hours": [study_hours],
    "Absences": [absences],
    "Sports": [sports],
    "Music": [music],
    "Volunteering": [volunteering],
    "Past_GPA": [past_gpa],
    "Gender_Male": [1 if gender == "Male" else 0],
    "Ethnicity_African American": [1 if ethnicity == "African American" else 0],
    "Ethnicity_Asian": [1 if ethnicity == "Asian" else 0],
    "Ethnicity_Other": [1 if ethnicity == "Other" else 0]
}

input_df = pd.DataFrame(input_data)

# Predictions
predicted_gpa = reg_model.predict(input_df)[0]
at_risk = clf_model.predict(input_df)[0]

# Display Results
st.subheader("Prediction Results")
st.write(f"Predicted GPA: **{predicted_gpa:.2f}**")
st.write(f"At Risk: **{'Yes' if at_risk == 1 else 'No'}**")

# Data Insights
#st.subheader("Data Insights")
#st.write("Correlation between features and GPA:")
#corr = df.corr()["Current_GPA"].drop("Current_GPA")
#st.bar_chart(corr)
# Data Insights Section
st.subheader("Data Insights")

# Correlation Analysis
st.write("Top Features Influencing GPA:")
correlations = df.corr()["Current_GPA"].drop("Current_GPA").sort_values(ascending=False)
st.bar_chart(correlations)

# GPA Distribution
st.write("Distribution of Current GPA:")
st.histogram(df["Current_GPA"], bins=20, title="GPA Distribution", color="blue")

# At-Risk Analysis
st.write("At-Risk Analysis:")
at_risk_count = df["At_Risk"].value_counts()
st.write(f"Total Students: {len(df)}")
st.write(f"At Risk: {at_risk_count[1]} ({at_risk_count[1] / len(df) * 100:.2f}%)")
st.write(f"Not At Risk: {at_risk_count[0]} ({at_risk_count[0] / len(df) * 100:.2f}%)")
st.bar_chart(at_risk_count)

# Feature Trends by Ethnicity
st.write("Feature Trends by Ethnicity:")
ethnicity_groups = df.groupby("Ethnicity").mean()[["Weekly_Study_Hours", "Current_GPA"]]
st.line_chart(ethnicity_groups)

# Advanced Insights
st.write("Advanced Insights:")
selected_feature = st.selectbox("Select a feature to analyze its relationship with GPA:",
                                ["Weekly_Study_Hours", "Absences", "Parental_Support", "Sports", "Music", "Volunteering"])
st.write(f"Average GPA based on {selected_feature}:")
avg_gpa_by_feature = df.groupby(selected_feature)["Current_GPA"].mean().sort_values()
st.bar_chart(avg_gpa_by_feature)



# In[ ]:



