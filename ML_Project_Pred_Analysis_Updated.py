#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

# Generate Synthetic Data
np.random.seed(42)
n_samples = 1000
data = {
    "Age": np.random.randint(15, 25, n_samples),
    "Gender": np.random.choice(["Male", "Female"], n_samples),
    "Ethnicity": np.random.choice(["Caucasian", "African American", "Asian", "Other"], n_samples),
    "Parental_Education": np.random.choice(["None", "High School", "College", "Bachelor's", "Higher Education"], n_samples),
    "Parental_Support": np.random.choice(["None", "Low", "Moderate", "High", "Very High"], n_samples),
    "Weekly_Study_Hours": np.random.randint(1, 20, n_samples),
    "Absences": np.random.randint(0, 15, n_samples),
    "Sports": np.random.choice([0, 1], n_samples),
    "Music": np.random.choice([0, 1], n_samples),
    "Volunteering": np.random.choice([0, 1], n_samples),
    "Past_GPA": np.round(np.random.uniform(2.0, 4.0, n_samples), 2)
}

# Generate Target Variables
data["Current_GPA"] = (
    0.3 * np.array(data["Weekly_Study_Hours"]) -
    0.1 * np.array(data["Absences"]) +
    0.15 * np.array(data["Past_GPA"]) +
    0.1 * (np.array(data["Sports"]) + np.array(data["Music"]) + np.array(data["Volunteering"])) +
    np.random.normal(0, 0.3, n_samples)
)
data["Current_GPA"] = np.clip(data["Current_GPA"], 0, 4)
data["At_Risk"] = (data["Current_GPA"] < 2.5).astype(int)

df = pd.DataFrame(data)

# Convert Categorical Columns
df = pd.get_dummies(df, columns=["Gender", "Ethnicity", "Parental_Education", "Parental_Support"], drop_first=True)

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
st.sidebar.header("Interactive Features")
age = st.sidebar.slider("Age", 15, 25, 20)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
ethnicity = st.sidebar.selectbox("Ethnicity", ["Caucasian", "African American", "Asian", "Other"])
parental_education = st.sidebar.selectbox("Parental Education", ["None", "High School", "College", "Bachelor's", "Higher Education"])
parental_support = st.sidebar.selectbox("Parental Support", ["None", "Low", "Moderate", "High", "Very High"])
study_hours = st.sidebar.slider("Weekly Study Hours", 1, 20, 10)
absences = st.sidebar.slider("Number of Absences", 0, 15, 3)
sports = st.sidebar.selectbox("Participates in Sports?", [0, 1])
music = st.sidebar.selectbox("Participates in Music?", [0, 1])
volunteering = st.sidebar.selectbox("Participates in Volunteering?", [0, 1])
past_gpa = st.sidebar.slider("Past GPA", 0.0, 4.0, 3.0)

# Prepare Input Data
input_data = {
    "Age": [age],
    "Weekly_Study_Hours": [study_hours],
    "Absences": [absences],
    "Sports": [sports],
    "Music": [music],
    "Volunteering": [volunteering],
    "Past_GPA": [past_gpa]
}

# Handle Dummy Variables
categorical_features = {
    "Gender_Male": 1 if gender == "Male" else 0,
    "Ethnicity_African American": 1 if ethnicity == "African American" else 0,
    "Ethnicity_Asian": 1 if ethnicity == "Asian" else 0,
    "Ethnicity_Other": 1 if ethnicity == "Other" else 0,
    "Parental_Education_High School": 1 if parental_education == "High School" else 0,
    "Parental_Education_College": 1 if parental_education == "College" else 0,
    "Parental_Education_Bachelor's": 1 if parental_education == "Bachelor's" else 0,
    "Parental_Education_Higher Education": 1 if parental_education == "Higher Education" else 0,
    "Parental_Support_Low": 1 if parental_support == "Low" else 0,
    "Parental_Support_Moderate": 1 if parental_support == "Moderate" else 0,
    "Parental_Support_High": 1 if parental_support == "High" else 0,
    "Parental_Support_Very High": 1 if parental_support == "Very High" else 0
}
input_data.update(categorical_features)
input_df = pd.DataFrame(input_data)

# Align with Training Features
missing_cols = set(X_train.columns) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0
input_df = input_df[X_train.columns]

# Predictions
predicted_gpa = reg_model.predict(input_df)[0]
at_risk = clf_model.predict(input_df)[0]

# Display Results
st.subheader("Prediction Results")
st.write(f"Predicted GPA: **{predicted_gpa:.2f}**")
st.write(f"At Risk: **{'Yes' if at_risk == 1 else 'No'}**")

# Data Insights Section
st.subheader("Data Insights")
st.write("Top Features Influencing GPA:")
correlations = df.corr()["Current_GPA"].drop("Current_GPA").sort_values(ascending=False)
st.bar_chart(correlations)

# GPA Distribution using Matplotlib
st.write("Distribution of Current GPA:")
fig, ax = plt.subplots()
ax.hist(df["Current_GPA"], bins=20, color="blue", alpha=0.7)
ax.set_title("GPA Distribution")
ax.set_xlabel("GPA")
ax.set_ylabel("Frequency")
st.pyplot(fig)  # Display the plot in Streamlit




