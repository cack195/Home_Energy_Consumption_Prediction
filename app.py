import streamlit as st
import pandas as pd
import numpy as np
import boto3
import joblib
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Get AWS details from environment variables
bucket_name = os.getenv('AWS_BUCKET_NAME')
model_key = os.getenv('MODEL_KEY')

def download_model_from_s3(bucket, key, download_path):
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, download_path)
    
# Comment out the AWS code if your are not using it if you are loading the model locally

model_local_path = 'My_model.pkl'

if not os.path.exists(model_local_path):
    st.write("Downloading the model from AWS S3...")
    download_model_from_s3(bucket_name, model_key, model_local_path)

# Load the trained model from the local path
ensemble_model = joblib.load(model_local_path)

# Function to create sine and cosine features
def create_features(hour, day, month, year):
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day / 7)
    day_cos = np.cos(2 * np.pi * day / 7)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    return [hour, year, day, month, hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos]

def max_days_in_month(year, month):
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    elif month in [4, 6, 9, 11]:
        return 30
    elif month == 2:
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            return 29
        else:
            return 28
    return 31

st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
    }
    .stApp {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    .block-container {
        width: 50%;
        text-align: center;
    }
    select {
        background-color: #333;
        color: white;
        border-radius: 5px;
        padding: 10px;
    }
    input {
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Home Energy Consumption Prediction (Location: Bahrain)")

year = st.selectbox("Year", options=[2021, 2022, 2023, 2024])
month = st.number_input("Month (1-12)", min_value=1, max_value=12)
max_days = max_days_in_month(year, month)
day = st.number_input(f"Day (1-{max_days})", min_value=1, max_value=max_days)
hour = st.number_input("Hour (0-23)", min_value=0, max_value=23)
actual_consumption = st.number_input("Actual Energy Consumption (kWh)", min_value=0.0)

if st.button("Predict"):
    features = create_features(hour, day, month, year)
    input_data = pd.DataFrame([features], columns=[
        'Hour', 'Year', 'Day', 'Month', 
        'HourSin', 'HourCos', 'Day_Sin', 'Day_Cos', 
        'MonthSin', 'MonthCos'
    ])
    
    predicted_consumption = ensemble_model.predict(input_data)[0]
    difference = actual_consumption - predicted_consumption
    percentage_difference = (difference / predicted_consumption) * 100 if predicted_consumption != 0 else 0

    if difference > 0:
        st.write(f"The actual energy consumption is above average by {percentage_difference:.2f}%.")
    elif difference < 0:
        st.write(f"The actual energy consumption is below average by {-percentage_difference:.2f}%.")
    else:
        st.write("The actual energy consumption is equal to the average.")
    
    st.write(f"Predicted Energy Consumption: {predicted_consumption:.2f} kWh")