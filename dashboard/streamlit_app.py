import streamlit as st
import pandas as pd
import joblib
from src.alerts import check_alert

st.set_page_config(page_title='AQI Dashboard - Sukkur', layout='wide')
st.title('Air Quality Index Prediction (Sukkur)')

model = joblib.load("model_registry/saved_models/aqi_model.pkl")
features = pd.read_csv("data/processed_data.csv")
input_features = features.drop(columns=["aqi"], errors='ignore').iloc[[-1]] # Use the latest features
st.dataframe(input_features)

if st.button("Predict AQI for Next 3 Days"):
    preds = model.predict(features.drop(columns=['aqi'], errors='ignore').tail(3))
    for idx, pred in enumerate(preds):
        st.metric(label=f"Day {idx+1} AQI", value=int(pred))
        st.info(check_alert(pred))

    st.write("Use EDA notebook for trends and more insights.")