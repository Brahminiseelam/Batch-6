
import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def load_models():
    model = joblib.load("xgboostairqualitymodel.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_models()
labels = ["Good", "Moderate", "Poor", "Hazardous"]

st.title("Air Quality Predictor")
st.write("Powered by XGBoost (93%+ accuracy)")

col1, col2, col3, col4, col5 = st.columns(5)
with col1: aqi = st.slider("AQI Value", 0, 500, 50)
with col2: co = st.slider("CO AQI", 0, 200, 70)
with col3: ozone = st.slider("Ozone AQI", 0, 200, 20)
with col4: no2 = st.slider("NO2 AQI", 0, 200, 30)
with col5: pm25 = st.slider("PM2.5 AQI", 0, 500, 26)

if st.button("Predict Air Quality"):
    data = np.array([aqi, co, ozone, no2, pm25]).reshape(1, -1)
    pred = model.predict(scaler.transform(data))[0]
    conf = model.predict_proba(scaler.transform(data)).max()
    st.success(f"{labels[pred]} ({conf:.1%} confidence)")
    st.balloons()
