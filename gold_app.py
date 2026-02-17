
import streamlit as st
import joblib
import numpy as np
import os
try:
    model_path = os.path.join(os.path.dirname(__file__), "gold_model.pkl")
    scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    st.error("Model file 'gold_model.pkl' not found. Please ensure it is in the same folder as this app.")
    st.stop()
st.title("🪙 Gold Price Prediction App")
st.write("This app predicts gold prices based on input market parameters.")
st.subheader("Enter the feature values:")
spx = st.number_input("SPX (S&P 500 Index)", value=0.0)
uso = st.number_input("USO (Crude Oil ETF)", value=0.0)
slv = st.number_input("SLV (Silver ETF)", value=0.0)
eurusd = st.number_input("EUR/USD Exchange Rate", value=0.0)
if st.button("🔮 Predict Gold Price"):
    try:
        features = np.array([[spx, uso, slv, eurusd]])
        prediction = model.predict(scaler.transform(features))

        st.success(f"🏆 Predicted Gold Price: ₹{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")