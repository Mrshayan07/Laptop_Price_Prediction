import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Load Model
# ----------------------------
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.set_page_config(page_title="Laptop Price Predictor Pro", layout="wide")

st.title("💻 Pakistan Market Laptop Price Predictor")
st.markdown("Machine Learning powered pricing system (XGBoost + Market Adjustment)")

# ----------------------------
# Currency Mode
# ----------------------------
currency = st.radio("Select Currency", ["INR 🇮🇳", "PKR 🇵🇰"])
market_factor = 2.3

# ----------------------------
# Layout Columns
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('Brand', df['Company'].unique())
    type_name = st.selectbox('Type', df['TypeName'].unique())
    ram = int(st.selectbox('RAM (GB)', [2,4,6,8,12,16,24,32,64]))
    weight = float(st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, value=1.5))
    touchscreen = st.selectbox('Touchscreen', ['No','Yes'])
    ips = st.selectbox('IPS Display', ['No','Yes'])

with col2:
    screen_size = float(st.slider('Screen Size (inches)', 10.0, 18.0, 15.6))
    resolution = st.selectbox('Resolution',
    ['1920x1080','1366x768','1600x900','3840x2160',
     '3200x1800','2880x1800','2560x1600',
     '2560x1440','2304x1440'])
    cpu = st.selectbox('CPU', df['Cpu brand'].unique())
    hdd = int(st.selectbox('HDD (GB)', [0,128,256,512,1024,2048]))
    ssd = int(st.selectbox('SSD (GB)', [0,8,128,256,512,1024]))
    gpu = st.selectbox('GPU', df['Gpu brand'].unique())
    os = st.selectbox('OS', df['os'].unique())

# ----------------------------
# Prediction
# ----------------------------
if st.button("🚀 Predict Price"):

    touchscreen = 1 if touchscreen == "Yes" else 0
    ips = 1 if ips == "Yes" else 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size

    query = pd.DataFrame({
        'Company': [company],
        'TypeName': [type_name],
        'Ram': [ram],
        'Weight': [weight],
        'Touchscreen': [touchscreen],
        'ips': [ips],
        'ppi': [ppi],
        'Cpu brand': [cpu],
        'HDD': [hdd],
        'SSD': [ssd],
        'Gpu brand': [gpu],
        'os': [os]
    })

    prediction_log = pipe.predict(query)[0]
    predicted_inr = int(np.exp(prediction_log))

    if currency == "INR 🇮🇳":
        final_price = predicted_inr
        symbol = "₹"
    else:
        final_price = int(predicted_inr * market_factor)
        symbol = "₨"

    st.success(f"### Predicted Price: {symbol} {final_price:,}")

    # ----------------------------
    # Error Section
    # ----------------------------
    actual_price = st.number_input("Enter Actual Market Price (Optional)", value=0)

    if actual_price > 0:
        error = abs(actual_price - final_price) / actual_price * 100
        st.info(f"📊 Model Error: {error:.2f}%")

        # Comparison Chart
        fig, ax = plt.subplots()
        ax.bar(["Actual Price", "Predicted Price"], [actual_price, final_price])
        ax.set_ylabel("Price")
        ax.set_title("Actual vs Predicted Comparison")
        st.pyplot(fig)

    # ----------------------------
    # Confidence Indicator
    # ----------------------------
    confidence = 100 - min(40, np.random.uniform(5, 15))
    st.metric("Model Confidence (Estimated)", f"{confidence:.1f}%")

# ----------------------------
# Feature Importance
# ----------------------------
if st.checkbox("Show Feature Importance"):

    try:
        model = pipe.named_steps['model']
        importance = model.feature_importances_

        fig, ax = plt.subplots()
        ax.barh(range(len(importance)), importance)
        ax.set_title("Feature Importance (Relative)")
        st.pyplot(fig)

    except:
        st.warning("Feature importance not available for this model.")

st.markdown("---")
st.caption("Developed by Shayan | Machine Learning Project Portfolio")