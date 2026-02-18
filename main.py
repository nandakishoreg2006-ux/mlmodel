import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time
from sklearn.ensemble import RandomForestRegressor

# 1. DATABASE LINK
FIREBASE_URL = "https://biochamber-52607-default-rtdb.firebaseio.com/"

# 2. ML MODEL (Cached so it doesn't re-train every refresh)
@st.cache_resource
def train_biomodel():
    np.random.seed(42)
    data = []
    for _ in range(1000):
        t = np.random.uniform(20, 45)
        p = np.random.uniform(4, 9)
        do = np.random.uniform(0, 100)
        od = np.random.uniform(0, 2)
        dist = np.sqrt((t - 37)**2 + ((p - 7)*5)**2) 
        growth = max(0, 1.0 - (dist / 20))
        data.append([t, p, do, od, growth])
    
    df = pd.DataFrame(data, columns=['t', 'p', 'do', 'od', 'growth'])
    model = RandomForestRegressor(n_estimators=50)
    model.fit(df[['t', 'p', 'do', 'od']], df['growth'])
    return model

ai_model = train_biomodel()

# 3. UI SETUP
st.set_page_config(page_title="BioChamber LIVE", layout="wide")
st.title("🌿 BioChamber: AI Monitoring & Control")

# Sidebar
target_temp = st.sidebar.number_input("Target Temp (°C)", value=37.0)
target_ph = st.sidebar.number_input("Target pH", value=7.0)
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 2, 10, 5)

# Container for live data
placeholder = st.empty()

# 4. MONITORING LOGIC
try:
    # Fetch real data from Firebase
    response = requests.get(f"{FIREBASE_URL}/sensors.json")
    
    # Debug: Show raw data if it fails
    if response.status_code != 200:
        st.error(f"Firebase Error: {response.status_code}")
    
    sensor_data = response.json()
    
    if sensor_data:
        # Get values
        temp = float(sensor_data.get('temp', 0))
        ph = float(sensor_data.get('ph', 0))
        do = float(sensor_data.get('do', 0))
        od = float(sensor_data.get('od', 0))

        # AI Prediction
        current_state = np.array([[temp, ph, do, od]])
        growth_eff = ai_model.predict(current_state)[0]

        # Control Logic
        directions = {"temp_instruction": "STABLE", "ph_instruction": "STABLE"}
        if temp < target_temp - 0.5: directions["temp_instruction"] = "HEAT_ON"
        elif temp > target_temp + 0.5: directions["temp_instruction"] = "COOLING_ON"
        if ph < target_ph - 0.2: directions["ph_instruction"] = "ADD_BASE"
        elif ph > target_ph + 0.2: directions["ph_instruction"] = "ADD_ACID"

        # Update Firebase
        requests.patch(f"{FIREBASE_URL}/control.json", data=json.dumps(directions))

        # Display Metrics
        with placeholder.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Temperature", f"{temp} °C")
            c2.metric("pH Level", ph)
            c3.metric("Growth (AI)", f"{round(growth_eff*100, 1)}%")
            c4.metric("OD", od)
            
            st.divider()
            st.subheader("📡 AI Directions Sent to IoT")
            st.info(f"Thermal: {directions['temp_instruction']} | pH: {directions['ph_instruction']}")

    else:
        st.warning("Database connected, but no data found in '/sensors'. Check your IoT device.")

except Exception as e:
    st.error(f"Application Error: {e}")

# Trigger auto-refresh
time.sleep(refresh_rate)
st.rerun()
