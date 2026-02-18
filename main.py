import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 1. DATABASE CONFIGURATION
# ==========================================
# Link: https://biochamber-52607-default-rtdb.firebaseio.com/
FIREBASE_URL = "https://biochamber-52607-default-rtdb.firebaseio.com/"

# ==========================================
# 2. ML MODEL (Trained for Deviation)
# ==========================================
@st.cache_resource
def train_biomodel():
    # Synthetic dataset training on the specific parameter names
    np.random.seed(42)
    data = []
    for _ in range(1000):
        t = np.random.uniform(20, 45)
        p = np.random.uniform(4, 9)
        do = np.random.uniform(0, 100)
        od = np.random.uniform(0, 2)
        
        # Growth Rate peaks at 37C and 7pH
        dist = np.sqrt((t - 37)**2 + ((p - 7)*5)**2) 
        growth = max(0, 1.0 - (dist / 20))
        data.append([t, p, do, od, growth])
    
    df = pd.DataFrame(data, columns=['temperature', 'ph', 'dissolved_oxygen', 'optical_density', 'growth'])
    model = RandomForestRegressor(n_estimators=50)
    model.fit(df[['temperature', 'ph', 'dissolved_oxygen', 'optical_density']], df['growth'])
    return model

ai_model = train_biomodel()

# ==========================================
# 3. DASHBOARD UI SETUP
# ==========================================
st.set_page_config(page_title="BioChamber LIVE", layout="wide")
st.title("🌿 BioChamber: AI Monitoring & Control")

# Sidebar for setting user optimums
st.sidebar.header("Target Parameters")
target_temp = st.sidebar.number_input("Target Temp (°C)", value=37.0)
target_ph = st.sidebar.number_input("Target pH", value=7.0)

placeholder = st.empty()

# ==========================================
# 4. ACTIVE MONITORING LOOP
# ==========================================
while True:
    try:
        # PULL FROM 'live_readings' NODE
        response = requests.get(f"{FIREBASE_URL}/live_readings.json")
        sensor_data = response.json()
        
        if sensor_data:
            # EXTRACT USING YOUR NEW KEYS
            temp = float(sensor_data.get('temperature', 0))
            ph = float(sensor_data.get('ph', 0))
            do = float(sensor_data.get('dissolved_oxygen', 0))
            od = float(sensor_data.get('optical_density', 0))

            # AI Prediction
            current_state = np.array([[temp, ph, do, od]])
            growth_eff = ai_model.predict(current_state)[0]
            deviation = 1.0 - growth_eff

            # Generate Directions
            directions = {
                "temp_instruction": "STABLE",
                "ph_instruction": "STABLE",
                "growth_efficiency": round(growth_eff, 2),
                "last_update": time.ctime()
            }

            if temp < target_temp - 0.5: directions["temp_instruction"] = "HEAT_ON"
            elif temp > target_temp + 0.5: directions["temp_instruction"] = "COOLING_ON"

            if ph < target_ph - 0.2: directions["ph_instruction"] = "ADD_BASE"
            elif ph > target_ph + 0.2: directions["ph_instruction"] = "ADD_ACID"

            # PUSH BACK TO 'control' NODE
            requests.patch(f"{FIREBASE_URL}/control.json", data=json.dumps(directions))

            # UI Update
            with placeholder.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Temperature", f"{temp} °C", f"{round(temp-target_temp, 1)} dev")
                c2.metric("pH Level", ph, f"{round(ph-target_ph, 1)} dev")
                c3.metric("Growth (AI)", f"{round(growth_eff*100, 1)}%")
                c4.metric("OD (Biomass)", od)

                st.divider()
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("🤖 AI Analysis")
                    st.progress(float(growth_eff))
                    if deviation > 0.2:
                        st.warning(f"Deviation Detected! Efficiency: {round(growth_eff*100)}%")
                    else:
                        st.success("Growth environment is optimal.")

                with col_b:
                    st.subheader("📡 IoT Control Directions")
                    st.info(f"**Heating/Cooling:** {directions['temp_instruction']}")
                    st.info(f"**pH Pump:** {directions['ph_instruction']}")
                    st.caption(f"Last updated: {directions['last_update']}")
        else:
            st.warning("Connected to database, but '/live_readings' node is empty.")

    except Exception as e:
        st.error(f"Sync Error: {e}")

    # Wait 5 seconds before next refresh
    time.sleep(5)
    st.rerun()
