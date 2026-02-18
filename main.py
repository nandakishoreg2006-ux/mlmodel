import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 1. DATABASE & ML CONFIGURATION
# ==========================================
FIREBASE_URL = "https://biochamber-52607-default-rtdb.firebaseio.com/"

@st.cache_resource
def train_control_model():
    # Training a model to understand "Growth Efficiency"
    # Inputs: [temp, ph, oxygen, od] -> Output: Growth Score (0 to 1)
    np.random.seed(42)
    X = []
    y = []
    for _ in range(1200):
        t = np.random.uniform(15, 50)
        p = np.random.uniform(3, 10)
        do = np.random.uniform(0, 100)
        od = np.random.uniform(0, 5)
        
        # Logic: Efficiency drops if any parameter deviates from E.coli optima (37C, 7pH, 40% DO)
        t_score = np.exp(-((t - 37)**2) / 50)
        p_score = np.exp(-((p - 7)**2) / 2)
        do_score = np.exp(-((do - 40)**2) / 1000)
        efficiency = (t_score * p_score * do_score)
        
        X.append([t, p, do, od])
        y.append(efficiency)
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model

ai_brain = train_control_model()

# ==========================================
# 2. UI LAYOUT & SETTINGS
# ==========================================
st.set_page_config(page_title="BioChamber AI", layout="wide")
st.title("🧪 BioChamber: Microbial Intelligence System")

# Sidebar: Selection of Microbe Profile
st.sidebar.header("Microbe Profile")
microbe = st.sidebar.selectbox("Current Microorganism", ["E. coli", "S. cerevisiae (Yeast)", "Custom"])

if microbe == "E. coli":
    ideal_t, ideal_p, ideal_do = 37.0, 7.0, 40.0
elif microbe == "S. cerevisiae (Yeast)":
    ideal_t, ideal_p, ideal_do = 30.0, 5.0, 20.0
else:
    ideal_t = st.sidebar.number_input("Custom Ideal Temp", value=37.0)
    ideal_p = st.sidebar.number_input("Custom Ideal pH", value=7.0)
    ideal_do = st.sidebar.number_input("Custom Ideal DO (%)", value=40.0)

# ==========================================
# 3. DYNAMIC FRAGMENT (Two Sections)
# ==========================================
@st.fragment(run_every=5)
def process_bioreactor():
    try:
        # PULL FROM 'live_readings'
        response = requests.get(f"{FIREBASE_URL}/live_readings.json")
        data = response.json()
        
        if data:
            # MAP TO YOUR DATABASE KEYS
            cur_t = float(data.get('temperature', 0))
            cur_p = float(data.get('ph', 0))
            cur_do = float(data.get('dissolved_oxygen', 0))
            cur_od = float(data.get('optical_density', 0))

            # AI: Predict Growth Efficiency
            current_features = np.array([[cur_t, cur_p, cur_do, cur_od]])
            efficiency = ai_brain.predict(current_features)[0]

            # AI: Determine Ideal Directions (Control Logic)
            actions = {"thermal": "STABLE", "ph_pump": "STABLE", "oxygen_flow": "STABLE"}
            
            # Temperature Control
            if cur_t < ideal_t - 0.5: actions["thermal"] = "HEAT_ON"
            elif cur_t > ideal_t + 0.5: actions["thermal"] = "COOLING_ON"
            
            # pH Control
            if cur_p < ideal_p - 0.2: actions["ph_pump"] = "ADD_BASE"
            elif cur_p > ideal_p + 0.2: actions["ph_pump"] = "ADD_ACID"
            
            # Dissolved Oxygen Control
            if cur_do < ideal_do - 5: actions["oxygen_flow"] = "INCREASE_AERATION"
            elif cur_do > ideal_do + 10: actions["oxygen_flow"] = "DECREASE_AERATION"

            # SYNC BACK TO FIREBASE
            control_payload = {
                "commands": actions,
                "ai_growth_score": round(efficiency, 3),
                "timestamp": time.strftime("%H:%M:%S")
            }
            requests.patch(f"{FIREBASE_URL}/control.json", data=json.dumps(control_payload))

            # --- SECTION 1: ACTIVE SENSOR DATA ---
            st.header("📊 Section 1: Active Sensor Data")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Temperature", f"{cur_t}°C")
            col2.metric("pH Level", f"{cur_p}")
            col3.metric("Dissolved Oxygen", f"{cur_do}%")
            col4.metric("Optical Density", f"{cur_od}")
            
            # --- SECTION 2: MAINTAINING & AI CONTROL ---
            st.divider()
            st.header("🤖 Section 2: AI Maintenance & Control")
            
            left, right = st.columns(2)
            with left:
                st.subheader("Microbe Health")
                st.write(f"**Current Efficiency:** {round(efficiency * 100, 1)}%")
                st.progress(float(efficiency))
                if efficiency > 0.85:
                    st.success("Environment is Perfectly Maintained.")
                else:
                    st.warning("AI is actively adjusting parameters to reach optimum.")

            with right:
                st.subheader("IoT Directives")
                st.write(f"🌡️ **Thermal:** `{actions['thermal']}`")
                st.write(f"🧪 **pH Regulator:** `{actions['ph_pump']}`")
                st.write(f"💨 **Aeration:** `{actions['oxygen_flow']}`")
                st.caption(f"Last AI Revision: {control_payload['timestamp']}")

        else:
            st.error("⚠️ No data in 'live_readings'. Check IoT connection.")

    except Exception as e:
        st.error(f"Communication Error: {e}")

# Run the system
process_bioreactor()
