import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder

# =====================================================================
# SYSTEM CONFIGURATION & CORAL THEME OVERRIDE
# =====================================================================
st.set_page_config(page_title="BPSO-SVM Weather Engine", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    /* Global Background - Inherited from Streamlit theme */
    .stApp { background-color: var(--background-color); }
    
    /* Clean Typography */
    h1, h2, h3, h4, h5, h6 { 
        color: var(--text-color); 
        font-family: 'Inter', sans-serif; 
        font-weight: 700; 
    }
    p { color: var(--text-color); opacity: 0.85; }
    
    /* Sleek Metric Cards */
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        border-radius: 8px;
        padding: 15px 20px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    div[data-testid="stMetricValue"] { color: #FF4B4B; font-weight: 800; }
    div[data-testid="stMetricLabel"] { color: var(--text-color) !important; opacity: 0.8; }
    
    /* Primary Action Button - Coral Red */
    div.stButton > button:first-child {
        background-color: #FF4B4B;
        color: white; border-radius: 6px; height: 55px; font-size: 18px;
        font-weight: 600; border: none; width: 100%; transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #FF3333;
        box-shadow: 0 4px 10px rgba(255, 75, 75, 0.3);
    }
    
    /* Sidebar Preset Buttons - Sleek Indigo Styling */
    section[data-testid="stSidebar"] div.stButton > button {
        background-color: #3B82F6 !important;
        color: white !important;
        border-radius: 6px !important;
        height: 40px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        border: none !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    section[data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #2563EB !important;
        box-shadow: 0 4px 10px rgba(59, 130, 246, 0.3) !important;
    }
    
    /* Custom Output Cards - Transparent to adapt to dark/light modes */
    .result-rain {
        background-color: rgba(239, 68, 68, 0.15); 
        border-left: 5px solid #EF4444;
        padding: 25px; 
        border-radius: 6px; 
        text-align: center; 
        color: var(--text-color);
    }
    .result-clear {
        background-color: rgba(16, 185, 129, 0.15); 
        border-left: 5px solid #10B981;
        padding: 25px; 
        border-radius: 6px; 
        text-align: center; 
        color: var(--text-color);
    }
    
    /* Evaluation Tables */
    .eval-table {
        width: 100%; 
        background-color: var(--secondary-background-color); 
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.2); 
        padding: 20px;
        color: var(--text-color);
    }
    .eval-table h4 {
        color: var(--text-color) !important;
        margin-top: 0;
    }
    .eval-table th {
        color: var(--text-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
    }
    .eval-table td {
        color: var(--text-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
    }

    /* Primary Header Banner */
    .main-header {
        background: linear-gradient(135deg, #FF4B4B 0%, #B91C1C 100%);
        padding: 35px 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 8px 20px -4px rgba(255, 75, 75, 0.4);
        margin-bottom: 30px;
    }
    .main-header h1 {
        color: #FFFFFF !important;
        margin: 0;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #FFE4E6 !important;
        font-size: 18px;
        margin-top: 8px;
        margin-bottom: 0;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# BACKGROUND GEOMETRY RECONSTRUCTION
# =====================================================================
@st.cache_data
def load_background_geometry():
    df = pd.read_csv("weatherAUS.csv")
    df = df.dropna(subset=['RainTomorrow'])
    df = df.drop(columns=['Sunshine', 'Evaporation', 'Cloud9am', 'Cloud3pm', 'Date'], errors='ignore')
    
    target_col = 'RainTomorrow'
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Robust isolation for Pandas 3.x
    categorical_cols = [col for col in df.columns if col not in numeric_cols and col != target_col]
    
    for col in numeric_cols: df[col] = df[col].fillna(df[col].median())
    
    label_encoders = {}
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        
    X = df.drop(columns=[target_col])
    scaler = StandardScaler().fit(X)
    return scaler, label_encoders, X.columns.tolist()

try:
    scaler, encoders, all_features = load_background_geometry()
    model = joblib.load("bpso_svm_weather_model.pkl")
    optimal_indices = joblib.load("optimal_feature_indices.pkl")
    optimized_features = [all_features[i] for i in optimal_indices]
except Exception as e:
    st.error(f"System Offline: Artifact missing. {e}")
    st.stop()

# =====================================================================
# SIDEBAR: PARAMETER CONTROL MATRIX
# =====================================================================
st.sidebar.markdown("<h2>Input Parameters</h2>", unsafe_allow_html=True)

# Initialize session state for input fields if not present
default_values = {
    'MinTemp': 15.0,
    'MaxTemp': 25.0,
    'Temp9am': 18.0,
    'WindGustSpeed': 40.0,
    'WindSpeed9am': 15.0,
    'WindSpeed3pm': 20.0,
    'Humidity9am': 60.0,
    'Humidity3pm': 50.0,
    'Pressure9am': 1015.0,
    'Pressure3pm': 1010.0,
    'RainToday': "No"
}
for key, val in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = val

for col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
    if col in encoders and col not in st.session_state:
        st.session_state[col] = encoders[col].classes_[0]

# Callback functions to update session state values for presets
def set_preset_rain():
    st.session_state['MinTemp'] = 12.0
    st.session_state['MaxTemp'] = 18.0
    st.session_state['Temp9am'] = 14.0
    st.session_state['WindGustSpeed'] = 60.0
    st.session_state['WindSpeed9am'] = 25.0
    st.session_state['WindSpeed3pm'] = 30.0
    st.session_state['Humidity9am'] = 85.0
    st.session_state['Humidity3pm'] = 90.0
    st.session_state['Pressure9am'] = 995.0
    st.session_state['Pressure3pm'] = 990.0
    st.session_state['RainToday'] = "Yes"

def set_preset_clear():
    st.session_state['MinTemp'] = 15.0
    st.session_state['MaxTemp'] = 27.0
    st.session_state['Temp9am'] = 18.0
    st.session_state['WindGustSpeed'] = 20.0
    st.session_state['WindSpeed9am'] = 10.0
    st.session_state['WindSpeed3pm'] = 12.0
    st.session_state['Humidity9am'] = 45.0
    st.session_state['Humidity3pm'] = 30.0
    st.session_state['Pressure9am'] = 1022.0
    st.session_state['Pressure3pm'] = 1020.0
    st.session_state['RainToday'] = "No"

# Render Quick Preset Buttons
st.sidebar.markdown("### Quick Presets")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.button("Rain Preset", on_click=set_preset_rain)
with col2:
    st.button("Clear Preset", on_click=set_preset_clear)

st.sidebar.markdown("<hr style='margin: 10px 0; border: 0; border-top: 1px solid #E6E6E9;'>", unsafe_allow_html=True)

input_data = {}

with st.sidebar.expander("Thermal Profile", expanded=True):
    input_data['MinTemp'] = st.number_input("Min Temp (°C)", value=st.session_state['MinTemp'], key='MinTemp')
    input_data['MaxTemp'] = st.number_input("Max Temp (°C)", value=st.session_state['MaxTemp'], key='MaxTemp')
    input_data['Temp9am'] = st.number_input("Temp at 9am (°C)", value=st.session_state['Temp9am'], key='Temp9am')

with st.sidebar.expander("Wind Dynamics", expanded=False):
    input_data['WindGustSpeed'] = st.number_input("Gust Speed (km/h)", value=st.session_state['WindGustSpeed'], key='WindGustSpeed')
    input_data['WindSpeed9am'] = st.number_input("Wind Speed 9am", value=st.session_state['WindSpeed9am'], key='WindSpeed9am')
    input_data['WindSpeed3pm'] = st.number_input("Wind Speed 3pm", value=st.session_state['WindSpeed3pm'], key='WindSpeed3pm')
    if 'WindGustDir' in optimized_features:
        gust_options = list(encoders['WindGustDir'].classes_)
        gust_index = gust_options.index(st.session_state['WindGustDir']) if st.session_state['WindGustDir'] in gust_options else 0
        input_data['WindGustDir'] = st.selectbox("Gust Direction", options=gust_options, index=gust_index, key='WindGustDir')
    if 'WindDir9am' in optimized_features:
        dir9_options = list(encoders['WindDir9am'].classes_)
        dir9_index = dir9_options.index(st.session_state['WindDir9am']) if st.session_state['WindDir9am'] in dir9_options else 0
        input_data['WindDir9am'] = st.selectbox("Wind Dir 9am", options=dir9_options, index=dir9_index, key='WindDir9am')
    if 'WindDir3pm' in optimized_features:
        dir3_options = list(encoders['WindDir3pm'].classes_)
        dir3_index = dir3_options.index(st.session_state['WindDir3pm']) if st.session_state['WindDir3pm'] in dir3_options else 0
        input_data['WindDir3pm'] = st.selectbox("Wind Dir 3pm", options=dir3_options, index=dir3_index, key='WindDir3pm')

with st.sidebar.expander("Moisture & Pressure", expanded=False):
    input_data['Humidity9am'] = st.number_input("Humidity 9am (%)", value=st.session_state['Humidity9am'], key='Humidity9am')
    input_data['Humidity3pm'] = st.number_input("Humidity 3pm (%)", value=st.session_state['Humidity3pm'], key='Humidity3pm')
    input_data['Pressure9am'] = st.number_input("Pressure 9am (hPa)", value=st.session_state['Pressure9am'], key='Pressure9am')
    input_data['Pressure3pm'] = st.number_input("Pressure 3pm (hPa)", value=st.session_state['Pressure3pm'], key='Pressure3pm')
    if 'RainToday' in optimized_features:
        rt_options = ["No", "Yes"]
        rt_index = rt_options.index(st.session_state['RainToday']) if st.session_state['RainToday'] in rt_options else 0
        input_data['RainToday'] = st.selectbox("Rain Today?", options=rt_options, index=rt_index, key='RainToday')

# =====================================================================
# MAIN DASHBOARD: THE CLEAN STAGE
# =====================================================================
st.markdown("""
    <div class="main-header">
        <h1>Atmospheric Early Warning System</h1>
        <p>Powered by Hybrid Binary Particle Swarm Optimization & Support Vector Machine</p>
    </div>
""", unsafe_allow_html=True)

# Metrics display
m1, m2, m3, m4 = st.columns(4)
with m1: st.metric(label="System Accuracy", value="79.68%")
with m2: st.metric(label="Swarm Fitness", value="0.5147 MCC")
with m3: st.metric(label="Rainfall Recall", value="78.00%")
with m4: st.metric(label="Feature Space", value="14 Active")

st.markdown("<br>", unsafe_allow_html=True)

# The Execution Stage
_, center_col, _ = st.columns([1, 2, 1])

with center_col:
    trigger = st.button("EXECUTE BPSO-SVM PREDICTION")
    
    if trigger:
        with st.spinner('Aligning vector geometry...'):
            time.sleep(1.0) 
            
            raw_input = pd.DataFrame(columns=all_features)
            raw_input.loc[0] = 0 
            
            for feature in optimized_features:
                val = input_data[feature]
                if feature in encoders: val = encoders[feature].transform([str(val)])[0]
                elif feature == 'RainToday': val = 1 if val == "Yes" else 0
                raw_input.at[0, feature] = val
                
            scaled_input_matrix = scaler.transform(raw_input)
            final_vector = scaled_input_matrix[:, optimal_indices]
            prediction = model.predict(final_vector)
            
        if prediction[0] == 1:
            st.markdown("""
                <div class="result-rain">
                    <h2 style='color:#EF4444; margin:0;'>RAINFALL DETECTED</h2>
                    <p style='margin-top:10px;'>The maximum-margin classifier indicates a high probability of precipitation.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="result-clear">
                    <h2 style='color:#10B981; margin:0;'>CLEAR SKIES</h2>
                    <p style='margin-top:10px;'>Atmospheric conditions remain stable. No significant rainfall expected.</p>
                </div>
            """, unsafe_allow_html=True)

st.markdown("<br><hr style='border:1px solid #E6E6E9;'><br>", unsafe_allow_html=True)

# =====================================================================
# SVM DECISION BOUNDARY LOGIC (TABBED ARCHITECTURE)
# =====================================================================
st.markdown("<h3>Support Vector Hyperplane Logic</h3>", unsafe_allow_html=True)
st.info("The Support Vector Machine (SVM) evaluates the non-linear boundaries of your 14 BPSO-optimized predictors. Below are the specific atmospheric combinations that shift the geometric prediction.")

# Implement clean tabbed layout to avoid clustering
tab_rain, tab_clear = st.tabs(["Convective Instability (Rain Drivers)", "Anti-Cyclonic Stability (Clear Drivers)"])

with tab_rain:
    st.markdown("""
    <div style='color: var(--text-color); opacity: 0.95; padding: 15px 5px;'>
    The SVM vectors map heavily toward <b>Rain</b> when the following mathematical thresholds are breached:
    <ul style='margin-top: 10px;'>
        <li><b>Moisture Saturation:</b> <code>Humidity3pm</code> and <code>Humidity9am</code> climb sharply (typically > <b>75%</b>).</li>
        <li><b>Barometric Depression:</b> <code>Pressure3pm</code> and <code>Pressure9am</code> drop significantly (< <b>1005 hPa</b>), indicating an incoming low-pressure trough.</li>
        <li><b>Active Turbulence:</b> <code>WindGustSpeed</code> is elevated, showing active aerodynamic shear and storm front movement.</li>
        <li><b>Thermal Compression:</b> The margin between <code>MinTemp</code> and <code>MaxTemp</code> narrows.</li>
        <li><b>Climatic Persistence:</b> <code>RainToday</code> is <b>Yes</b>, proving an active moisture cell is already anchored.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
with tab_clear:
    st.markdown("""
    <div style='color: var(--text-color); opacity: 0.95; padding: 15px 5px;'>
    The maximum-margin classifier secures a <b>No Rain</b> output when it detects atmospheric suppression:
    <ul style='margin-top: 10px;'>
        <li><b>Dry Air Masses:</b> Afternoon humidity levels remain moderate to low (< <b>50%</b>).</li>
        <li><b>High Atmospheric Pressure:</b> Barometric readings remain high and stable (> <b>1015 hPa</b>).</li>
        <li><b>Stable Airflow:</b> Wind speeds and gusts remain at normal baseline levels without erratic spikes.</li>
        <li><b>Broad Thermal Range:</b> Normal expected gaps between morning and afternoon temperatures.</li>
        <li><b>Climatic Persistence:</b> <code>RainToday</code> is <b>No</b>.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
    <div style='color: var(--text-color); background-color: var(--secondary-background-color); border: 1px solid rgba(128, 128, 128, 0.2); padding: 15px; border-left: 4px solid #FF4B4B; border-radius: 4px;'>
    <b>Panel Demonstration:</b> Open the sidebar, set <code>RainToday</code> to <b>Yes</b>, push <code>Humidity3pm</code> to <b>90%</b>, drop <code>Pressure3pm</code> to <b>990 hPa</b>, and raise <code>WindGustSpeed</code> to <b>60 km/h</b>. Execute the engine and observe how the hyperplane conclusively maps to Rainfall.
    </div>
""", unsafe_allow_html=True)

st.markdown("<br><hr style='border:1px solid #E6E6E9;'><br>", unsafe_allow_html=True)

# =====================================================================
# BOTTOM SECTION: ENGINE ANALYTICS
# =====================================================================
st.markdown("<h3>Model Evaluation Architecture</h3>", unsafe_allow_html=True)
st.markdown("<p>Historical performance metrics derived from the 42,658-row holdout testing matrix.</p>", unsafe_allow_html=True)

col_cm, col_cr = st.columns(2)

with col_cm:
    st.markdown("""
    <div class="eval-table">
        <h4 style="margin-top:0;">Confusion Matrix</h4>
        <table style="width:100%; text-align:center; border-collapse: collapse;">
            <tr style="background-color: rgba(128, 128, 128, 0.08);">
                <th style="padding:10px; border:1px solid rgba(128, 128, 128, 0.2);"></th>
                <th style="padding:10px; border:1px solid rgba(128, 128, 128, 0.2);">Predicted Dry</th>
                <th style="padding:10px; border:1px solid rgba(128, 128, 128, 0.2);">Predicted Rain</th>
            </tr>
            <tr>
                <td style="padding:10px; border:1px solid rgba(128, 128, 128, 0.2); font-weight:bold; background-color: rgba(128, 128, 128, 0.08);">Actual Dry</td>
                <td style="padding:10px; border:1px solid rgba(128, 128, 128, 0.2); color:#10B981; font-weight:bold;">26,570</td>
                <td style="padding:10px; border:1px solid rgba(128, 128, 128, 0.2); color:#EF4444; font-weight:bold;">6,525</td>
            </tr>
            <tr>
                <td style="padding:10px; border:1px solid rgba(128, 128, 128, 0.2); font-weight:bold; background-color: rgba(128, 128, 128, 0.08);">Actual Rain</td>
                <td style="padding:10px; border:1px solid rgba(128, 128, 128, 0.2); color:#EF4444; font-weight:bold;">2,141</td>
                <td style="padding:10px; border:1px solid rgba(128, 128, 128, 0.2); color:#10B981; font-weight:bold;">7,422</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

with col_cr:
    st.markdown("""
    <div class="eval-table">
        <h4 style="margin-top:0;">Classification Report</h4>
        <table style="width:100%; text-align:left; border-collapse: collapse;">
            <tr style="background-color: rgba(128, 128, 128, 0.08); border-bottom: 2px solid rgba(128, 128, 128, 0.3);">
                <th style="padding:10px;">Class</th>
                <th style="padding:10px;">Precision</th>
                <th style="padding:10px;">Recall</th>
                <th style="padding:10px;">F1-Score</th>
            </tr>
            <tr style="border-bottom: 1px solid rgba(128, 128, 128, 0.2);">
                <td style="padding:10px; font-weight:bold;">No Rain (0)</td>
                <td style="padding:10px;">0.93</td>
                <td style="padding:10px;">0.80</td>
                <td style="padding:10px;">0.86</td>
            </tr>
            <tr>
                <td style="padding:10px; font-weight:bold;">Rain (1)</td>
                <td style="padding:10px;">0.53</td>
                <td style="padding:10px; color:#FF4B4B; font-weight:bold;">0.78</td>
                <td style="padding:10px;">0.63</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)