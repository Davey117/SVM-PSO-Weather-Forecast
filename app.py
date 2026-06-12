import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder

# =====================================================================
# SYSTEM CONFIGURATION & CORAL THEME OVERRIDE
# =====================================================================
st.set_page_config(page_title="BPSO-SVM Weather Engine", page_icon="🌤️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    /* Global Background - Soft Gray matching your reference */
    .stApp { background-color: #F4F6F9; }
    
    /* Clean Typography */
    h1, h2, h3 { color: #262730; font-family: 'Inter', sans-serif; font-weight: 700; }
    p { color: #5C5C6A; }
    
    /* Sleek Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 15px 20px;
        border: 1px solid #E6E6E9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    div[data-testid="stMetricValue"] { color: #FF4B4B; font-weight: 800; }
    
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
    
    /* Custom Output Cards */
    .result-rain {
        background-color: #FFF0F0; border-left: 5px solid #FF4B4B;
        padding: 25px; border-radius: 6px; text-align: center; color: #7F1D1D;
    }
    .result-clear {
        background-color: #F0FFF4; border-left: 5px solid #10B981;
        padding: 25px; border-radius: 6px; text-align: center; color: #064E3B;
    }
    
    /* Evaluation Tables */
    .eval-table {
        width: 100%; background-color: white; border-radius: 8px;
        border: 1px solid #E6E6E9; padding: 20px;
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
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != target_col]
    
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
st.sidebar.markdown("<h2>⚙️ Input Parameters</h2>", unsafe_allow_html=True)

input_data = {}

with st.sidebar.expander("🌡️ Thermal Profile", expanded=True):
    input_data['MinTemp'] = st.number_input("Min Temp (°C)", value=15.0)
    input_data['MaxTemp'] = st.number_input("Max Temp (°C)", value=25.0)
    input_data['Temp9am'] = st.number_input("Temp at 9am (°C)", value=18.0)

with st.sidebar.expander("💨 Wind Dynamics", expanded=False):
    input_data['WindGustSpeed'] = st.number_input("Gust Speed (km/h)", value=40.0)
    input_data['WindSpeed9am'] = st.number_input("Wind Speed 9am", value=15.0)
    input_data['WindSpeed3pm'] = st.number_input("Wind Speed 3pm", value=20.0)
    if 'WindGustDir' in optimized_features: input_data['WindGustDir'] = st.selectbox("Gust Direction", options=encoders['WindGustDir'].classes_)
    if 'WindDir9am' in optimized_features: input_data['WindDir9am'] = st.selectbox("Wind Dir 9am", options=encoders['WindDir9am'].classes_)
    if 'WindDir3pm' in optimized_features: input_data['WindDir3pm'] = st.selectbox("Wind Dir 3pm", options=encoders['WindDir3pm'].classes_)

with st.sidebar.expander("💧 Moisture & Pressure", expanded=False):
    input_data['Humidity9am'] = st.number_input("Humidity 9am (%)", value=60.0)
    input_data['Humidity3pm'] = st.number_input("Humidity 3pm (%)", value=50.0)
    input_data['Pressure9am'] = st.number_input("Pressure 9am (hPa)", value=1015.0)
    input_data['Pressure3pm'] = st.number_input("Pressure 3pm (hPa)", value=1010.0)
    if 'RainToday' in optimized_features: input_data['RainToday'] = st.selectbox("Rain Today?", options=["No", "Yes"])

# =====================================================================
# MAIN DASHBOARD: THE CLEAN STAGE
# =====================================================================
st.markdown("<h1>🌦️ Atmospheric Early Warning System</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 16px; margin-bottom:30px;'>Powered by Hybrid Binary Particle Swarm Optimization & Support Vector Machine</p>", unsafe_allow_html=True)

# Metrics display - Restored visibility
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
                    <h2 style='color:#991B1B; margin:0;'>🚨 RAINFALL DETECTED</h2>
                    <p style='margin-top:10px;'>The maximum-margin classifier indicates a high probability of precipitation.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="result-clear">
                    <h2 style='color:#065F46; margin:0;'>✅ CLEAR SKIES</h2>
                    <p style='margin-top:10px;'>Atmospheric conditions remain stable. No significant rainfall expected.</p>
                </div>
            """, unsafe_allow_html=True)

st.markdown("<br><hr style='border:1px solid #E6E6E9;'><br>", unsafe_allow_html=True)

# =====================================================================
# BOTTOM SECTION: ENGINE ANALYTICS (FILLING THE SPACE)
# =====================================================================
st.markdown("<h3>📊 Model Evaluation Architecture</h3>", unsafe_allow_html=True)
st.markdown("<p>Historical performance metrics derived from the 42,658-row holdout testing matrix.</p>", unsafe_allow_html=True)

col_cm, col_cr = st.columns(2)

with col_cm:
    st.markdown("""
    <div class="eval-table">
        <h4 style="color:#262730; margin-top:0;">Confusion Matrix</h4>
        <table style="width:100%; text-align:center; border-collapse: collapse;">
            <tr style="background-color:#F4F6F9;">
                <th style="padding:10px; border:1px solid #E6E6E9;"></th>
                <th style="padding:10px; border:1px solid #E6E6E9;">Predicted Dry</th>
                <th style="padding:10px; border:1px solid #E6E6E9;">Predicted Rain</th>
            </tr>
            <tr>
                <td style="padding:10px; border:1px solid #E6E6E9; font-weight:bold; background-color:#F4F6F9;">Actual Dry</td>
                <td style="padding:10px; border:1px solid #E6E6E9; color:#065F46; font-weight:bold;">26,570</td>
                <td style="padding:10px; border:1px solid #E6E6E9; color:#991B1B;">6,525</td>
            </tr>
            <tr>
                <td style="padding:10px; border:1px solid #E6E6E9; font-weight:bold; background-color:#F4F6F9;">Actual Rain</td>
                <td style="padding:10px; border:1px solid #E6E6E9; color:#991B1B;">2,141</td>
                <td style="padding:10px; border:1px solid #E6E6E9; color:#065F46; font-weight:bold;">7,422</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

with col_cr:
    st.markdown("""
    <div class="eval-table">
        <h4 style="color:#262730; margin-top:0;">Classification Report</h4>
        <table style="width:100%; text-align:left; border-collapse: collapse;">
            <tr style="background-color:#F4F6F9; border-bottom:2px solid #E6E6E9;">
                <th style="padding:10px;">Class</th>
                <th style="padding:10px;">Precision</th>
                <th style="padding:10px;">Recall</th>
                <th style="padding:10px;">F1-Score</th>
            </tr>
            <tr style="border-bottom:1px solid #E6E6E9;">
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