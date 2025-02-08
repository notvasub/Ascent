import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMapWithTime
from streamlit_folium import folium_static
import joblib
from data_collection import LaunchDataCollector
from model import LaunchImpactModel
from weather_integration import LaunchWeatherAnalyzer

# Initialize components
collector = LaunchDataCollector()
model = LaunchImpactModel()
analyzer = LaunchWeatherAnalyzer()

# Load trained model (ensure models are pre-trained)
try:
    model.load_models()
    st.success("Model loaded successfully!")
    if model.feature_columns is None:
        st.error("Error: Model feature columns are not set. Ensure the model was trained properly.")
        st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Rocket Launch Impact Visualization")
st.sidebar.header("Input Launch Parameters")

# User Inputs
rocket_type = st.sidebar.selectbox("Rocket Type", ['Falcon 9', 'Atlas V', 'Ariane 5'])
fuel_type = st.sidebar.selectbox("Fuel Type", ['RP-1/LOX', 'LH2/LOX', 'Solid'])
payload_mass = st.sidebar.slider("Payload Mass (kg)", 1000, 20000, 5000)
launch_site_lat = st.sidebar.number_input("Launch Site Latitude", value=28.5729)
launch_site_lon = st.sidebar.number_input("Launch Site Longitude", value=-80.6490)

if st.sidebar.button("Simulate Launch"):
    if model.feature_columns is None:
        st.error("Model feature columns are missing. Cannot proceed with predictions.")
        st.stop()
    
    # Create input DataFrame
    input_data = pd.DataFrame({
        'payload_mass': [payload_mass],
        'launch_site_lat': [launch_site_lat],
        'launch_site_lon': [launch_site_lon],
        f'rocket_type_{rocket_type}': [1],
        f'fuel_type_{fuel_type}': [1]
    })
    
    # Ensure all expected columns exist
    for col in model.feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Predict pollutant emissions
    predictions = model.predict(input_data)
    st.write("### Predicted Emissions:")
    st.write(predictions)
    
    if predictions is None or predictions.empty:
        st.error("No predictions generated. Check model output.")
    else:
        # Get weather conditions & dispersion patterns
        try:
            analysis = analyzer.analyze_launch_conditions(launch_site_lat, launch_site_lon, predictions)
            dispersion_data = analysis.get('dispersion_patterns', {})
        except Exception as e:
            st.error(f"Error retrieving weather and dispersion data: {e}")
            dispersion_data = {}
        
        # Generate animated heatmap data
        heat_data = []
        timestamps = []
        duration_hours = 24
        
        for hour in range(duration_hours):
            frame_data = []
            dispersion_factor = np.exp(-hour / 6.0) * 10  # Scaling factor for better visibility
            
            for pollutant, distances in dispersion_data.items():
                for distance, heights in distances.items():
                    for _, conc in heights.items():
                        frame_data.append([
                            launch_site_lat + np.random.normal(0, dispersion_factor * 0.01),
                            launch_site_lon + np.random.normal(0, dispersion_factor * 0.01),
                            conc * dispersion_factor
                        ])
            
            if frame_data:
                heat_data.append(frame_data)
                timestamps.append(f"Hour {hour}")
        
        if heat_data:
            # Create folium map
            m = folium.Map(location=[launch_site_lat, launch_site_lon], zoom_start=7)
            HeatMapWithTime(heat_data, index=timestamps, radius=20).add_to(m)
            folium_static(m)
        else:
            st.error("No heatmap data generated. Check dispersion model.")
