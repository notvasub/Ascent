import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap
from model import LaunchImpactModel
from weather_integration import LaunchWeatherAnalyzer
import numpy as np

model = LaunchImpactModel()
weather_analyzer = LaunchWeatherAnalyzer()
model.load_models()

st.title("ASCENT: Aerospace System for Chemical Emissions & Numerical Tracking")
st.sidebar.header("Launch Input Parameters")

latitude = st.sidebar.number_input("Launch Latitude", value=28.5729, format="%.4f")
longitude = st.sidebar.number_input("Launch Longitude", value=-80.6490, format="%.4f")
payload_mass = st.sidebar.number_input("Payload Mass (kg)", value=5000, min_value=0)
fuel_type = st.sidebar.selectbox("Fuel Type", ["RP-1/LOX", "LH2/LOX", "Solid"])
rocket_type = st.sidebar.selectbox("Rocket Type", ["Falcon 9", "Atlas V", "Soyuz", "Electron"])
sim_duration = st.sidebar.slider("Simulation Duration (hours)", 1, 48, 24)

if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = None
if "pollutant_data" not in st.session_state:
    st.session_state.pollutant_data = None

if st.sidebar.button("Run Simulation"):

    input_data = pd.DataFrame({
        'payload_mass': [payload_mass],
        'launch_site_lat': [latitude],
        'launch_site_lon': [longitude],
        f'rocket_type_{rocket_type}': [1],
        f'fuel_type_{fuel_type}': [1]
    })
    
    emissions = model.predict(input_data, duration_hours=sim_duration)
    emissions['co2'] *= (payload_mass / 4000) ** 1.2 * np.random.uniform(1.0, 1.3)
    emissions['nox'] *= (1.5 if fuel_type == "RP-1/LOX" else 1.0) * np.random.uniform(1.0, 1.3)
    emissions['al2o3'] *= (2.0 if fuel_type == "Solid" else 0.8) * np.random.uniform(1.0, 1.3)
    
    results = weather_analyzer.analyze_launch_conditions(latitude, longitude, {
        'co2': emissions['co2'].sum(),
        'nox': emissions['nox'].sum(),
        'al2o3': emissions['al2o3'].sum()
    })
    
    dispersion_data = results['dispersion_patterns']
    weather_info = results['weather']
    
    wind_speed = weather_info.wind_speed * 1.5  
    wind_direction = weather_info.wind_direction  
    heatmap_data = {pollutant: [] for pollutant in dispersion_data.keys()}
    
    for pollutant, time_data in dispersion_data.items():
        for hour, distance_data in time_data.items():
            for distance, concentration_levels in distance_data.items():
                for altitude, concentration in concentration_levels.items():
                    try:
                        dist_float = float(distance.rstrip('m'))
                        decay_factor = 1 / (1 + dist_float / 500)  
                        wind_adjustment_lat = dist_float * 0.0002 * np.cos(np.radians(wind_direction)) * wind_speed / 10
                        wind_adjustment_lon = dist_float * 0.0002 * np.sin(np.radians(wind_direction)) * wind_speed / 10
                        adjusted_lat = latitude + wind_adjustment_lat
                        adjusted_lon = longitude + wind_adjustment_lon
                        heatmap_data[pollutant].append([
                            adjusted_lat,
                            adjusted_lon,
                            float(concentration) * decay_factor
                        ])
                    except (ValueError, TypeError, AttributeError):
                        continue

    st.session_state.simulation_results = results
    st.session_state.pollutant_data = heatmap_data

if st.session_state.pollutant_data:

    selected_pollutant = st.selectbox(
        "Select Pollutant to Display",
        list(st.session_state.pollutant_data.keys()),
        format_func=lambda x: x.upper()
    )
    
    st.subheader(f"Pollutant Dispersion Heatmap - {selected_pollutant.upper()}")
    m = folium.Map(location=[latitude, longitude], zoom_start=8)
    HeatMap(st.session_state.pollutant_data[selected_pollutant], radius=20, blur=15, max_zoom=1).add_to(m)
    folium_static(m)
    
    weather_info = st.session_state.simulation_results['weather']
    st.subheader("Current Weather Conditions at Launch Site")
    st.write(f"**Temperature:** {weather_info.temperature:.2f} K")
    st.write(f"**Pressure:** {weather_info.pressure} hPa")
    st.write(f"**Humidity:** {weather_info.humidity}%")
    st.write(f"**Wind Speed:** {weather_info.wind_speed} m/s")
    st.write(f"**Wind Direction:** {weather_info.wind_direction}°")
    st.write(f"**Cloud Cover:** {weather_info.cloud_cover}%")
    
    df_dispersion = pd.DataFrame({
        'Pollutant': [],
        'Latitude': [],
        'Longitude': [],
        'Concentration': []
    })
    
    for pollutant, data in st.session_state.pollutant_data.items():
        temp_df = pd.DataFrame(data, columns=['Latitude', 'Longitude', 'Concentration'])
        temp_df['Pollutant'] = pollutant
        df_dispersion = pd.concat([df_dispersion, temp_df], ignore_index=True)
    
    csv = df_dispersion.to_csv(index=False).encode('utf-8')
    st.download_button("Download Dispersion Data", csv, "dispersion_data.csv", "text/csv")
