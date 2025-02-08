# test_integration.py

from src.data_collection import LaunchDataCollector
from src.model import LaunchImpactModel
from src.weather_integration import LaunchWeatherAnalyzer
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. Data Collection and Model Training
    logger.info("Starting data collection and model training...")
    collector = LaunchDataCollector()
    launches = collector.fetch_launch_data()
    atmospheric = collector.fetch_atmospheric_data()
    features, targets = collector.prepare_training_data()
    
    model = LaunchImpactModel()
    metrics = model.train(features, targets)
    
    # 2. Make a prediction for a new launch
    logger.info("Making predictions for a new launch...")
    sample_launch = {
        'payload_mass': 5000.0,
        'launch_site_lat': 28.5729,  # Kennedy Space Center
        'launch_site_lon': -80.6490,
        'altitude': 1000.0,
        'temperature': 288.15,
        'pressure': 101325.0,
        'rocket_type': 'Falcon 9',
        'fuel_type': 'RP-1/LOX'
    }
    
    # Convert to DataFrame with same structure as training data
    sample_df = pd.DataFrame([sample_launch])
    sample_df = pd.get_dummies(sample_df, columns=['rocket_type', 'fuel_type'])
    
    # Add any missing columns from training data
    for col in features.columns:
        if col not in sample_df.columns:
            sample_df[col] = 0
            
    # Ensure columns are in same order
    sample_df = sample_df[features.columns]
    
    predictions = model.predict(sample_df)
    
    # 3. Weather Integration
    logger.info("Analyzing weather conditions and dispersion...")
    analyzer = LaunchWeatherAnalyzer()
    
    # Convert predictions to g/s for dispersion calculation
    emission_rates = {
        'co2': predictions['co2'][0] * 1000,  # Convert to g/s
        'nox': predictions['nox'][0] * 1000,
        'al2o3': predictions['al2o3'][0] * 1000
    }
    
    weather_analysis = analyzer.analyze_launch_conditions(
        sample_launch['launch_site_lat'],
        sample_launch['launch_site_lon'],
        emission_rates
    )
    
    risk_assessment = analyzer.generate_risk_assessment(weather_analysis)
    
    # Print results
    print("\nPredicted Emissions:")
    for pollutant, value in predictions.items():
        print(f"{pollutant}: {value[0]:.2f} ppm")
    
    print("\nWeather Conditions:")
    weather = weather_analysis['weather']
    print(f"Temperature: {weather.temperature - 273.15:.1f}°C")
    print(f"Wind Speed: {weather.wind_speed} m/s")
    print(f"Wind Direction: {weather.wind_direction}°")
    
    print("\nRisk Assessment:")
    for risk_type, level in risk_assessment.items():
        print(f"{risk_type}: {level}")

if __name__ == "__main__":
    main()