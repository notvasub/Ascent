# src/weather_integration.py

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any  # Added Any to imports
from dataclasses import dataclass
import json
from pathlib import Path
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WeatherCondition:
    temperature: float  # Kelvin
    pressure: float    # hPa
    humidity: float    # %
    wind_speed: float  # m/s
    wind_direction: float  # degrees
    cloud_cover: float    # %

class WeatherIntegrator:
    def __init__(self):
        self.api_key = os.getenv('OPENWEATHER_API_KEY', 'demo_key')
        self.base_url = "http://api.openweathermap.org/data/2.5"
        
    def get_current_weather(self, lat: float, lon: float) -> WeatherCondition:
        """
        Fetch current weather conditions for a given location
        """
        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'standard'  # Use Kelvin for temperature
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return WeatherCondition(
                temperature=data['main']['temp'],
                pressure=data['main']['pressure'],
                humidity=data['main']['humidity'],
                wind_speed=data['wind']['speed'],
                wind_direction=data['wind'].get('deg', 0),
                cloud_cover=data['clouds']['all']
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching weather data: {e}")
            # Return dummy data for demo purposes
            return self._get_dummy_weather()
    
    def _get_dummy_weather(self) -> WeatherCondition:
        """
        Generate dummy weather data for testing or when API is unavailable
        """
        return WeatherCondition(
            temperature=288.15,  # 15°C
            pressure=1013.25,    # Standard atmospheric pressure
            humidity=70.0,
            wind_speed=5.0,
            wind_direction=180.0,
            cloud_cover=30.0
        )

class DispersionCalculator:
    def __init__(self):
        """
        Initialize the dispersion calculator with default parameters
        """
        self.stability_classes = {
            'A': {'a': 0.527, 'b': 0.865},  # Very unstable
            'B': {'a': 0.371, 'b': 0.866},  # Moderately unstable
            'C': {'a': 0.209, 'b': 0.897},  # Slightly unstable
            'D': {'a': 0.128, 'b': 0.905},  # Neutral
            'E': {'a': 0.098, 'b': 0.902},  # Slightly stable
            'F': {'a': 0.065, 'b': 0.902},  # Moderately stable
        }
    
    def determine_stability_class(self, weather: WeatherCondition) -> str:
        """
        Determine Pasquill-Gifford stability class based on weather conditions
        Simplified version for demonstration
        """
        if weather.wind_speed < 2:
            return 'F'
        elif weather.wind_speed < 3:
            return 'E'
        elif weather.wind_speed < 5:
            return 'D'
        elif weather.wind_speed < 6:
            return 'C'
        elif weather.wind_speed < 8:
            return 'B'
        else:
            return 'A'
    
    def calculate_dispersion(self, 
                           weather: WeatherCondition,
                           emission_rate: float,
                           distance: float,
                           stack_height: float = 10.0) -> Dict[str, float]:
        """
        Calculate pollutant dispersion using Gaussian plume model
        """
        stability_class = self.determine_stability_class(weather)
        params = self.stability_classes[stability_class]
        
        # Calculate dispersion coefficients
        sigma_y = params['a'] * distance ** params['b']
        sigma_z = params['a'] * distance ** params['b']  # Simplified for demo
        
        # Calculate concentrations at different heights
        heights = [0, 50, 100, 200, 500]  # meters
        concentrations = {}
        
        for z in heights:
            # Gaussian plume equation
            concentration = (emission_rate / (2 * np.pi * weather.wind_speed * sigma_y * sigma_z) *
                           np.exp(-0.5 * (z - stack_height)**2 / sigma_z**2))
            
            concentrations[f'{z}m'] = concentration
            
        return concentrations

class LaunchWeatherAnalyzer:
    def __init__(self):
        self.weather = WeatherIntegrator()
        self.dispersion = DispersionCalculator()
        
    def analyze_launch_conditions(self, 
                                lat: float, 
                                lon: float, 
                                predicted_emissions: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze launch conditions and predict dispersion patterns
        """
        # Get current weather conditions
        weather_conditions = self.weather.get_current_weather(lat, lon)
        
        # Calculate dispersion for each pollutant
        dispersion_patterns = {}
        for pollutant, emission_rate in predicted_emissions.items():
            # Calculate dispersion at different distances
            distances = [100, 500, 1000, 2000, 5000]  # meters
            patterns = {}
            
            for distance in distances:
                concentrations = self.dispersion.calculate_dispersion(
                    weather_conditions,
                    emission_rate,
                    distance
                )
                patterns[f'{distance}m'] = concentrations
                
            dispersion_patterns[pollutant] = patterns
        
        return {
            'weather': weather_conditions,
            'dispersion_patterns': dispersion_patterns
        }
    
    def generate_risk_assessment(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate a risk assessment based on weather and dispersion patterns
        """
        weather = analysis_results['weather']
        patterns = analysis_results['dispersion_patterns']
        
        # Simple risk assessment logic
        risk_levels = {
            'weather_risk': 'LOW' if weather.wind_speed < 10 and weather.cloud_cover < 80 else 'MODERATE',
            'dispersion_risk': 'LOW',
            'overall_risk': 'LOW'
        }
        
        # Check maximum concentrations
        max_concentrations = {}
        for pollutant, distances in patterns.items():
            max_conc = 0
            for distance_data in distances.values():
                max_conc = max(max_conc, max(distance_data.values()))
            max_concentrations[pollutant] = max_conc
            
            if max_conc > 1.0:  # Example threshold
                risk_levels['dispersion_risk'] = 'HIGH'
                risk_levels['overall_risk'] = 'HIGH'
        
        return risk_levels

if __name__ == "__main__":
    # Test the weather integration
    analyzer = LaunchWeatherAnalyzer()
    
    # Example launch site (Kennedy Space Center coordinates)
    test_location = {
        'lat': 28.5729,
        'lon': -80.6490
    }
    
    # Example predicted emissions (g/s)
    test_emissions = {
        'co2': 1000.0,
        'nox': 50.0,
        'al2o3': 10.0
    }
    
    # Run analysis
    results = analyzer.analyze_launch_conditions(
        test_location['lat'],
        test_location['lon'],
        test_emissions
    )
    
    # Generate risk assessment
    risk_assessment = analyzer.generate_risk_assessment(results)
    
    # Print results
    print("\nWeather Conditions:")
    print(f"Temperature: {results['weather'].temperature - 273.15:.1f}°C")
    print(f"Wind Speed: {results['weather'].wind_speed} m/s")
    print(f"Wind Direction: {results['weather'].wind_direction}°")
    
    print("\nDispersion Patterns (sample):")
    for pollutant, patterns in results['dispersion_patterns'].items():
        print(f"\n{pollutant.upper()}:")
        for distance, concentrations in list(patterns.items())[:2]:  # Show first two distances
            print(f"  {distance}:")
            for height, conc in list(concentrations.items())[:3]:  # Show first three heights
                print(f"    {height}: {conc:.2e} g/m³")
    
    print("\nRisk Assessment:")
    for risk_type, level in risk_assessment.items():
        print(f"{risk_type}: {level}")