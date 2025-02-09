import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import json
import os
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WeatherCondition:
    temperature: float 
    pressure: float    
    humidity: float    
    wind_speed: float  
    wind_direction: float  
    cloud_cover: float    

class WeatherIntegrator:
    def __init__(self):
        self.api_key = os.getenv('OPENWEATHER_API_KEY', 'demo_key')
        self.base_url = "http://api.openweathermap.org/data/2.5"
        
    def get_current_weather(self, lat: float, lon: float) -> WeatherCondition:
        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'standard'  
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return WeatherCondition(
                temperature=data['main']['temp'],
                pressure=data['main']['pressure'],
                humidity=data['main']['humidity'],
                wind_speed=max(data['wind']['speed'], 1.0),
                wind_direction=data['wind'].get('deg', 0),
                cloud_cover=data['clouds']['all']
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching weather data: {e}")
            return self._get_dummy_weather()
    
    def _get_dummy_weather(self) -> WeatherCondition:
        return WeatherCondition(
            temperature=288.15,
            pressure=1013.25,
            humidity=70.0,
            wind_speed=5.0,
            wind_direction=180.0,
            cloud_cover=30.0
        )

class DispersionCalculator:
    def __init__(self):
        self.stability_classes = {
            'A': {'a': 0.527, 'b': 0.865},
            'B': {'a': 0.371, 'b': 0.866},
            'C': {'a': 0.209, 'b': 0.897},
            'D': {'a': 0.128, 'b': 0.905},
            'E': {'a': 0.098, 'b': 0.902},
            'F': {'a': 0.065, 'b': 0.902},
        }
    
    def determine_stability_class(self, weather: WeatherCondition) -> str:
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
    
    def calculate_dispersion(self, weather: WeatherCondition, emission_rate: float, distance: float, time_step: int) -> Dict[str, float]:
        stability_class = self.determine_stability_class(weather)
        params = self.stability_classes[stability_class]
        wind_effect = weather.wind_speed / 5.0 
        sigma_y = params['a'] * (distance ** params['b']) * wind_effect
        sigma_z = params['a'] * (distance ** params['b']) * wind_effect  
        heights = [0, 50, 100, 200, 500]  
        concentrations = {}
        for z in heights:
            concentration = (emission_rate / (2 * np.pi * weather.wind_speed * sigma_y * sigma_z) *
                            np.exp(-0.5 * (z - 50)**2 / sigma_z**2) * np.exp(-time_step / 10.0))  
            concentrations[f'{z}m'] = concentration  
        return concentrations

class LaunchWeatherAnalyzer:
    def __init__(self):
        self.weather = WeatherIntegrator()
        self.dispersion = DispersionCalculator()
        
    def analyze_launch_conditions(self, lat: float, lon: float, predicted_emissions: Dict[str, float]) -> Dict[str, Any]:
        weather_conditions = self.weather.get_current_weather(lat, lon)
        dispersion_patterns = {}
        for pollutant, emission_rate in predicted_emissions.items():
            distances = [100, 500, 1000, 2000, 5000]
            patterns = {}
            for time_step in range(24):
                time_patterns = {}
                for distance in distances:
                    concentrations = self.dispersion.calculate_dispersion(
                        weather_conditions,
                        emission_rate,
                        distance,
                        time_step
                    )
                    time_patterns[f'{distance}m'] = concentrations
                dispersion_patterns.setdefault(pollutant, {}).update({f'Hour {time_step}': time_patterns})
        return {
            'weather': weather_conditions,
            'dispersion_patterns': dispersion_patterns
        }

if __name__ == "__main__":
    analyzer = LaunchWeatherAnalyzer()
    test_location = {'lat': 28.5729, 'lon': -80.6490}
    test_emissions = {'co2': 1000.0, 'nox': 50.0, 'al2o3': 10.0}
    results = analyzer.analyze_launch_conditions(test_location['lat'], test_location['lon'], test_emissions)
    print(results)
