# src/data_collection.py

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import requests
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LaunchDataCollector:
    def __init__(self):
        self.launch_data = None
        self.atmospheric_data = None
        
    def fetch_launch_data(self) -> pd.DataFrame:
        """
        Fetch and process historical launch data
        For hackathon purposes, we'll create synthetic data
        In production, this would connect to real APIs
        """
        # Create synthetic launch data
        launches = []
        for _ in range(100):  # 100 sample launches
            launch = {
                'date': pd.Timestamp('2020-01-01') + pd.Timedelta(days=np.random.randint(0, 365*3)),
                'rocket_type': np.random.choice(['Falcon 9', 'Atlas V', 'Ariane 5']),
                'fuel_type': np.random.choice(['RP-1/LOX', 'LH2/LOX', 'Solid']),
                'payload_mass': np.random.normal(5000, 1000),  # kg
                'launch_site_lat': np.random.normal(28.5, 0.1),  # Kennedy Space Center approximation
                'launch_site_lon': np.random.normal(-80.6, 0.1),
                'launch_success': np.random.choice([True, False], p=[0.95, 0.05])
            }
            launches.append(launch)
        
        self.launch_data = pd.DataFrame(launches)
        logger.info(f"Created synthetic launch dataset with {len(launches)} entries")
        return self.launch_data

    def fetch_atmospheric_data(self) -> pd.DataFrame:
        """
        Generate synthetic atmospheric measurements
        """
        if self.launch_data is None:
            raise ValueError("Launch data must be fetched first")
            
        atmospheric_readings = []
        
        for _, launch in self.launch_data.iterrows():
            # Create readings for different altitudes
            for altitude in [1000, 5000, 10000, 20000]:  # meters
                base_co2 = 410 + np.random.normal(0, 5)  # ppm
                base_nox = 0.1 + np.random.normal(0, 0.02)  # ppm
                base_al2o3 = 0.01 + np.random.normal(0, 0.002)  # ppm
                
                reading = {
                    'date': launch['date'],
                    'altitude': altitude,
                    'co2_concentration': base_co2 * (1 - altitude/100000),  # Decrease with altitude
                    'nox_concentration': base_nox * (1 - altitude/80000),
                    'al2o3_concentration': base_al2o3 * (1 - altitude/90000),
                    'temperature': 288.15 * (1 - 0.0065 * altitude / 1000),  # Standard atmosphere
                    'pressure': 101325 * (1 - 0.0065 * altitude / 288.15) ** 5.2561  # Standard atmosphere
                }
                atmospheric_readings.append(reading)
        
        self.atmospheric_data = pd.DataFrame(atmospheric_readings)
        logger.info(f"Created synthetic atmospheric dataset with {len(atmospheric_readings)} entries")
        return self.atmospheric_data

    def prepare_training_data(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Combine launch and atmospheric data into training format
        """
        if self.launch_data is None or self.atmospheric_data is None:
            raise ValueError("Both launch and atmospheric data must be fetched first")
            
        # Merge data based on date
        combined_data = pd.merge(
            self.launch_data,
            self.atmospheric_data,
            on='date',
            how='inner'
        )
        
        # Create feature matrix
        features = combined_data[[
            'payload_mass', 'launch_site_lat', 'launch_site_lon',
            'altitude', 'temperature', 'pressure',
            'rocket_type', 'fuel_type'  # Added these columns for one-hot encoding
        ]]
        
        # Add one-hot encoded categorical variables
        features = pd.get_dummies(features, columns=['rocket_type', 'fuel_type'])
        
        # Create separate target variables
        targets = {
            'co2': combined_data['co2_concentration'],
            'nox': combined_data['nox_concentration'],
            'al2o3': combined_data['al2o3_concentration']
        }
        
        logger.info("Prepared training data with shape: " + str(features.shape))
        return features, targets

if __name__ == "__main__":
    # Test the data collection
    collector = LaunchDataCollector()
    launches = collector.fetch_launch_data()
    atmospheric = collector.fetch_atmospheric_data()
    features, targets = collector.prepare_training_data()
    
    print("\nLaunch data sample:")
    print(launches.head())
    print("\nAtmospheric data sample:")
    print(atmospheric.head())
    print("\nFeature matrix shape:", features.shape)
    for target_name, target_data in targets.items():
        print(f"{target_name} target shape:", target_data.shape)