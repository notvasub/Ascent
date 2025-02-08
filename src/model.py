import joblib
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class LaunchImpactModel:
    def __init__(self):
        self.models = {}  # Dictionary to store pollutant models
        self.feature_columns = None  # List of feature names

    def train(self, features: pd.DataFrame, targets: pd.DataFrame):
        """Train separate models for each pollutant and save feature columns."""
        self.models = {
            'co2': RandomForestRegressor(),
            'nox': RandomForestRegressor(),
            'al2o3': RandomForestRegressor()
        }
        
        for pollutant, model in self.models.items():
            model.fit(features, targets[pollutant])

        # Save feature column names
        self.feature_columns = list(features.columns)
        
        # Ensure model directory exists
        os.makedirs("models", exist_ok=True)

        # Save trained models and feature columns
        for pollutant, model in self.models.items():
            joblib.dump(model, f"models/{pollutant}_model.joblib")
        
        joblib.dump(self.feature_columns, "models/feature_columns.joblib")
        print("✅ Model trained and saved successfully!")

    def load_models(self):
        """Load trained models and feature columns."""
        try:
            self.feature_columns = joblib.load("models/feature_columns.joblib")  # Load feature names
            for pollutant in ['co2', 'nox', 'al2o3']:
                self.models[pollutant] = joblib.load(f"models/{pollutant}_model.joblib")
            print("✅ Models loaded successfully!")
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            self.models = {}
            self.feature_columns = None

    def predict(self, input_data: pd.DataFrame):
        """Make predictions for given launch parameters."""
        if not self.models:
            raise ValueError("Models are not loaded. Call load_models() first.")
        
        if self.feature_columns is None:
            raise ValueError("Feature columns are missing. Ensure the model was trained correctly.")

        # Ensure input_data has the correct feature columns
        for col in self.feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0  # Fill missing columns with 0
        
        input_data = input_data[self.feature_columns]  # Ensure correct column order

        # Predict emissions
        predictions = {pollutant: self.models[pollutant].predict(input_data)[0] for pollutant in self.models}
        return pd.DataFrame([predictions])  # Return as a DataFrame

