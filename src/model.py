import joblib
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class LaunchImpactModel:
    def __init__(self):
        self.models = {}
        self.feature_columns = None  

    def train(self, features: pd.DataFrame, targets: pd.DataFrame):
        """Train separate models for each pollutant and save feature columns."""
        self.models = {
            'co2': RandomForestRegressor(n_estimators=100, random_state=42),
            'nox': RandomForestRegressor(n_estimators=100, random_state=42),
            'al2o3': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        for pollutant, model in self.models.items():
            model.fit(features, targets[pollutant])

        self.feature_columns = list(features.columns)
        
        os.makedirs("models", exist_ok=True)

        for pollutant, model in self.models.items():
            joblib.dump(model, f"models/{pollutant}_model.joblib")
        
        joblib.dump(self.feature_columns, "models/feature_columns.joblib")
        print("✅ Model trained and saved successfully!")

    def load_models(self):
        try:
            self.feature_columns = joblib.load("models/feature_columns.joblib") 
            for pollutant in ['co2', 'nox', 'al2o3']:
                self.models[pollutant] = joblib.load(f"models/{pollutant}_model.joblib")
            print("✅ Models loaded successfully!")
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            self.models = {}
            self.feature_columns = None

    def predict(self, input_data: pd.DataFrame, duration_hours: int = 24):
        if not self.models:
            raise ValueError("Models are not loaded. Call load_models() first.")
        
        if self.feature_columns is None:
            raise ValueError("Feature columns are missing. Ensure the model was trained correctly.")

        for col in self.feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0 
        
        input_data = input_data[self.feature_columns] 

        time_series_predictions = []
        
        for hour in range(duration_hours):
            decay_factor = np.exp(-hour / 12.0)  
            predictions = {pollutant: self.models[pollutant].predict(input_data)[0] * decay_factor for pollutant in self.models}
            time_series_predictions.append(predictions)
        
        return pd.DataFrame(time_series_predictions) 

if __name__ == "__main__":
    model = LaunchImpactModel()
    model.load_models()
    
    example_input = pd.DataFrame({
        'payload_mass': [5000],
        'launch_site_lat': [28.5729],
        'launch_site_lon': [-80.6490],
        'rocket_type_Falcon 9': [1],
        'fuel_type_RP-1/LOX': [1]
    })
    
    predictions = model.predict(example_input)
    print(predictions)
