import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import geopandas as gpd
import folium
from folium.plugins import HeatMapWithTime
import matplotlib.colors as mcolors
from scipy.stats import norm
from IPython.display import HTML

class DataVisualizer:
    def __init__(self, features: pd.DataFrame, targets: dict, predictions: pd.DataFrame, launch_data: pd.DataFrame):
        self.features = features
        self.targets = pd.DataFrame(targets)
        self.predictions = predictions.fillna(0)  # Ensure no NaN values
        self.launch_data = launch_data

    def plot_actual_vs_predicted(self):
        """Creates scatter plots comparing actual vs. predicted values for pollutants, including error histograms."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        pollutants = ['co2', 'nox', 'al2o3']
        
        for i, pollutant in enumerate(pollutants):
            sns.scatterplot(x=self.targets[pollutant], y=self.predictions[pollutant], ax=axes[0, i], alpha=0.5)
            axes[0, i].plot([self.targets[pollutant].min(), self.targets[pollutant].max()],
                            [self.targets[pollutant].min(), self.targets[pollutant].max()],
                            'r--', lw=2)  # Ideal y = x line
            axes[0, i].set_xlabel("Actual Concentration (ppm)")
            axes[0, i].set_ylabel("Predicted Concentration (ppm)")
            axes[0, i].set_title(f"{pollutant.upper()} Prediction Accuracy")
            
            # Error Distribution
            errors = self.targets[pollutant] - self.predictions[pollutant]
            sns.histplot(errors, kde=True, ax=axes[1, i], bins=20)
            axes[1, i].set_title(f"{pollutant.upper()} Error Distribution")
        
        plt.suptitle("Model Predictions vs Actual Values")
        plt.show()
    
    def create_animated_dispersion_map(self, pollutant: str, duration_hours: int = 24):
        """Creates an animated heatmap of gas dispersion over time with a playbar, using rocket type dynamics."""
        center_lat = self.launch_data['launch_site_lat'].mean()
        center_lon = self.launch_data['launch_site_lon'].mean()
        
        # Ensure predictions vary by rocket type
        if 'rocket_type' in self.features.columns:
            rocket_effect = self.features.groupby('rocket_type')[pollutant].mean().to_dict()
        else:
            rocket_effect = {}
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        heat_data = []
        timestamps = []
        
        for hour in range(duration_hours):
            dispersion_factor = np.exp(-hour / 6.0) * 10  # Increased scale for visibility
            frame_data = []
            
            for _, launch in self.launch_data.iterrows():
                rocket_type = launch.get('rocket_type', 'default')
                rocket_multiplier = rocket_effect.get(rocket_type, 1.0)
                
                for _ in range(100):
                    lat_offset = np.random.normal(0, dispersion_factor * 0.005)
                    lon_offset = np.random.normal(0, dispersion_factor * 0.005)
                    value = max(self.predictions.get(pollutant, pd.Series([0])).mean() * rocket_multiplier * dispersion_factor, 0)
                    frame_data.append([
                        launch['launch_site_lat'] + lat_offset,
                        launch['launch_site_lon'] + lon_offset,
                        value
                    ])
            
            if not frame_data:
                print(f"Warning: No heatmap data generated for hour {hour}.")
            else:
                heat_data.append(frame_data)
                timestamps.append(f"Hour {hour}")
        
        if heat_data:
            HeatMapWithTime(heat_data, index=timestamps, radius=20).add_to(m)
            m.save("animated_dispersion_map.html")
            print("Animated dispersion map saved as animated_dispersion_map.html")
        else:
            print("Error: No valid heatmap data generated.")
    
    def show_all_visualizations(self):
        """Runs all visualization functions."""
        self.plot_actual_vs_predicted()
        self.create_animated_dispersion_map('co2')

# Example usage
if __name__ == "__main__":
    from data_collection import LaunchDataCollector
    try:
        from model import LaunchImpactModel
    except ImportError as e:
        print("Error importing LaunchImpactModel. Ensure model.py is correctly set up.")
        raise e
    
    # Collect and prepare data
    collector = LaunchDataCollector()
    launches = collector.fetch_launch_data()
    atmospheric = collector.fetch_atmospheric_data()
    features, targets = collector.prepare_training_data()
    
    # Train the model
    model = LaunchImpactModel()
    model.train(features, targets)
    predictions = pd.DataFrame(model.predict(features)).fillna(0)  # Ensure predictions are a DataFrame and not NaN
    
    # Print sample predictions for debugging
    print("Sample predictions:")
    print(predictions.head())
    
    # Create visualizations
    visualizer = DataVisualizer(features, targets, predictions, launches)
    visualizer.show_all_visualizations()