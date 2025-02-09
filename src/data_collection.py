import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LaunchDataCollector:
    def __init__(self, csv_file: str = "structured_launch_data.csv"):
        self.csv_file = csv_file
        self.launch_data = None
    
    def load_data(self) -> pd.DataFrame:
        try:
            self.launch_data = pd.read_csv(self.csv_file, parse_dates=['date'])
            logger.info("Launch data successfully loaded from CSV.")
            return self.launch_data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return pd.DataFrame()
    
    def get_summary(self) -> pd.DataFrame:
        if self.launch_data is None:
            logger.error("No data loaded. Please run load_data() first.")
            return pd.DataFrame()
        
        summary = self.launch_data.groupby(['rocket_type', 'fuel_type']).agg(
            launch_count=('date', 'count'),
            avg_payload_mass=('payload_mass', 'mean')
        ).reset_index()
        return summary
    
    def filter_by_success(self, success: bool) -> pd.DataFrame:
        if self.launch_data is None:
            logger.error("No data loaded. Please run load_data() first.")
            return pd.DataFrame()
        
        return self.launch_data[self.launch_data['launch_success'] == success]
    
    def get_atmospheric_impact(self) -> pd.DataFrame:
        if self.launch_data is None:
            logger.error("No data loaded. Please run load_data() first.")
            return pd.DataFrame()
        
        impact = self.launch_data.groupby('altitude').agg(
            avg_co2=('co2_concentration', 'mean'),
            avg_nox=('nox_concentration', 'mean'),
            avg_al2o3=('al2o3_concentration', 'mean')
        ).reset_index()
        return impact

if __name__ == "__main__":
    collector = LaunchDataCollector()
    data = collector.load_data()
    print("Summary of launches:")
    print(collector.get_summary())
    print("\nAtmospheric impact analysis:")
    print(collector.get_atmospheric_impact())
