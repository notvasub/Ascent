import pandas as pd
import logging
from src.model import LaunchImpactModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CSV_FILE = "data/structured_launch_data.csv"

def load_data(csv_file: str):
    try:
        data = pd.read_csv(csv_file, parse_dates=['date'])
        logger.info(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns from {csv_file}.")
        return data
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None

def preprocess_data(data: pd.DataFrame):
    feature_columns = ['altitude', 'fuel_type', 'rocket_type', 'payload_mass'] 
    target_columns = ['co2_concentration', 'nox_concentration', 'al2o3_concentration']
    
    if not all(col in data.columns for col in feature_columns + target_columns):
        logger.error("Missing necessary columns in the dataset.")
        return None, None
    
    features = data[feature_columns]
    targets = data[target_columns].rename(columns={
        'co2_concentration': 'co2',
        'nox_concentration': 'nox',
        'al2o3_concentration': 'al2o3'
    })
    
    features = pd.get_dummies(features, columns=['fuel_type', 'rocket_type'], drop_first=True)
    
    return features, targets

def retrain_model():

    logger.info("Starting model retraining...")
    data = load_data(CSV_FILE)
    
    if data is None:
        logger.error("No data available for training.")
        return
    
    features, targets = preprocess_data(data)
    if features is None or targets is None:
        logger.error("Data preprocessing failed.")
        return
    
    model = LaunchImpactModel()
    model.train(features, targets)
    logger.info("✅ Model retraining complete!")

if __name__ == "__main__":
    retrain_model()