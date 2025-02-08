# test_pipeline.py

from src.data_collection import LaunchDataCollector
from src.model import LaunchImpactModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. Data Collection
    logger.info("Starting data collection...")
    collector = LaunchDataCollector()
    launches = collector.fetch_launch_data()
    atmospheric = collector.fetch_atmospheric_data()
    features, targets = collector.prepare_training_data()
    
    # 2. Model Training
    logger.info("Training models...")
    model = LaunchImpactModel()
    metrics = model.train(features, targets)
    
    # 3. Save Models
    logger.info("Saving models...")
    model.save_models()
    
    # 4. Test Predictions
    logger.info("Testing predictions...")
    sample_features = features.iloc[[0]]  # Use first row as example
    predictions = model.predict(sample_features)
    
    # Print results
    print("\nModel Performance Metrics:")
    for target_name, metric_values in metrics.items():
        print(f"\n{target_name.upper()} Model:")
        for metric_name, value in metric_values.items():
            print(f"  {metric_name}: {value:.4f}")
    
    print("\nSample Prediction Results:")
    for target_name, pred_values in predictions.items():
        print(f"{target_name}: {pred_values[0]:.4f}")

if __name__ == "__main__":
    main()