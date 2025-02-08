from src.data_collection import LaunchDataCollector
from src.model import LaunchImpactModel

# Step 1: Initialize Data Collector
collector = LaunchDataCollector()

# Step 2: Fetch Data (Fixing the Error)
launch_data = collector.fetch_launch_data()
atmospheric_data = collector.fetch_atmospheric_data()

if launch_data is None or atmospheric_data is None:
    raise ValueError("Error: Could not fetch launch or atmospheric data. Check your data sources.")

# Step 3: Prepare Features and Targets
features, targets = collector.prepare_training_data()

# Step 4: Train and Save Model
model = LaunchImpactModel()
model.train(features, targets)

print("✅ Model retrained successfully!")
