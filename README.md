# ASCENT: Aerospace System for Chemical Emissions & Numerical Tracking

An AI-powered tool for predicting and visualizing atmospheric chemical composition changes from rocket launches.

## Overview

ASCENT helps predict and analyze the environmental impact of rocket launches by modeling the dispersion of key pollutants (CO2, NOx, and Al2O3) based on launch parameters and weather conditions.

## Features

- Real-time launch impact simulation
- Dynamic pollutant dispersion visualization
- Weather condition integration
- Multiple rocket and fuel type support
- Interactive heatmap generation
- Time-series predictions up to 48 hours
- Adjustable launch parameters:
    - Payload mass
    - Launch coordinates
    - Rocket type
    - Fuel type
    - Simulation duration

## Installation

```bash
git clone https://github.com/notvasub/ascent.git
cd ascent
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run src/app.py
```

Train/retrain the model with new data:
```bash
python src/retrain.py
```

## Project Structure

- `src/app.py`: Main Streamlit interface
- `src/model.py`: Machine learning model implementation
- `src/data_collection.py`: Data handling utilities
- `src/weather_integration.py`: Weather data integration
- `models/`: Saved model files
- `data/`: Holds the training data

## License

This project is licensed under the MIT License - see the LICENSE file for details.