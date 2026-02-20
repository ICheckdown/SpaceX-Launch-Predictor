# SpaceX Launch Success Predictor

A machine learning system that predicts SpaceX launch outcomes. The model analyzes 150 historical launches with weather features to predict launch success or failure. 

## 🚀 Project Overview

This project demonstrates an end-to-end machine learning pipeline:
- Integrates Launch Library 2 API and Open-Meteo Weather API
- Decision Tree classifier with class imbalance handling
- REST API with FastAPI for real-time predictions
- **Accuracy**: 87% on test data (141 successes, 9 failures)

## 📊 Key Findings

The model identified **year** as the most important factor, followed by weather conditions. 

**Feature Importance Ranking**:
1. Year (likely due to technological advancements)
2. Weather conditions (wind speed, temperature, precipitation, cloud cover)


## 🛠️ Tech Stack

- **Python 3.x**
- **Data Processing**: pandas, requests
- **Machine Learning**: scikit-learn
- **API Framework**: FastAPI, Pydantic
- **Model Persistence**: joblib


## 🔧 Installation
**requirements.txt**:
```
pandas
requests
scikit-learn
fastapi
uvicorn
pydantic
joblib
```

## Usage

### Running the API
```bash
uvicorn api:app --reload
```
Run this in the terminal of api.py



## 📈 Model Performance
```
Accuracy: 87%

Classification Report:
              precision    recall  f1-score   support
     Success       0.96      0.90      0.93        29
     Failure       0.00      0.00      0.00         1

Confusion Matrix:
[[26  3]
 [ 1  0]]
```

Due to class imbalance, the model struggles to correctly classify launches that fail. Future improvements would definetly be to scrape more failures. 


## 🔍 Data Collection Process

### 1. Launch Data (Launch Library 2 API)
- **Source**: `https://ll.thespacedevs.com/2.3.0/launches/`
- **Filter**: SpaceX launches only
- **Fields**: Launch date, status, rocket type, location, and coordinates

### 2. Weather Data (Open-Meteo API)
- **Source**: `https://archive-api.open-meteo.com/v1/archive`
- **Features**: Temperature, wind speed, precipitation, and cloud cover


### 3. Feature Engineering
- **Temporal**: Month, day of week, year
- **Weather**: Wind speed, temperature, precipitation, cloud cover
- **Target**: Binary classification (0 = Success, 1 = Failure)



## 📝 API Input Selection

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| month | int | 1-12 | Month of launch |
| day_of_week | int | 0-6 | Day of week (0=Monday) |
| year | int | 2000-2030 | Year of launch |
| wind_speed | float | ≥0 | Wind speed in m/s |
| temperature | float | any | Temperature in Celsius |
| precipitation | float | ≥0 | Precipitation in mm |
| cloud_cover | float | 0-100 | Cloud cover percentage |

## API Example Request
```json
{
    "month": 6,
    "day_of_week": 2,
    "year": 2024,
    "wind_speed": 12.5,
    "temperature": 28.0,
    "precipitation": 0.0,
    "cloud_cover": 25.0
}
```
## API Response

```json
{
  "prediction": "Success",
  "probability_of_failure": 0.087,
  "probability_of_success": 0.913,
  "confidence": 0.913,
  "risk_level": "Low",
  "input_features": {
    "month": 6,
    "day_of_week": 2,
    "year": 2024,
    "wind_speed": 12.5,
    "temperature": 28.0,
    "precipitation": 0.0,
    "cloud_cover": 25.0
  }
}
```
