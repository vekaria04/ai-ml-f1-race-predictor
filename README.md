# 🏁 AI/ML F1 Race Predictor (2025 Season)

This project uses real-world F1 data to train a machine learning model that predicts the finishing positions of drivers in the 2025 Formula 1 season. It leverages the [FastF1](https://theoehrly.github.io/Fast-F1/) library for data access and a RandomForest classifier for predictions.

## 🔍 Features

- Loads official race and qualifying data using FastF1
- Uses grid position, qualifying position, average lap pace, and weather stats
- Predicts race results for upcoming or hypothetical Grands Prix
- Caches sessions to allow offline predictions

## 📦 Installation

```bash
git clone https://github.com/vekaria04/ai-ml-f1-race-predictor.git
cd ai-ml-f1-race-predictor
pip install -r requirements.txt
```

## 🚀 How to Use

### 1. Train the model
Train on past races from the 2025 season:
```bash
python train_model.py
```
### 2. Predict an upcoming race
Predict results for the Canadian GP using existing cache or manual data:
```bash
python predict.py
```
❗ Make sure you've run some past races (e.g. Australia, Miami) to populate the f1_cache folder before predicting.

## 🧠 Model Details
- RandomForestClassifier with 300 estimators
  
Features:
  - Driver
  - Grid Position
  - Qualifying Position
  - Average Lap Time
  - Weather (Track & Air Temp, Humidity)

## 🗃 File Structure
- load_data.py – Extracts and prepares features from FastF1
- train_model.py – Trains and evaluates the ML model
- predict.py – Runs prediction on upcoming races
- f1_cache/ – Stores downloaded race data
- models/ – Contains trained model and scalers

## 🏎 Example Output
```bash
Predicted Results for Canada 2025:
 Abbreviation  GridPosition  PredictedPosition
          VER             1                  1
          LEC             3                  2
          NOR             2                  3
```

## 📌 Requirements
Python 3.10+
pandas, scikit-learn, joblib, FastF1

Install with:
```bash
pip install -r requirements.txt
```
