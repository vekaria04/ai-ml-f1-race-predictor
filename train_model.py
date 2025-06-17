import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from load_data import load_race_features

SEASON = 2025
RACES = [
    "Australian Grand Prix",
    "Chinese Grand Prix",
    "Japanese Grand Prix",
    "Bahrain Grand Prix",
    "Saudi Arabian Grand Prix",
    "Miami Grand Prix",
    "Emilia Romagna Grand Prix",
    "Monaco Grand Prix",
    "Spanish Grand Prix",
]

frames = []
print("Loading race data …")
for gp in RACES:
    try:
        frames.append(load_race_features(SEASON, gp))
    except Exception as e:
        print(f"  !! Skipping {gp}: {e}")

if not frames:
    raise RuntimeError("No usable race data — all races skipped.")

data = pd.concat(frames, ignore_index=True)
print(f"\nFinal dataset shape: {data.shape}")  # rows, cols

# ------------ Encoding --------------- #
driver_le = LabelEncoder().fit(data["Driver"])
race_le   = LabelEncoder().fit(data["Race"])

data["DriverEnc"] = driver_le.transform(data["Driver"])
data["RaceEnc"]   = race_le.transform(data["Race"])

X_cols = [
    "DriverEnc", "GridPosition", "AvgLapTime",
    "TrackTemperature", "AirTemperature", "Humidity", "Pressure",
    "WindSpeed", "WindDirection", "RaceEnc"
]

# Ensure all columns exist
missing = [col for col in X_cols if col not in data.columns]
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

X = data[X_cols]
y = data["Position"]

# ------------ Scaling & Model -------- #
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_tr, y_tr)

print("\nModel evaluation")
print("Accuracy:", accuracy_score(y_te, model.predict(X_te)))
print(classification_report(y_te, model.predict(X_te), digits=3))

# ------------ Save ------------------- #
os.makedirs("models", exist_ok=True)
joblib.dump(model,     "models/rf_model.pkl")
joblib.dump(scaler,    "models/scaler.pkl")
joblib.dump(driver_le, "models/driver_le.pkl")
joblib.dump(race_le,   "models/race_le.pkl")
print("Artifacts saved in /models/")
