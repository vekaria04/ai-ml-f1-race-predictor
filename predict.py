# predict.py
# ----------
# Predict finishing positions for a target Grand Prix using a model
# trained with train_model.py.  If FastF1 does not yet have quali/race
# data (because the session hasn't happened), the script lets you type
# the expected starting grid and fills in the missing columns with
# reasonable defaults so the model can still run.

import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from load_data import load_race_features

# --------------------------------------------------
# CONFIGURATION – change GP / YEAR if you like
# --------------------------------------------------
YEAR = 2025
GP_NAME = "Canada"          # FastF1 uses plain "Canada"
MODEL_DIR = Path("models")  # where train_model.py stored artefacts
# --------------------------------------------------

# ──────────────────────────────────────────────────
# 1.  Load model & transformers
# ──────────────────────────────────────────────────
try:
    model     = joblib.load(MODEL_DIR / "rf_model.pkl")
    scaler    = joblib.load(MODEL_DIR / "scaler.pkl")
    driver_le = joblib.load(MODEL_DIR / "driver_le.pkl")
    race_le   = joblib.load(MODEL_DIR / "race_le.pkl")
except FileNotFoundError as e:
    sys.exit(f"[ERROR] {e}\nHave you run train_model.py successfully?")

# ──────────────────────────────────────────────────
# 2.  Attempt to fetch real session-data via FastF1
# ──────────────────────────────────────────────────
print(f"\n▶ Loading session data for {GP_NAME} {YEAR} …")
try:
    df = load_race_features(YEAR, GP_NAME)
    fallback_mode = False
except Exception as exc:
    print(f"⚠  FastF1 could not provide quali/race laps: {exc}")
    print("   → Falling back to manual-grid mode.")
    fallback_mode = True

# ──────────────────────────────────────────────────
# 3A.  Fallback – let user enter the starting grid
# ──────────────────────────────────────────────────
if fallback_mode:
    # 3A-1.  Ask user for starting grid abbreviations
    grid_str = input(
        "\nEnter driver abbreviations in GRID ORDER "
        "(comma separated, e.g. 'VER,LEC,SAI,PER,…'):\n> "
    ).strip()
    abbr_list = [a.strip().upper() for a in grid_str.split(",") if a.strip()]
    if not abbr_list:
        sys.exit("No drivers entered – cannot make a prediction.")

    # 3A-2. Build minimal dataframe with required columns
    n = len(abbr_list)
    df = pd.DataFrame({
        "Abbreviation"     : abbr_list,
        "Driver"           : abbr_list,          # same key the encoder expects
        "GridPosition"     : range(1, n + 1),
        "QualiPos"         : range(1, n + 1),    # placeholder
        "AvgLapTime"       : np.nan,             # will be filled with mean
        "TrackTemperature" : np.nan,
        "AirTemperature"   : np.nan,
        "Humidity"         : np.nan,
        "Pressure"         : np.nan,
        "WindSpeed"        : np.nan,
        "WindDirection"    : np.nan,
        "Race"             : GP_NAME,
    })

    # 3A-3.  Fill numerical NaNs with means from the *training* scaler
    # (scaler.mean_ is on the same order of columns fed during training)
    scaler_feature_order = [
        "DriverEnc","GridPosition","QualiPos","AvgLapTime",
        "TrackTemperature","AirTemperature","Humidity","RaceEnc"
    ]
    # Map from feature-name to the mean that StandardScaler learned
    mean_lookup = dict(zip(scaler_feature_order, scaler.mean_))

    df["AvgLapTime"]       = mean_lookup["AvgLapTime"]
    df["TrackTemperature"] = mean_lookup["TrackTemperature"]
    df["AirTemperature"]   = mean_lookup["AirTemperature"]
    df["Humidity"]         = mean_lookup["Humidity"]

# ──────────────────────────────────────────────────
# 4.  Encode & scale features
# ──────────────────────────────────────────────────
try:
    df["DriverEnc"] = driver_le.transform(df["Driver"])
except ValueError as e:
    unknown = str(e).split(":")[-1].strip()
    sys.exit(f"[ERROR] Unknown driver abbreviation: {unknown}\n"
             "The model was trained on drivers it had already seen.\n"
             "Please use abbreviations present in the 2025 season.")

# Race label – if unseen, we append it to the encoder
if GP_NAME in race_le.classes_:
    df["RaceEnc"] = race_le.transform([GP_NAME] * len(df))
else:
    # append new label to encoder classes_ so transform works
    race_le.classes_ = np.append(race_le.classes_, GP_NAME)
    df["RaceEnc"] = race_le.transform([GP_NAME] * len(df))

X_cols = [
    "DriverEnc", "GridPosition", "QualiPos", "AvgLapTime",
    "TrackTemperature", "AirTemperature", "Humidity", "RaceEnc"
]
X_scaled = scaler.transform(df[X_cols])

# ──────────────────────────────────────────────────
# 5.  Predict finishing positions
# ──────────────────────────────────────────────────
df["PredictedPosition"] = model.predict(X_scaled)
df = df.sort_values("PredictedPosition")

# ──────────────────────────────────────────────────
# 6.  Display
# ──────────────────────────────────────────────────
print("\n────────────────────────────────────────────")
print(f"Predicted finishing order – {GP_NAME} {YEAR}")
print("────────────────────────────────────────────")
print(
    df[["PredictedPosition", "GridPosition", "Abbreviation"]]
      .to_string(index=False)
)
print("────────────────────────────────────────────")
