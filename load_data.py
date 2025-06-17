# load_data.py
# -------------
# Fetch per-driver features from a single race session
# – Race results (Grid, Finish position)
# – Average quick-lap pace
# – Session-average weather
# -----------------------------------------------------------------

import os
import fastf1
import pandas as pd

# Setup FastF1 cache
os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")

def load_race_features(year: int, gp_name: str) -> pd.DataFrame:
    """Return one row per driver with race + quali + pace + weather features."""

    try:
        race = fastf1.get_session(year, gp_name, "R")
        race.load()
        results = race.results[["Abbreviation", "GridPosition", "Position"]]
        print(f"  ✓ Race loaded: {len(results)} rows")
    except Exception as e:
        raise RuntimeError(f"{gp_name} – Race load failed: {e}")

    # ---------- QUALIFYING -------------------------------------------------
    try:
        quali = fastf1.get_session(year, gp_name, "Q")
        quali.load()

        q_laps = quali.laps.pick_quicklaps()

        # Fix for FutureWarning
        fastest_laps = (
            q_laps.groupby("Driver", group_keys=False)
            .apply(lambda df: df.nsmallest(1, "LapTime"))
            .reset_index(drop=True)
        )

        fastest_laps["Abbreviation"] = fastest_laps["Driver"].map(
            lambda d: quali.get_driver(d)["Abbreviation"]
        )

        quali_df = (
            fastest_laps
            .groupby("Abbreviation")["LapTime"]
            .min()
            .sort_values()
            .reset_index()
        )
        quali_df["QualiPos"] = range(1, len(quali_df) + 1)

        print(f"  ✓ Quali fastest laps: {len(fastest_laps)}")
        print(f"  ✓ Quali dataframe   : {len(quali_df)} rows")

    except Exception as e:
        raise RuntimeError(f"{gp_name} – Quali load failed: {e}")

    # ---------- RACE PACE --------------------------------------------------
    try:
        quick = race.laps.pick_quicklaps()
        avg_times = (
            quick.groupby("Driver")["LapTime"]
            .apply(lambda x: x.dt.total_seconds().mean())
            .reset_index()
        )

        drv_map = race.drivers
        avg_times["Abbreviation"] = avg_times["Driver"].map(
            lambda d: drv_map[d]["Abbreviation"] if d in drv_map else None
        )

        avg_df = avg_times[["Abbreviation", "LapTime"]].rename(columns={"LapTime": "AvgLapTime"})
        print(f"  ✓ Avg lap times: {len(avg_df)} rows")
    except Exception as e:
        raise RuntimeError(f"{gp_name} – AvgLapTime failed: {e}")

    # ---------- MERGE ------------------------------------------------------
    try:
        df = (
            results
            .merge(quali_df, on="Abbreviation", how="inner")
            .merge(avg_df, on="Abbreviation", how="inner")
        )
        print(f"  ✓ After merge: {len(df)} rows")
    except Exception as e:
        raise RuntimeError(f"{gp_name} – Merging failed: {e}")

    # ---------- WEATHER ----------------------------------------------------
    try:
        weather_means = race.weather_data.mean(numeric_only=True)
        for col in ["TrackTemperature", "AirTemperature", "Humidity", "Pressure", "WindSpeed", "WindDirection"]:
            df[col] = weather_means.get(col, 0)
    except Exception as e:
        raise RuntimeError(f"{gp_name} – Weather data failed: {e}")

    df["Race"] = gp_name
    df["Driver"] = df["Abbreviation"]
    print(f"  → Final rows: {len(df)}")
    return df
