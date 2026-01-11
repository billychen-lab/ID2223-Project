# feature6_predict_next_hour_with_lag_all_stations.py
# Use the "time + lag_1h + lag_24h" model to predict next-hour trips_per_hour for all stations
# Steps:
#   1) Call feature3_ingest_online.py to fetch real-time GBFS status and write to the online FG
#   2) Read the latest one-hour records (one row per station) with lag features from the offline FG
#   3) Merge the real-time num_bikes_available / num_docks_available from the online FG
#   4) Load station_id -> station_name mapping from the raw tripdata CSV and add it to the results
#   5) Output to CSV and print the top N results in the terminal

import os
import hopsworks
import joblib
import pandas as pd
import numpy as np

from huggingface_hub import hf_hub_download
import streamlit as st

from feature3_ingest_online import main as ingest_online


@st.cache_resource
def load_rf_model():
    path = hf_hub_download(
        repo_id="yunquan01/ID2223-assets",
        repo_type="dataset",
        filename="models/citibike_rf_with_lag_v1.joblib",
    )
    return joblib.load(path)


MAPPING_CSV = "station_id_mapping.csv"  # File generated in step 1
PROJECT = "yunquanlab"

OFFLINE_FG_NAME = "citibike_hourly_station"
OFFLINE_FG_VERSION = 2

ONLINE_FG_NAME = "citibike_hourly_station_online"
ONLINE_FG_VERSION = 1

# MODEL_PATH = "models/citibike_rf_with_lag_v1.joblib"

FEATURE_COLS = ["hour", "dow", "is_weekend", "lag_1h", "lag_24h"]
TARGET_COL = "trips_per_hour"

# OUTPUT_CSV = "predictions_with_lag_next_hour_all_stations.csv"
OUTPUT_CSV = os.path.join("/tmp", "predictions_with_lag_next_hour_all_stations.csv")
# ❗Change this path to the real location of your raw tripdata CSV on your own machine
# TRIPDATA_CSV = os.path.join("D:\ID2223\project\citibike-tripdata_1.csv")
TRIPDATA_CSV = "citibike-tripdata_1.csv"


def get_online_features(fs):
    """
    Read the latest hour of num_bikes_available / num_docks_available from the online FG,
    then map GBFS UUIDs to the offline FG's numeric start_station_id using station_id_mapping.csv.
    The returned DataFrame contains only three columns:
        start_station_id (string form of the numeric station id, matching the offline FG)
        num_bikes_available
        num_docks_available
    """

    # 1. Read online FG
    fg_online = fs.get_feature_group(ONLINE_FG_NAME, version=ONLINE_FG_VERSION)
    online_df = fg_online.read()  # Here start_station_id is the GBFS UUID string

    # Keep only the latest hour batch
    if "event_time" in online_df.columns:
        online_df["event_time"] = pd.to_datetime(online_df["event_time"])
        latest_t = online_df["event_time"].max()
        online_df = online_df[online_df["event_time"] == latest_t].copy()

    # 2. Read the mapping table we generated earlier (offline numeric id <-> GBFS UUID)
    mapping = pd.read_csv(
        MAPPING_CSV,
        dtype={"start_station_id": str, "gbfs_id": str},  # start_station_id: offline numeric id
    )

    # online start_station_id is GBFS UUID; rename it to gbfs_id for joining
    online_df["gbfs_id"] = online_df["start_station_id"].astype(str)

    # 3. Join on gbfs_id to get the offline numeric station id
    joined = online_df.merge(
        mapping,  # contains gbfs_id + offline start_station_id
        on="gbfs_id",
        how="inner",
    )

    # joined now contains:
    #   - start_station_id_x : GBFS UUID
    #   - start_station_id_y : offline FG numeric station id (string)
    #   - num_bikes_available
    #   - num_docks_available
    #   - other columns (event_time, etc.)

    # 4. Keep only "offline numeric id + real-time inventory" columns
    online_features = joined[[
        "start_station_id_y",
        "num_bikes_available",
        "num_docks_available",
    ]].copy()

    # Rename start_station_id_y to start_station_id
    online_features = online_features.rename(
        columns={"start_station_id_y": "start_station_id"}
    )

    # Ensure consistent dtype with the offline FG (use string)
    online_features["start_station_id"] = (
        online_features["start_station_id"].astype(str)
    )

    # Final output columns:
    #   start_station_id (offline numeric id)
    #   num_bikes_available
    #   num_docks_available
    return online_features


def load_station_name_mapping():
    """
    Build a station_id -> station_name mapping from the raw tripdata CSV.
    If the file is not found, return None (results will not include station_name).
    """
    if not os.path.exists(TRIPDATA_CSV):
        print(f"⚠️  Raw tripdata CSV not found at {TRIPDATA_CSV}, "
              "so the prediction output will not include the station_name column.")
        return None

    print(f"Loading raw tripdata from: {TRIPDATA_CSV}")
    raw = pd.read_csv(
        TRIPDATA_CSV,
        usecols=["start_station_id", "start_station_name"],
    )

    raw = raw.dropna(subset=["start_station_id", "start_station_name"])

    # Cast to string to avoid inconsistencies like 6297 vs 6297.0
    raw["start_station_id"] = raw["start_station_id"].astype(str)

    # A station_id may have slightly different name spellings; take the most frequent one
    name_map = (
        raw.groupby("start_station_id")["start_station_name"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
    )

    print(f"Built station_name mapping for {len(name_map)} stations.")
    return name_map


def main():
    # --------------------------------------------------
    # 1. Run online ingestion once
    # --------------------------------------------------
    print("🔄 Running online ingestion (feature3_ingest_online)...")
    ingest_online()

    # --------------------------------------------------
    # 2. Log in to Hopsworks and read offline & online FGs
    # --------------------------------------------------

    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT", "yunquanlab"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    )
    fs = project.get_feature_store()

    # =============== 1. Read offline FG and take the last hour (with lags) ===============
    fg_offline = fs.get_feature_group(OFFLINE_FG_NAME, version=OFFLINE_FG_VERSION)
    df_offline = fg_offline.read()

    # Drop rows without lag features
    df_offline = df_offline.dropna(subset=["lag_1h", "lag_24h"])

    # Sort by time
    time_col = "started_hour" if "started_hour" in df_offline.columns else "event_time"
    df_offline = df_offline.sort_values(time_col)

    last_ts = df_offline[time_col].max()
    offline_last = df_offline[df_offline[time_col] == last_ts].copy()

    print(f"Offline last timestamp = {last_ts}, rows = {len(offline_last)}")

    # =============== 2. Load the 5-feature RF model and compute historical prediction ===============
    # model = joblib.load(MODEL_PATH)
    model = load_rf_model()

    FEATURE_COLS = ["hour", "dow", "is_weekend", "lag_1h", "lag_24h"]
    X_hist = offline_last[FEATURE_COLS]
    y_hist = model.predict(X_hist)

    # =============== 3. Read online FG and keep only the latest hour ===============
    fg_online = fs.get_feature_group(ONLINE_FG_NAME, version=ONLINE_FG_VERSION)
    online_df = fg_online.read()

    # Force event_time to pandas datetime
    online_df["event_time"] = pd.to_datetime(online_df["event_time"])
    online_last_ts = online_df["event_time"].max()
    online_last = online_df[online_df["event_time"] == online_last_ts].copy()

    print(f"Online last timestamp = {online_last_ts}, rows = {len(online_last)}")

    # =============== 4. Align start_station_id types and merge ===============
    offline_last["start_station_id"] = offline_last["start_station_id"].astype(str)
    # online_last["start_station_id"] = online_last["start_station_id"].astype(str)
    online_df = get_online_features(fs)

    merged = offline_last.merge(
        online_df,
        on="start_station_id",
        how="left",
    )

    print("Merged shape:", merged.shape)
    print(merged[["start_station_id", "num_bikes_available", "num_docks_available"]].head(10))

    # If missing, fill with 0 for easier downstream calculation
    merged["num_bikes_available"] = merged["num_bikes_available"].fillna(0).astype("float64")
    merged["num_docks_available"] = merged["num_docks_available"].fillna(0).astype("float64")

    # Check how many rows have non-zero inventory/capacity
    bikes = merged["num_bikes_available"]
    docks = merged["num_docks_available"]
    capacity = bikes + docks
    print("Number of rows with bikes > 0:", (bikes > 0).sum())
    print("Number of rows with docks > 0:", (docks > 0).sum())
    print("Number of rows with capacity > 0:", (capacity > 0).sum())

    # =============== 5. Compute real-time heuristic prediction y_rt + ensemble ===============
    # 5.1 Historical prediction is already available: y_hist
    # 5.2 Real-time: compute y_rt from occupancy ratio
    bikes = bikes.astype("float64")
    docks = docks.astype("float64")
    capacity = bikes + docks

    # Avoid division by zero
    capacity_safe = capacity.replace(0, np.nan)
    occ_ratio = bikes / capacity_safe  # r = bikes / (bikes + docks)

    print("Occupancy ratio examples:")
    print(occ_ratio.head())

    # Compute mean occupancy ratio on valid stations
    mean_r = occ_ratio[occ_ratio.notna()].mean()
    print("mean_r =", mean_r)

    if pd.isna(mean_r) or mean_r <= 0:
        # FG has data, but after merging, all stations for this hour still have capacity 0/NaN
        # so we ignore real-time signals this time and use only the historical model.
        print("⚠ Occupancy ratios are all NaN or 0. Ignoring real-time info and using the historical model only.")
        y_rt = np.zeros_like(y_hist)
        y_final = y_hist.copy()
    else:
        # Normal case: at least some stations have valid inventory info
        occ_ratio = occ_ratio.fillna(mean_r)

        mean_y_hist = y_hist.mean()
        k = mean_y_hist / (mean_r + 1e-6)
        print("scale k =", k)

        y_rt = k * occ_ratio
        # Ensemble
        y_final = 0.4 * y_hist + 0.6 * y_rt

    # =============== 6. Write back to merged for analysis/saving CSV ===============
    merged["predicted_trips_hist"] = y_hist
    merged["predicted_trips_realtime"] = y_rt
    merged["predicted_trips_ensemble"] = y_final

    name_map = load_station_name_mapping()
    if name_map is not None:
        merged = merged.merge(name_map, on="start_station_id", how="left")

    # Sort & print top 10
    top = merged.sort_values("predicted_trips_ensemble", ascending=False).head(10)
    print('\n🚲  Top 10 stations with the highest predicted next-hour demand:')
    print(
        top[
            [
                "start_station_id",
                "start_station_name",  # This column may be missing; comment out if unavailable
                "num_bikes_available",
                "num_docks_available",
                "lag_1h",
                "lag_24h",
                "predicted_trips_hist",
                "predicted_trips_realtime",
                "predicted_trips_ensemble",
            ]
        ]
    )

    # You can save merged / top to CSV here
    # output_csv = "predictions_with_lag_next_hour_all_stations_ensemble.csv"
    output_csv = os.path.join("/tmp", "predictions_with_lag_next_hour_all_stations_ensemble.csv")
    merged.to_csv(output_csv, index=False)
    print(f"\n✅ Saved predictions for ALL stations to: {output_csv}")


if __name__ == "__main__":
    main()