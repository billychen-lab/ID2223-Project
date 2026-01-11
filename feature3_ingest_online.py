# feature3_ingest_online.py
# Function: Fetch station_status from CitiBike's GBFS real-time endpoint,
#           construct a batch of features, and write them into Hopsworks
#           feature group: citibike_hourly_station_online (version=1)

import os
import requests
import pandas as pd
import hopsworks as hs


# --------- 1. Configuration ---------

# CitiBike GBFS station_status endpoint
GBFS_STATUS_URL = "https://gbfs.citibikenyc.com/gbfs/en/station_status.json"

# Your Hopsworks project name (optional; usually the default project is selected during login)
PROJECT_NAME = "yunquanlab"

# Online FG name / version
ONLINE_FG_NAME = "citibike_hourly_station_online"
ONLINE_FG_VERSION = 1


# --------- 2. Fetch GBFS real-time data and construct a DataFrame ---------

def fetch_gbfs_snapshot() -> pd.DataFrame:
    """Fetch a station_status snapshot from GBFS and convert it into a DataFrame matching the online FG."""

    resp = requests.get(GBFS_STATUS_URL, timeout=10)
    resp.raise_for_status()
    payload = resp.json()

    stations = payload["data"]["stations"]

    # Keep only the fields we need
    df = pd.DataFrame(stations)[
        ["station_id", "num_bikes_available", "num_docks_available"]
    ]

    # Current time (UTC), floored to the hour
    now = pd.Timestamp.utcnow().floor("H")

    df["event_time"] = now
    df["hour"] = now.hour
    df["dow"] = now.dayofweek  # Monday=0, Sunday=6
    df["is_weekend"] = df["dow"].isin([5, 6]).astype("int32")

    # Align with FG schema: station_id -> start_station_id
    df = df.rename(columns={"station_id": "start_station_id"})

    # Reorder columns to match the FG schema
    df = df[
        [
            "start_station_id",
            "event_time",
            "hour",
            "dow",
            "is_weekend",
            "num_bikes_available",
            "num_docks_available",
        ]
    ]

    # --------- 3. Cast dtypes to match the FG schema ---------
    # Schema reference (as shown in the Hopsworks UI):
    # start_station_id : string/varchar
    # event_time       : timestamp
    # hour             : bigint   (=> pandas int64)
    # dow              : bigint   (=> pandas int64)
    # is_weekend       : int      (=> pandas int32)
    # num_bikes_available : int   (=> pandas int32)
    # num_docks_available : int   (=> pandas int32)

    df["start_station_id"] = df["start_station_id"].astype(str)
    df["hour"] = df["hour"].astype("int64")
    df["dow"] = df["dow"].astype("int64")

    df["is_weekend"] = df["is_weekend"].astype("int32")
    df["num_bikes_available"] = df["num_bikes_available"].fillna(0).astype("int32")
    df["num_docks_available"] = df["num_docks_available"].fillna(0).astype("int32")

    print("Sample of fetched GBFS data:")
    print(df.head())
    print("\nDtypes:\n", df.dtypes)

    return df


# --------- 4. Log in to Hopsworks and write to the online FG ---------

def main():
    # Requires the environment variable HOPSWORKS_API_KEY (which you have already set)
    project = hs.login(
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        host="c.app.hopsworks.ai",
        project=PROJECT_NAME,
    )

    fs = project.get_feature_store()

    # Get the online feature group you created
    fg = fs.get_feature_group(
        name=ONLINE_FG_NAME,
        version=ONLINE_FG_VERSION,
    )

    # Fetch a real-time snapshot
    online_df = fetch_gbfs_snapshot()

    # Insert into the FG (writes offline+online depending on FG configuration)
    # If you don't want to block, you can pass write_options={"wait_for_job": False}
    fg.insert(online_df)

    print(
        f"\n✅ Inserted {len(online_df)} rows into "
        f"'{ONLINE_FG_NAME}' v{ONLINE_FG_VERSION}."
    )


if __name__ == "__main__":
    main()
