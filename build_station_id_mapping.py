# build_station_id_mapping.py
import pandas as pd
import requests

# 你的 tripdata CSV 路径（注意改成你自己的）
TRIPDATA_CSV = r"D:\ID2223\project\citibike-tripdata_1.csv"

# GBFS station information 接口（有名字 / 经纬度 / station_id）
STATION_INFO_URL = "https://gbfs.citibikenyc.com/gbfs/en/station_information.json"

def main():
    # 1) 读 tripdata，提取唯一的站点（老的数字 ID）
    trip = pd.read_csv(TRIPDATA_CSV)

    trip_stations = trip[
        ["start_station_id", "start_station_name", "start_lat", "start_lng"]
    ].dropna().drop_duplicates()

    # 保守一点，全转成字符串
    trip_stations["start_station_id"] = trip_stations["start_station_id"].astype(str)

    # 2) 拉 GBFS station_information（新 ID + 名字 + 经纬度）
    resp = requests.get(STATION_INFO_URL, timeout=10)
    resp.raise_for_status()
    stations = pd.DataFrame(resp.json()["data"]["stations"])

    # 只保留我们需要的字段
    stations = stations[["station_id", "name", "lat", "lon"]].rename(
        columns={"name": "gbfs_name", "lat": "start_lat", "lon": "start_lng"}
    )

    # 3) 用经纬度做近似匹配（先四舍五入到 4 位小数，减小浮点误差）
    trip_stations["lat_r"]  = trip_stations["start_lat"].round(4)
    trip_stations["lng_r"]  = trip_stations["start_lng"].round(4)
    stations["lat_r"]       = stations["start_lat"].round(4)
    stations["lng_r"]       = stations["start_lng"].round(4)

    mapping = pd.merge(
        trip_stations,
        stations,
        on=["lat_r", "lng_r"],
        how="inner",
        suffixes=("_offline", "_gbfs"),
    )

    mapping = mapping[[
        "start_station_id",        # offline 数字 ID（字符串形式）
        "start_station_name",      # tripdata 名字
        "gbfs_name",               # GBFS 名字（一般和上面一样）
        "station_id",              # GBFS 字符串 ID
    ]].rename(columns={"station_id": "gbfs_id"})

    print("Mapping rows:", len(mapping))
    print(mapping.head())

    mapping.to_csv("station_id_mapping.csv", index=False)
    print("\n✅ Saved mapping to station_id_mapping.csv")


if __name__ == "__main__":
    main()
