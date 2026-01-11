# ID2223-Project
# 🚲 Citibike Intelligent Assistant  
*Real-time bike demand prediction + RAG + LLM Chat UI*

> Course project for **ID2223 – Scalable Machine Learning and Deep Learning Systems** @ KTH.

You can use our UI here: https://huggingface.co/spaces/yunquan01/ID2223-Project

---

## 1. Project Overview

This repo contains the code for my end-to-end **Citibike intelligent assistant**:

- Pulls **real-time Citibike station status** from the official GBFS API.
- Uses **Hopsworks Feature Store** (offline + online feature groups) and a **Random Forest model with lag features** to predict **next-hour demand for each station**.
- Combines historical model predictions with **current bikes / docks availability** into an **ensemble prediction**.
- Builds a **RAG (Retrieval-Augmented Generation)** index over the latest prediction CSV (plus optional documentation).
- Wraps everything in a **chat UI** powered by an LLM (OpenAI), deployed as a **Hugging Face Space**.

Users can ask questions such as:

- *“Where is the best place to rent a bike in Manhattan in the next hour?”*  
- *“I’m near Greenpoint – which nearby stations will likely have bikes available?”*  
- *“Which stations will be the busiest in the next hour?”*

…and get answers grounded in the **latest predictions + external docs**, not just the LLM’s prior knowledge.

---

## 2. System Architecture

### 2.1 Data & Feature Store

1. **Historical trip data (offline)**  
   - Source: Citibike monthly tripdata CSV (e.g., `202509-citibike-tripdata_1.csv`).  
   - Processed into an **offline feature group** in Hopsworks: `citibike_hourly_station` (v2).  
   - Main columns:
     - `start_station_id`
     - `started_hour` / `event_time`
     - `hour`, `dow` (day of week), `is_weekend`
     - `lag_1h`, `lag_24h` (previous demand)
     - `trips_per_hour` (label)

2. **Real-time station status (online)**  
   - Source: Citibike GBFS `station_status.json`.  
   - Ingested into an **online feature group** in Hopsworks:  
     `citibike_hourly_station_online` (v1).  
   - Main columns:
     - `start_station_id` (GBFS station id, later mapped to offline numeric id)
     - `event_time`
     - `hour`, `dow`, `is_weekend`
     - `num_bikes_available`, `num_docks_available`

3. **Station id mapping**  
   - Script `build_station_id_mapping.py` builds a mapping:
     - `start_station_id` (offline numeric id from tripdata)
     - `gbfs_id` (online UUID from GBFS)
   - Saved as `station_id_mapping.csv`, used to join online and offline worlds.

---

### 2.2 Demand Prediction Model

The core prediction model is a **Random Forest Regressor** trained on **5 historical features**:

```text
hour, dow, is_weekend, lag_1h, lag_24h  ->  trips_per_hour
