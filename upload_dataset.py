import chromadb
import pandas as pd
import os

# Initialize ChromaDB client
client = chromadb.Client()
collection = client.get_or_create_collection("energial_data")
 
# Dataset metadata and formatting
datasets = {
    "energy_consumption.csv": lambda row: f"Timestamp: {row['timestamp']}, Building_id: {row['building_id']},Energy_usage: {row['energy_usage']} kWh, Energy_source: {row['energy_source']} , Temperature: {row['temperature']},Humidity: {row['humidity']}, ",
    "energy_generation.csv": lambda row: f"Timestamp: {row['timestamp']} ,Energy_source: {row['energy_source']},,Energy_usage: {row['energy_usage']} kWh, Location: {row['location']} ",
    "weather.csv": lambda row: f"Timestamp: {row['timestamp']}, Location: {row['location']}, Temperature: {row['temperature']},Humidity: {row['humidity']}, Solar_irradiance: {row['solar_irradiance']}",
    "building_information.csv": lambda row: f"Building_id: {row['building_id']}, Building_name: {row['building_name']},Location: {row['location']}, Building_type: {row['building_type']},Square_footage:{row['square_footage']} "
}

for file, formatter in datasets.items():
    df = pd.read_csv(f"data/{file}")
    for i, row in df.iterrows():
        doc = formatter(row)
        collection.add(
            documents=[doc],
            ids=[f"{file}_{i}"]
        )

print("✅ All 4 datasets uploaded to ChromaDB.")
