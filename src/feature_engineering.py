import pandas as pd
import hopsworks
import os
from dotenv import load_dotenv
from datetime import datetime
from utils import print_banner, load_config

def compute_features():
    print_banner()

    # --- Load configuration ---
    config = load_config()
    input_file = config['data_paths']['raw']
    output_file = config['data_paths']['cleaned']
    group_name = config['feature_store']['group_name']
    group_version = config['feature_store']['version']
    description = config['feature_store']['description']
    primary_key = config['feature_store']['primary_key']

    # --- Load and verify data ---
    print(f"\nLoading raw data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Rows loaded: {len(df)}")

    if 'main_aqi' not in df.columns:
        raise ValueError("Missing 'main_aqi' column in raw_data.csv")

    # --- Convert timestamps ---
    df['dt_readable'] = pd.to_datetime(df['dt_readable'], errors='coerce')
    df = df.dropna(subset=['dt_readable']).sort_values('dt_readable')

    # --- Fill missing numeric values ---
    pollutants = ['co','no','no2','o3','so2','pm2_5','pm10','nh3']
    print("Filling missing pollutant data...")
    for col in pollutants:
        df[col] = pd.to_numeric(df[col], errors='coerce').ffill().bfill()

    # --- AQI Target ---
    df['aqi'] = pd.to_numeric(df['main_aqi'], errors='coerce')
    df = df.drop(columns=['main_aqi'], errors='ignore')

    # --- Time-based features ---
    print("Generating time-based features...")
    df['hour'] = df['dt_readable'].dt.hour
    df['day_of_week'] = df['dt_readable'].dt.day_name()
    df['month'] = df['dt_readable'].dt.month
    df['year'] = df['dt_readable'].dt.year
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)

    # --- Rolling averages ---
    print("Computing rolling averages...")
    df = df.set_index('dt_readable')
    for col in pollutants + ['aqi']:
        df[f'{col}_rolling3'] = df[col].rolling(3, min_periods=1).mean()
        df[f'{col}_rolling6'] = df[col].rolling(6, min_periods=1).mean()
        df[f'{col}_rolling12'] = df[col].rolling(12, min_periods=1).mean()
    df = df.reset_index()

    # --- Clean column names ---
    df.columns = (
        df.columns
        .str.replace('.', '_', regex=False)
        .str.replace('-', '_', regex=False)
        .str.lower()
    )

    # --- Fix for Hopsworks primary key + NaN string issue ---
    df['dt_readable'] = df['dt_readable'].astype(str)

    if 'fetched_at' in df.columns:
        df['fetched_at'] = df['fetched_at'].fillna("unknown").astype(str)
    else:
        # Create it if missing (safety guard)
        df['fetched_at'] = "unknown"

    # --- Save cleaned data locally ---
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\n✅ Cleaned dataset saved at: {output_file} ({len(df)} rows)")

    # --- Upload to Hopsworks ---
    print("\nConnecting to Hopsworks...")
    load_dotenv()

    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()

    print(f"Creating or updating feature group '{group_name}' (v{group_version}) on Hopsworks...")
    fg = fs.get_or_create_feature_group(
        name=group_name,
        version=group_version,
        description=description,
        primary_key=[primary_key],
        online_enabled=True
    )

    print("Uploading dataset to Hopsworks Feature Store...")
    fg.insert(df)
    print(f"\n✅ Features successfully inserted into '{group_name}' (version {group_version}).")


if __name__ == "__main__":
    compute_features()