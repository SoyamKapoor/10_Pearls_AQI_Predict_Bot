import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pytz

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT = "27.7244"
LON = "68.8228"
CSV_PATH = "data/raw_data.csv"

# Adjust date range as needed (max 1 year via student plan)
START_DATE = datetime(2024, 10, 1)
END_DATE = datetime(2025, 10, 22)

def fetch_historical_aqi(dt):
    """Fetch historical AQI for a specific timestamp"""
    start = int(dt.timestamp())
    end = int((dt + timedelta(hours=1)).timestamp())
    url = (f"http://api.openweathermap.org/data/2.5/air_pollution/history"
           f"?lat={LAT}&lon={LON}&start={start}&end={end}&appid={API_KEY}")

    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()

        # Check if data exists
        if "list" in data and data["list"]:
            row = data["list"][0]
            components = row["components"]

            # Time conversions
            pk_tz = pytz.timezone('Asia/Karachi')
            fetched_at = datetime.now(pytz.utc).astimezone(pk_tz).strftime('%Y-%m-%d %H:%M:%S')
            dt_local = datetime.fromtimestamp(row["dt"], tz=pytz.utc).astimezone(pk_tz).strftime('%Y-%m-%d %H:%M:%S')

            # Structure record
            record = {
                "fetched_at": fetched_at,
                "dt": row["dt"],
                "dt_readable": dt_local,
                "coord_lat": data["coord"]["lat"],
                "coord_lon": data["coord"]["lon"],
                "main_aqi": row["main"]["aqi"],
                "co": components["co"],
                "no": components["no"],
                "no2": components["no2"],
                "o3": components["o3"],
                "so2": components["so2"],
                "pm2_5": components["pm2_5"],
                "pm10": components["pm10"],
                "nh3": components["nh3"]
            }
            return record
    except Exception as e:
        print(f"Error fetching data for {dt}: {e}")
    return None


def backfill_to_csv(start_date, end_date, csv_path):
    """Fetch past AQI records, fill missing hours, and store in CSV"""
    current = start_date
    records = []

    # Step 1: Fetch day-by-day data
    print(f"Fetching AQI data from {start_date.date()} to {end_date.date()} ...")
    while current <= end_date:
        rec = fetch_historical_aqi(current)
        if rec:
            records.append(rec)
        current += timedelta(days=1)

    if not records:
        print("No historical data retrieved. Check API limits or plan restrictions.")
        return

    # Step 2: Create DataFrame
    df = pd.DataFrame(records, columns=[
        "fetched_at", "dt", "dt_readable", "coord_lat", "coord_lon",
        "main_aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"
    ])
    df['dt_readable'] = pd.to_datetime(df['dt_readable'])
    df = df.sort_values('dt_readable').drop_duplicates(subset=['dt_readable'])

    # Step 3: Resample hourly + Interpolate missing hours
    df = df.set_index('dt_readable').resample('H').interpolate().reset_index()
    print(f"After resampling & interpolation: {len(df)} hourly rows.")

    # Step 4: Save/append to CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
        print(f"Created new CSV file: {csv_path}")
    else:
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, df]).drop_duplicates(subset=["dt_readable"])
        combined.to_csv(csv_path, index=False)
        print(f"Appended new data; total rows now: {len(combined)}")

    print("✅ Backfill completed successfully!")


if __name__ == "__main__":
    if not API_KEY:
        raise ValueError("OPENWEATHER_API_KEY not found. Please set it in your .env file.")
    backfill_to_csv(START_DATE, END_DATE, CSV_PATH)