import requests
import os
import pandas as pd
from datetime import datetime
import pytz
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT = "27.7244"  # Sukkur latitude
LON = "68.8228"  # Sukkur longitude

def fetch_current_aqi():
    """Fetch current air quality data from OpenWeatherMap API"""
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
    
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        
        # Extract info correctly based on actual JSON structure
        measurement = data['list'][0]
        coord_lat = data['coord']['lat']
        coord_lon = data['coord']['lon']
        
        # Convert UTC time to Pakistan time zone
        utc_now = datetime.now(pytz.utc)
        pk_tz = pytz.timezone('Asia/Karachi')
        local_now = utc_now.astimezone(pk_tz)
        
        dt_utc = datetime.fromtimestamp(measurement['dt'], tz=pytz.utc)
        dt_local = dt_utc.astimezone(pk_tz)
        
        # Prepare single structured record
        record = {
            'fetched_at': local_now.strftime('%Y-%m-%d %H:%M:%S'),
            'dt': measurement['dt'],
            'dt_readable': dt_local.strftime('%Y-%m-%d %H:%M:%S'),
            'coord_lat': coord_lat,
            'coord_lon': coord_lon,
            'main_aqi': measurement['main']['aqi'],
            'co': measurement['components']['co'],
            'no': measurement['components']['no'],
            'no2': measurement['components']['no2'],
            'o3': measurement['components']['o3'],
            'so2': measurement['components']['so2'],
            'pm2_5': measurement['components']['pm2_5'],
            'pm10': measurement['components']['pm10'],
            'nh3': measurement['components']['nh3']
        }
        
        df = pd.DataFrame([record])
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except KeyError as e:
        print(f"Error parsing response: {e}")
        print("Response data:", data)
        return None

def append_to_csv(df, csv_path):
    """Append new data to CSV or create file if missing"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
        print(f"Created new CSV: {csv_path}")
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)
        print(f"Appended new data to: {csv_path}")

if __name__ == "__main__":
    if not API_KEY:
        raise ValueError("OPENWEATHER_API_KEY not found. Please set it in your .env file.")
    
    print(f"API Key loaded: {API_KEY[:8]}...")
    print(f"Fetching AQI data for coordinates: {LAT}, {LON} (Sukkur)")
    
    df = fetch_current_aqi()
    
    if df is not None and not df.empty:
        csv_path = "data/raw_data.csv"
        append_to_csv(df, csv_path)
        
        print("\nLatest data fetched successfully:")
        print(df.to_string(index=False))
    else:
        print("Failed to fetch data. Please check your API key or network connection.")