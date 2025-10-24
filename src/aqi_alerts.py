import pandas as pd
import json
from datetime import datetime
from utils import print_banner

# AQI Categories (US EPA Standard)
AQI_CATEGORIES = {
    'Good': (0, 50),
    'Moderate': (51, 100),
    'Unhealthy for Sensitive Groups': (101, 150),
    'Unhealthy': (151, 200),
    'Very Unhealthy': (201, 300),
    'Hazardous': (301, 500)
}

def get_aqi_category(aqi_value):
    """Determine AQI category based on value"""
    for category, (low, high) in AQI_CATEGORIES.items():
        if low <= aqi_value <= high:
            return category
    return "Hazardous"  # Above 500

def check_alerts():
    print_banner()
    print("\n🚨 AQI Alert System - Checking Current Conditions...\n")
    
    # Load latest data
    df = pd.read_csv("data/cleaned_data.csv")
    df['dt_readable'] = pd.to_datetime(df['dt_readable'])
    
    # Get latest reading
    latest = df.sort_values('dt_readable').iloc[-1]
    current_aqi = latest['aqi']
    timestamp = latest['dt_readable']
    
    # Determine category
    category = get_aqi_category(current_aqi)
    
    print(f"📅 Timestamp: {timestamp}")
    print(f"📊 Current AQI: {current_aqi:.1f}")
    print(f"🏷️  Category: {category}")
    print("─" * 60)
    
    # Alert logic
    if current_aqi <= 50:
        print("✅ AIR QUALITY: GOOD")
        print("   Air quality is satisfactory, and air pollution poses little or no risk.")
        alert_level = "none"
        
    elif current_aqi <= 100:
        print("⚠️  AIR QUALITY: MODERATE")
        print("   Air quality is acceptable. However, there may be a risk for some people,")
        print("   particularly those who are unusually sensitive to air pollution.")
        alert_level = "low"
        
    elif current_aqi <= 150:
        print("🟡 ALERT: UNHEALTHY FOR SENSITIVE GROUPS")
        print("   Members of sensitive groups may experience health effects.")
        print("   The general public is less likely to be affected.")
        print("\n   👤 Affected: Children, elderly, people with respiratory conditions")
        alert_level = "medium"
        
    elif current_aqi <= 200:
        print("🟠 ALERT: UNHEALTHY")
        print("   Some members of the general public may experience health effects;")
        print("   members of sensitive groups may experience more serious health effects.")
        print("\n   🚨 Recommendation: Reduce prolonged outdoor exertion")
        alert_level = "high"
        
    elif current_aqi <= 300:
        print("🔴 ALERT: VERY UNHEALTHY")
        print("   Health alert: The risk of health effects is increased for everyone.")
        print("\n   🚨 Recommendation: Avoid outdoor activities")
        alert_level = "very_high"
        
    else:
        print("🆘 HAZARDOUS ALERT!")
        print("   Health warning of emergency conditions: everyone is more likely to be affected.")
        print("\n   🚨 URGENT: Stay indoors and keep windows closed")
        alert_level = "hazardous"
    
    # Save alert to log
    alert_log = {
        "timestamp": str(timestamp),
        "aqi": float(current_aqi),
        "category": category,
        "alert_level": alert_level
    }
    
    with open("alert_log.json", 'w') as f:
        json.dump(alert_log, f, indent=4)
    
    print("\n✅ Alert logged to: alert_log.json")
    
    return alert_log

if __name__ == "__main__":
    check_alerts()
