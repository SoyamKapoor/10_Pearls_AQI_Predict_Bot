import pandas as pd
import joblib

# Load processed features and trained model
df = pd.read_csv("data/processed_data.csv")
model = joblib.load("model_registry/saved_models/aqi_model.pkl")

# Use only numeric features for prediction
X = df.select_dtypes(include=['number']).drop(columns=["aqi"], errors='ignore')

# Predict AQI for all rows
preds = model.predict(X)

# Print actual vs predicted AQI values
results = pd.DataFrame({
    "Actual AQI": df["aqi"],
    "Predicted AQI": preds
})
print(results)