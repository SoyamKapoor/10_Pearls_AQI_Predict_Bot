import hopsworks
import os
import pandas as pd
from dotenv import load_dotenv
from utils import print_banner, load_config
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
import json


def train_best_model():
    print_banner()

    # --- Load config and environment ---
    config = load_config()
    load_dotenv()
    group_name = config['feature_store']['group_name']
    version = config['feature_store']['version']
    target_col = "aqi"

    # --- Connect to Hopsworks ---
    print("\nLogging into Hopsworks...")
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()

    # --- Fetch dataset (from offline Feature Group) ---
    print(f"\nFetching data from feature group '{group_name}' (v{version})...")
    fg = fs.get_feature_group(name=group_name, version=version)
    
    try:
        print("Reading data using Arrow Flight (fast path)...")
        df = fg.read()
    except Exception as e:
        print(f"⚠️ Arrow Flight timeout: {e}")
        print("➡️ Falling back to standard read...")
        df = fg.read(read_options={"use_arrow_flight": False})

    print(f"Loaded {len(df)} rows from '{group_name}' (v{version})")

    # --- Prepare features and target ---
    # Drop metadata and non-predictive columns
    X = df.drop(columns=[target_col, "fetched_at", "dt_readable"], errors='ignore')
    y = df[target_col]

    # Drop or encode non-numeric columns automatically
    non_numeric_cols = X.select_dtypes(include=['object']).columns.tolist()
    if non_numeric_cols:
        print(f"\nDetected non-numeric columns: {non_numeric_cols}")
        for col in non_numeric_cols:
            if X[col].nunique() <= 10:
                print(f"Encoding categorical column: {col}")
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
            else:
                print(f"Dropping non-numeric column: {col}")
                X = X.drop(columns=[col])

    # --- Final dataset check ---
    print(f"\nFinal training columns ({len(X.columns)}): {list(X.columns[:10])}...")

    # --- Split into train/test sets ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Train Random Forest (best known model) ---
    print("\n" + "="*60)
    print("🔹 Training Random Forest Regressor (best model)...")
    print("="*60)
    
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # --- Evaluate model performance ---
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n✅ Model Training Complete!")
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²:   {r2:.4f}")
    
    # Calculate accuracy percentage (R² as primary metric)
    accuracy_pct = r2 * 100
    print(f"   Accuracy: {accuracy_pct:.2f}%")

    # --- Save the model locally ---
    os.makedirs("models", exist_ok=True)
    model_path = "models/best_aqi_model.pkl"
    joblib.dump(model, model_path)
    
    # --- Save metadata ---
    metadata = {
        "model_name": "Random Forest",
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R²": float(r2),
        "accuracy_percentage": float(accuracy_pct),
        "n_estimators": 200,
        "features": list(X.columns),
        "trained_at": pd.Timestamp.now().isoformat()
    }
    metadata_path = "models/best_aqi_model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"\n✅ Model saved at: {model_path}")
    print(f"✅ Metadata saved at: {metadata_path}")
    print(f"\n💡 To load later: model = joblib.load('{model_path}')")
    print("="*60)


if __name__ == "__main__":
    train_best_model()