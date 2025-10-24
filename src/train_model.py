import hopsworks
import os
import pandas as pd
from dotenv import load_dotenv
from utils import print_banner, load_config
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np


def train_and_register_model():
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
        print("\nReading data using Arrow Flight (fast path)...")
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
    print(f"\nFinal training columns ({len(X.columns)}): {list(X.columns[:10])}... (truncated)")

    # --- Split into train/test sets ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Train and evaluate multiple models ---
    print("\n" + "="*60)
    print("Training and evaluating multiple models...")
    print("="*60)

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42)
    }

    results = {}

    for model_name, model in models.items():
        print(f"\n🔹 Training {model_name}...")
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)

        # Evaluate
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results[model_name] = {
            "model": model,
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2
        }

        print(f"   MAE:  {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   R²:   {r2:.4f}")

    # --- Select the best model (lowest MAE) ---
    print("\n" + "="*60)
    print("Model Comparison Summary:")
    print("="*60)
    print(f"{'Model':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
    print("-"*60)
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['MAE']:<10.4f} {metrics['RMSE']:<10.4f} {metrics['R²']:<10.4f}")

    best_model_name = min(results, key=lambda x: results[x]['MAE'])
    best_model = results[best_model_name]["model"]
    best_mae = results[best_model_name]["MAE"]
    best_rmse = results[best_model_name]["RMSE"]
    best_r2 = results[best_model_name]["R²"]

    print("\n" + "="*60)
    print(f"✅ Best Model: {best_model_name}")
    print(f"   MAE:  {best_mae:.4f}")
    print(f"   RMSE: {best_rmse:.4f}")
    print(f"   R²:   {best_r2:.4f}")
    print("="*60)

    # --- Save the best model locally ---
    os.makedirs("models", exist_ok=True)
    model_path = "models/best_aqi_model.pkl"
    joblib.dump(best_model, model_path)
    
    # Save metadata
    metadata = {
        "model_name": best_model_name,
        "MAE": best_mae,
        "RMSE": best_rmse,
        "R²": best_r2,
        "features": list(X.columns)
    }
    metadata_path = "models/best_aqi_model_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"\n✅ Best model saved at: {model_path}")
    print(f"✅ Metadata saved at: {metadata_path}")
    print(f"\n💡 To load later: model = joblib.load('{model_path}')")


if __name__ == "__main__":
    train_and_register_model()