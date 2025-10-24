import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import print_banner
import joblib
import os
import json

def train_mlp_model():
    print_banner()
    print("\n🧠 Training Neural Network (MLP) for Deep Learning...\n")
    
    # Load data
    df = pd.read_csv("data/cleaned_data.csv")
    print(f"✅ Loaded {len(df)} rows")
    
    # Prepare features and target
    target_col = "aqi"
    X = df.drop(columns=[target_col, "fetched_at", "dt_readable"], errors='ignore')
    y = df[target_col]
    
    # Encode categorical columns
    non_numeric_cols = X.select_dtypes(include=['object']).columns.tolist()
    if non_numeric_cols:
        print(f"\nEncoding categorical columns: {non_numeric_cols}")
        for col in non_numeric_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
    
    print(f"\nFinal feature count: {X.shape[1]}")
    print(f"Feature columns: {list(X.columns[:10])}... (truncated)")
    
    # Normalize features (important for neural networks)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Build Multi-Layer Perceptron Neural Network
    print("\n" + "="*60)
    print("🔹 Neural Network Architecture:")
    print("   Input Layer:  {} features".format(X.shape[1]))
    print("   Hidden Layer 1: 64 neurons (ReLU)")
    print("   Hidden Layer 2: 32 neurons (ReLU)")
    print("   Hidden Layer 3: 16 neurons (ReLU)")
    print("   Output Layer: 1 neuron (Linear)")
    print("="*60)
    
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),  # 3 hidden layers
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=True
    )
    
    print("\n🔥 Training Neural Network (this may take 1-2 minutes)...\n")
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\n📈 Evaluating on test set...")
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print("\n" + "="*60)
    print("✅ Neural Network Training Complete!")
    print("="*60)
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²:   {r2:.4f}")
    print(f"   Accuracy: {r2 * 100:.2f}%")
    print("="*60)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/mlp_aqi_model.pkl"
    scaler_path = "models/mlp_scaler.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Save metadata
    metadata = {
        "model_name": "Multi-Layer Perceptron (Neural Network)",
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R²": float(r2),
        "accuracy_percentage": float(r2 * 100),
        "architecture": "64-32-16 neurons",
        "features": list(X.columns),
        "trained_at": pd.Timestamp.now().isoformat()
    }
    
    metadata_path = "models/mlp_aqi_model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\n✅ Neural Network model saved: {model_path}")
    print(f"✅ Scaler saved: {scaler_path}")
    print(f"✅ Metadata saved: {metadata_path}")
    print(f"\n💡 To load later: model = joblib.load('{model_path}')")

if __name__ == "__main__":
    train_mlp_model()