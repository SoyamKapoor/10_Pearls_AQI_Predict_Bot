import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from utils import print_banner

def analyze_feature_importance():
    print_banner()
    print("\n🔍 Analyzing Feature Importance with SHAP...\n")

    # Load trained model
    model = joblib.load("models/best_aqi_model.pkl")
    
    # Load cleaned data
    df = pd.read_csv("data/cleaned_data.csv")
    
    # Prepare features (same as training)
    X = df.drop(columns=['aqi', 'fetched_at', 'dt_readable'], errors='ignore')
    
    # Handle categorical encoding if needed
    from sklearn.preprocessing import LabelEncoder
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    # Sample data for SHAP (use 100 samples for speed)
    X_sample = X.sample(min(100, len(X)), random_state=42)
    
    print(f"Computing SHAP values for {len(X_sample)} samples...")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # --- Visualization 1: Summary Plot ---
    print("\n📊 Generating SHAP Summary Plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches='tight')
    print("✅ Saved: shap_summary_plot.png")
    plt.close()
    
    # --- Visualization 2: Feature Importance Bar Plot ---
    print("\n📊 Generating Feature Importance Bar Plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("shap_feature_importance.png", dpi=300, bbox_inches='tight')
    print("✅ Saved: shap_feature_importance.png")
    plt.close()
    
    # --- Get Top Features ---
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    print("\n🏆 Top 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Save to file
    feature_importance.to_csv("feature_importance_shap.csv", index=False)
    print("\n✅ Feature importance saved to: feature_importance_shap.csv")
    
    print("\n✅ SHAP Analysis Complete!")

if __name__ == "__main__":
    analyze_feature_importance()