import pandas as pd
import joblib
import shap

def explain():
    df = pd.read_csv("data/processed_data.csv")
    X = df.drop(columns=['aqi'], errors='ignore')
    model = joblib.load("model_registry/saved_models/aqi_model.pkl")
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, show=False)
    shap.save_html("docs/shap_summary.html", shap.summary_plot(shap_values, X, show=False))
    print("Feature importance plot saved in docs/.")

if __name__ == "__main__":
    explain()