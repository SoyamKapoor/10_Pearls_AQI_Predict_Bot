import joblib

def save_model(model, path="model_registry/saved_models/aqi_model.pkl"):
    joblib.dump(model, path)

def load_model(path="model_registry/saved_models/aqi_model.pkl"):
    return joblib.load(path)