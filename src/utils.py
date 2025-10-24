import yaml
import os

def print_banner():
    print("---- AQI Prediction System ----")

def load_config(config_path=None):
    """
    Loads the config YAML file from config/config.yaml
    so all scripts can use the same settings.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        config_path = os.path.abspath(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config