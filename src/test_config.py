from utils import load_config, print_banner

print_banner()

config = load_config()

print("\nLoaded Configuration:")
print(config)

print("\nFeature group name:", config['feature_store']['group_name'])
print("Model name:", config['model_registry']['model_name'])
print("City:", config['city'])
print("Raw data path:", config['data_paths']['raw'])