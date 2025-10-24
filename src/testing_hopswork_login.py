import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()
print("Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
print("✅ Logged into project:", project.name)