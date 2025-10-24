import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("📊 Loading Data and Model Results...")

# Load cleaned data
df = pd.read_csv("data/cleaned_data.csv")
df['dt_readable'] = pd.to_datetime(df['dt_readable'])

# Load model metadata
with open("models/best_aqi_model_metadata.json", 'r') as f:
    metadata = json.load(f)

print(f"\n✅ Model: {metadata['model_name']}")
print(f"✅ Accuracy: {metadata['accuracy_percentage']:.2f}%")
print(f"✅ MAE: {metadata['MAE']:.4f}")
print(f"✅ R²: {metadata['R²']:.4f}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. AQI Over Time
axes[0, 0].plot(df['dt_readable'], df['aqi'], color='teal', linewidth=1)
axes[0, 0].set_title('AQI Trend Over Time', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('AQI')
axes[0, 0].grid(True, alpha=0.3)

# 2. AQI Distribution
axes[0, 1].hist(df['aqi'], bins=50, color='coral', edgecolor='black')
axes[0, 1].set_title('AQI Distribution', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('AQI Value')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# 3. Pollutants Correlation Heatmap
pollutants = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3', 'aqi']
corr = df[pollutants].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 0], cbar=True)
axes[1, 0].set_title('Pollutants Correlation Matrix', fontsize=14, fontweight='bold')

# 4. Model Performance Metrics
metrics = ['MAE', 'RMSE', 'R²']
values = [metadata['MAE'], metadata.get('RMSE', 0.0088), metadata['R²']]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
axes[1, 1].bar(metrics, values, color=colors, edgecolor='black')
axes[1, 1].set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Value')
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add accuracy text
axes[1, 1].text(1, metadata['R²']/2, f"Accuracy: {metadata['accuracy_percentage']:.2f}%", 
                ha='center', va='center', fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig('analysis_results.png', dpi=300, bbox_inches='tight')
print("\n📈 Charts saved as 'analysis_results.png'")
plt.show()