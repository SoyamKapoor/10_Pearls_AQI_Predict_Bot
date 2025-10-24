import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load dataset
df = pd.read_csv("data/cleaned_data.csv")
df['dt_readable'] = pd.to_datetime(df['dt_readable'])

print("✅ Dataset Loaded Successfully!")
print("\nBasic Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe().T)

# --- Missing Values ---
print("\nMissing Values per Column:")
print(df.isna().sum())

# --- Feature Relationships ---
# 1. AQI Trend over Time
plt.figure()
plt.plot(df['dt_readable'], df['aqi'], color='teal', linewidth=1)
plt.title("AQI Trend Over Time", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Air Quality Index (AQI)")
plt.grid(alpha=0.5)
plt.tight_layout()
plt.savefig("eda_aqi_trend.png", dpi=300)
plt.show()

# 2. Pollutant Distributions
pollutants = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
df[pollutants].hist(bins=30, figsize=(15, 10), color='skyblue', edgecolor='black')
plt.suptitle("Pollutant Concentration Distribution", fontsize=18)
plt.tight_layout()
plt.savefig("eda_pollutant_distributions.png", dpi=300)
plt.show()

# 3. Boxplots for Outlier Detection
plt.figure(figsize=(16, 8))
sns.boxplot(data=df[pollutants])
plt.title("Pollutant Outlier Analysis", fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_boxplots.png", dpi=300)
plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df[pollutants + ['aqi']].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap: Pollutants vs AQI", fontsize=16)
plt.tight_layout()
plt.savefig("eda_heatmap.png", dpi=300)
plt.show()


print("\n✅ EDA Completed! Charts saved as:")
print(" - eda_aqi_trend.png")
print(" - eda_pollutant_distributions.png")
print(" - eda_boxplots.png")
print(" - eda_heatmap.png")