import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data/logistics_raw_sample.csv")
print("Initial shape:", df.shape)
print(df.isnull().sum())

# Remove exact duplicate rows
df = df.drop_duplicates()

numeric_cols = ["distance_km","packages","temperature_c","humidity_pct","delivery_time_min","transport_cost"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values with the most common category
df["traffic"] = df["traffic"].fillna(df["traffic"].mode()[0])
df["vehicle_type"] = df["vehicle_type"].fillna(df["vehicle_type"].mode()[0])

# IQR outlier capping
for col in ["distance_km","delivery_time_min","transport_cost"]:
    q1, q3 = df[col].quantile([0.25,0.75])
    iqr = q3-q1
    df[col] = df[col].clip(q1-1.5*iqr, q3+1.5*iqr)

# Min-Max normalization
scale_cols = ["distance_km","packages","temperature_c","humidity_pct"]
df[scale_cols] = MinMaxScaler().fit_transform(df[scale_cols])

df.to_csv("data/logistics_cleaned_sample.csv", index=False)
print("Cleaning completed. Final shape:", df.shape)
