import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/logistics_eda_sample.csv")

print(df.head())
print(df.describe())

# Central tendency
print("Mean delivery time:", df["delivery_time_min"].mean())
print("Median delivery time:", df["delivery_time_min"].median())
print("Mean transport cost:", df["transport_cost"].mean())

# Correlation
print(df[["distance_km", "shipment_volume",
          "transport_cost", "delivery_time_min"]].corr())

# Visualization 1
df["delivery_time_min"].plot(kind="hist", bins=18)
plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Orders")
plt.show()

# Visualization 2
plt.scatter(df["distance_km"], df["delivery_time_min"])
plt.title("Distance vs Delivery Time")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (minutes)")
plt.show()

# Visualization 3
df.groupby("traffic")["transport_cost"].mean().plot(kind="bar")
plt.title("Average Transport Cost by Traffic Level")
plt.xlabel("Traffic")
plt.ylabel("Average Cost")
plt.show()
