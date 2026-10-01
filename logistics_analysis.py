import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load sample logistics data
df = pd.read_csv("data/logistics_sample.csv")

print(df.head())
print(df.isnull().sum())

# KPI 1: Average delivery time
avg_delivery_time = df["delivery_time_min"].mean()

# KPI 2: On-time delivery rate
on_time_rate = df["on_time"].mean() * 100

# KPI 3: Average transport cost
avg_cost = df["transport_cost"].mean()

print("Average delivery time:", round(avg_delivery_time, 2), "minutes")
print("On-time delivery rate:", round(on_time_rate, 2), "%")
print("Average transport cost:", round(avg_cost, 2))

# Simple regression: predict delivery time from distance and package count
X = df[["distance_km", "packages"]]
y = df["delivery_time_min"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("First 5 predicted delivery times:", predictions[:5])

# Simple visualization
df["delivery_time_min"].plot(kind="hist", bins=15)
plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Number of Orders")
plt.show()
