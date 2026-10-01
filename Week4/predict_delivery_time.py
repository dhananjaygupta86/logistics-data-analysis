import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("data/logistics_prediction_sample.csv")

# Target: delivery time
X = df.drop(columns=["order_id", "delivery_time_min"])
y = df["delivery_time_min"]

categorical = ["traffic", "weather", "vehicle_type"]
numeric = ["distance_km", "packages", "transport_cost"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
], remainder="passthrough")

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200, random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R-squared:", r2)

# 5-fold cross-validation
cv_rmse = np.sqrt(-cross_val_score(
    model, X, y, cv=5, scoring="neg_mean_squared_error"
).mean())
print("5-fold CV RMSE:", cv_rmse)

# Simple operational recommendations
print("\nOptimization ideas:")
print("- Prioritize lower-distance routes when service levels are similar.")
print("- Add traffic-aware route planning for high-traffic periods.")
print("- Allocate larger-capacity vehicles to high-package orders.")
print("- Use predicted delivery time to identify orders needing early intervention.")
