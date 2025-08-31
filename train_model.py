# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/student_scores.csv")

X = df[["hours_studied", "attendance_pct", "assignments_submitted", "previous_score"]]
y = df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}, RMSE: {mse**0.5:.2f}, R2: {r2:.3f}")

joblib.dump(model, "models/rf_model.joblib")
print("Saved models/rf_model.joblib")
