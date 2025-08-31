# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Try importing XGBoost
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False

MODEL_PATH = "models/trained_model.joblib"
DATA_PATH = "data/student_scores.csv"

st.set_page_config(page_title="Data Dashboard + ML Predictions", layout="wide")

st.title("📊 Data Dashboard with Machine Learning Predictions")
st.markdown("Upload any dataset and this dashboard will automatically perform EDA, train a model, and allow predictions.")

# --- Sidebar - dataset controls ---
st.sidebar.header("Data & Model")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
if st.sidebar.button("Generate sample data"):
    st.info("Generating sample dataset...")
    import generate_data
    st.success("Sample data generated as data/student_scores.csv. Refresh the app.")
    st.stop()

if uploaded:
    df = pd.read_csv(uploaded)
    st.sidebar.success("Loaded uploaded CSV")
    current_data_path = "uploaded_dataset.csv"
    df.to_csv(current_data_path, index=False)
else:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        current_data_path = DATA_PATH
    else:
        st.error("No dataset found. Upload a CSV or click 'Generate sample data'.")
        st.stop()

st.sidebar.write("Dataset preview:")
st.sidebar.dataframe(df.head())

st.header("Dataset overview")
st.write(f"Rows: {df.shape[0]} — Columns: {df.shape[1]}")
st.dataframe(df.head(10))

# --- Detect numeric columns automatically ---
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
if len(numeric_cols) < 2:
    st.error("Dataset must have at least 2 numeric columns (1 target + 1 feature).")
    st.stop()

st.sidebar.subheader("Choose target column")
TARGET = st.sidebar.selectbox("Target (what you want to predict)", numeric_cols)
FEATURES = [col for col in numeric_cols if col != TARGET]
st.sidebar.write("Using features:", FEATURES)

# --- Choose model type ---
st.sidebar.subheader("Choose model type")
model_options = ["Random Forest", "Linear Regression"]
if xgb_available:
    model_options.append("XGBoost")
model_choice = st.sidebar.selectbox("Model", model_options)

# --- Basic EDA ---
st.subheader("Exploratory Data Analysis")
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**Distribution of {TARGET}**")
    fig = px.histogram(df, x=TARGET, nbins=30)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Correlation heatmap**")
    corr = df[numeric_cols].corr()
    fig2, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig2)

st.subheader("Feature relationships")
for feature in FEATURES[:3]:  # limit to 3 scatterplots
    fig3 = px.scatter(df, x=feature, y=TARGET, trendline="ols",
                      title=f"{feature} vs {TARGET}")
    st.plotly_chart(fig3, use_container_width=True)

# --- Train/test split ---
X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model selection ---
if model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
elif model_choice == "Linear Regression":
    model = LinearRegression()
elif model_choice == "XGBoost" and xgb_available:
    model = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
else:
    st.error("XGBoost not installed. Please install with `pip install xgboost`.")
    st.stop()

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)

# --- Evaluation ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader(f"Model performance ({model_choice})")
st.metric("RMSE", f"{mse**0.5:.2f}")
st.metric("R2 score", f"{r2:.3f}")

# --- Feature importances (if available) ---
if hasattr(model, "feature_importances_"):
    st.subheader("Feature importances")
    importances = model.feature_importances_
    imp_df = pd.DataFrame({"feature": FEATURES, "importance": importances}).sort_values("importance", ascending=False)
    fig4 = px.bar(imp_df, x="feature", y="importance", title="Feature importances")
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info(f"{model_choice} does not provide feature importances.")

# --- Interactive prediction ---
st.subheader("Make a prediction")
st.markdown(f"Enter feature values and get a predicted **{TARGET}**:")

input_data = {}
for feature in FEATURES:
    default_val = float(df[feature].median())
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    val = st.number_input(
        f"{feature}",
        min_value=min_val,
        max_value=max_val,
        value=default_val
    )
    input_data[feature] = val

if st.button("Predict"):
    X_new = pd.DataFrame([input_data])
    pred = model.predict(X_new)[0]
    st.success(f"Predicted {TARGET}: {pred:.2f}")

# --- Download section ---
st.subheader("Download")
with open(MODEL_PATH, "rb") as f:
    st.download_button("⬇️ Download trained model", data=f, file_name="trained_model.joblib")

with open(current_data_path, "rb") as f:
    st.download_button("⬇️ Download dataset", data=f, file_name=os.path.basename(current_data_path))

st.markdown("---")
st.markdown("**Notes:** Works with any dataset with numeric columns. Choose your target column and model type in the sidebar.**")
