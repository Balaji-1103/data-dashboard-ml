# 📊 Data Dashboard with Machine Learning Predictions  

An interactive **Streamlit dashboard** that lets you:  
✅ Upload any CSV dataset  
✅ Explore data with visualizations (EDA)  
✅ Train ML models (Random Forest, Linear Regression, XGBoost)  
✅ Evaluate model performance (RMSE, R²)  
✅ Make real-time predictions with custom inputs  

---

## 🚀 How to Run  

1. Clone the repo  
   ```bash
   git clone https://github.com/Balaji-1103/data-dashboard-ml.git
   cd data-dashboard-ml

2. Create & activate virtual environment (Windows PowerShell)
   ```bash
   python -m venv .venv
   .venv\Scripts\activate

3. Install dependencies
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt

4. Generate sample data (optional)
   ```bash
   python generate_data.py

5. Run the app
   ```bash
   streamlit run app.py


Now open your browser at 👉 http://localhost:8501

📂 Project Structure
data-dashboard-ml/
├─ app.py                 # Main Streamlit app
├─ generate_data.py       # Script to create sample dataset
├─ train_model.py         # Optional: train model manually
├─ requirements.txt       # Python dependencies
├─ README.md              # Documentation
├─ .gitignore             # Ignore data, models, venv
├─ data/                  # Sample dataset (ignored in git)
└─ models/                # Trained models (ignored in git)

📊 How to Use

1. Upload your own CSV (or generate the built-in sample dataset).

2. Choose the target column (the value you want to predict).

3. The app automatically uses other numeric columns as features.

4. View EDA charts → distribution, heatmap, scatter plots.

5. The model is trained → you get RMSE & R² metrics.

6. Enter feature values in the sidebar → click Predict.

7. Download the trained model or dataset for later use.
