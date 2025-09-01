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
powershell
Copy code
python -m venv .venv
.venv\Scripts\activate
If you see an activation error, run this once:

powershell
Copy code
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
3. Install dependencies
bash
Copy code
pip install --upgrade pip
pip install -r requirements.txt
(Optional: if you want XGBoost support)

bash
Copy code
pip install xgboost
4. Generate sample data (optional)
bash
Copy code
python generate_data.py
5. Run the app
bash
Copy code
streamlit run app.py
Now open your browser at 👉 http://localhost:8501

📂 Project Structure
bash
Copy code
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
Upload your own CSV (or generate the built-in sample dataset).

Choose the target column (the value you want to predict).

The app automatically uses other numeric columns as features.

View EDA charts → distribution, heatmap, scatter plots.

The model is trained → you get RMSE & R² metrics.

Enter feature values in the sidebar → click Predict.

Download the trained model or dataset for later use.

📁 Example Datasets
Try with these free datasets:

Student Scores Dataset

Boston Housing Prices

Wine Quality

👨‍💻 Author
Created by Balaji M — GitHub Profile
