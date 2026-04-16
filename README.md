🏦 Bank Deposit Predictor

📌 Overview

The Bank Deposit Predictor is a machine learning web application built using Streamlit.
It predicts whether a client is likely to subscribe to a term deposit based on their personal and campaign-related information.

The system uses a trained ML model along with feature scaling to generate predictions and probability insights in a user-friendly interface.

🚀 Features

- 📊 Predicts customer subscription to term deposit
- 🤖 Machine Learning model integration
- 🎯 Probability-based prediction output
- 📱 Clean and interactive UI using Streamlit
- 📈 Visual probability breakdown (bars & stats)
- ⚡ Fast real-time predictions

🛠️ Tech Stack

Frontend/UI:Streamlit
Backend: Python
Libraries:

- Pandas
- NumPy
- Scikit-learn
- Joblib

📂 Project Structure
project/
│── app.py # Main Streamlit application
│── best_model.pkl # Trained ML model
│── scaler.pkl # Feature scaler
│── top_features.pkl # Selected features
│── README.md

⚙️ Installation & Setup

1️⃣ Clone the repository
git clone https://github.com/your-username/bank-deposit-predictor.git
cd bank-deposit-predictor

2️⃣ Install dependencies
pip install -r requirements.txt
_(If you don’t have requirements.txt, install manually:)_
pip install streamlit pandas numpy scikit-learn joblib
3️⃣ Run the application
streamlit run app.py

📊 How It Works

- User inputs customer data (age, balance, loans, etc.)
- Data is transformed using a pre-trained scaler
- ML model predicts:
  - ✔️ Likelihood of subscription
  - 📈 Probability scores

- Results are displayed with visual feedback

🔐 Model Details

- Model trained using classification algorithms
- Feature selection applied using `top_features.pkl`
- Scaling applied before prediction

📌 Future Improvements

- Add more features for prediction
- Deploy using cloud platforms (Streamlit Cloud / Heroku)
- Improve model accuracy
- Add user authentication

👨‍💻 Author

Suprim Shakya

⭐ Acknowledgements

- Open-source ML libraries
- Dataset for bank marketing prediction
