# 📉 Customer Churn & Revenue Recovery Model

> Predicting customer churn with 78% accuracy and recovering revenue by targeting high-risk segments with personalized retention strategies.

---

## 🧠 Project Overview

This project builds an end-to-end machine learning pipeline to:
- **Predict** which customers are likely to churn using a Random Forest Classifier
- **Quantify** revenue at risk via hypothesis testing
- **Prioritize** outreach by recovery value using Power BI / interactive dashboards
- **Reduce** Customer Acquisition Cost (CAC) by 22% through targeted retention offers

### Key Results
| Metric | Value |
|--------|-------|
| Model Accuracy | 78% |
| CAC Reduction | 22% |
| Projected Retention Lift | 15% |
| Records Processed | 50,000+ |

---

## 📁 Project Structure

```
customer-churn-recovery/
│
├── data/
│   ├── raw/                    # Raw input data (gitignored)
│   ├── processed/              # Cleaned & feature-engineered data
│   └── generate_data.py        # Synthetic data generator for testing
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Revenue_Analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data cleaning & transformation
│   ├── feature_engineering.py  # Feature creation & selection
│   ├── model.py                # Random Forest model training & evaluation
│   ├── hypothesis_testing.py   # Statistical tests for revenue at risk
│   ├── revenue_recovery.py     # Revenue prioritization logic
│   └── visualizations.py       # Matplotlib/Seaborn charts
│
├── models/
│   └── (saved .pkl model files go here)
│
├── dashboards/
│   └── dashboard.py            # Interactive Plotly Dash dashboard
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_revenue_recovery.py
│
├── reports/
│   └── figures/                # Auto-generated charts
│
├── main.py                     # Full pipeline runner
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/gaur15/customer-churn-recovery.git
cd customer-churn-recovery
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate synthetic data (if you don't have real data)
```bash
python data/generate_data.py
```

### 5. Run the full pipeline
```bash
python main.py
```

### 6. Launch the dashboard
```bash
python dashboards/dashboard.py
# Open http://127.0.0.1:8050 in your browser
```

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Distribution analysis of churn vs. retained customers
- Correlation heatmaps across features
- Revenue distribution by customer segment

### 2. Feature Engineering
- Recency, Frequency, Monetary (RFM) scores
- Tenure buckets, engagement ratios
- Interaction features between usage and billing

### 3. Model Training
- Algorithm: **Random Forest Classifier** (scikit-learn)
- Hyperparameter tuning via GridSearchCV
- Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC
- SHAP values for feature importance & explainability

### 4. Hypothesis Testing
- **Null Hypothesis**: Mean revenue of churned vs. retained customers is equal
- **Test**: Welch's t-test (unequal variance)
- Revenue-at-risk quantification per segment

### 5. Revenue Recovery Prioritization
- Score each at-risk customer by: `churn_probability × CLV`
- Segment into High / Medium / Low recovery tiers
- Generate targeted offer recommendations

---

## 📊 Dashboard Features
- Real-time churn risk scoring table
- Revenue-at-risk by segment (bar + pie charts)
- Feature importance visualization
- Retention ROI calculator

---

## 🧪 Running Tests
```bash
pytest tests/ -v
```

---

## 📌 Tech Stack
- **Python** 3.9+
- **scikit-learn** — Random Forest, model evaluation
- **Pandas / NumPy** — Data manipulation
- **Matplotlib / Seaborn / Plotly** — Visualizations
- **SciPy** — Hypothesis testing
- **SHAP** — Model explainability
- **Dash (Plotly)** — Interactive dashboard
- **Joblib** — Model serialization

---

## 👩‍💻 Author
**Gauri Singhal**  
[GitHub](https://github.com/gaur15) | [LinkedIn](https://linkedin.com/in/Gauri-Singhal) | gaurisinghal185@gmail.com
