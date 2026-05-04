# fraud-anomaly-detection
End-to-end fraud and anomaly detection pipeline — EDA, feature engineering, 
and 3 unsupervised models (Z-Score, Isolation Forest, DBSCAN) on 10K 
marketplace transactions. Isolation Forest achieves 100% precision & recall.



# 🔍 Fraud & Anomaly Detection in Online Marketplaces

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Sklearn](https://img.shields.io/badge/Scikit--Learn-1.4-orange)
![Kaggle](https://img.shields.io/badge/Notebook-Kaggle-20BEFF)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Project Overview

This project builds an end-to-end **fraud and anomaly detection pipeline** for online marketplace transactions. Using a synthetic dataset of 10,000 transactions (9,000 normal + 1,000 fraudulent), three unsupervised anomaly detection models are implemented, compared, and evaluated:

- **Z-Score** — statistical baseline
- **Isolation Forest** — tree-based anomaly detection
- **DBSCAN** — density-based clustering

The project covers the full data science workflow: exploratory data analysis, feature engineering, model training, evaluation, and visualization.

---

## 📂 Repository Structure

```
fraud-anomaly-detection/
│
├── data/
│   └── marketplace_transactions.csv   # Synthetic transaction dataset
│
├── notebooks/
│   └── fraud_anomaly_detection.ipynb  # Main Kaggle notebook
│
├── images/
│   ├── class_distribution.png
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── anomaly_score_distribution.png
│   ├── model_comparison.png
│   └── feature_separation.png
│
└── README.md
```

---

## 📊 Dataset

The dataset contains **10,000 synthetic marketplace transactions** with a **10% fraud rate**, designed to reflect realistic fraud patterns in e-commerce platforms.

| Feature | Description |
|---|---|
| `transaction_amount` | Order value in dollars |
| `seller_feedback_score` | Seller reputation score (0–100) |
| `seller_account_age_days` | Account age in days |
| `num_transactions_last_7d` | Transaction velocity (last 7 days) |
| `num_transactions_last_30d` | Transaction velocity (last 30 days) |
| `dispute_rate` | Ratio of disputed transactions |
| `return_rate` | Ratio of returned items |
| `num_policy_violations` | Count of policy breaches |
| `login_hour` | Hour of login activity |
| `unique_items_listed` | Number of unique listings |
| `is_fraud` | Label: 0 = Normal, 1 = Fraud |

**Key fraud patterns in the data:**
- New accounts (1–90 days) vs established sellers (180–3,650 days)
- Transaction velocity spikes (15–80 transactions/week vs 1–10 normal)
- Low feedback scores (0–60) vs high scores (80–100)
- Night-time login activity (0–5am)
- Mass listings (50–500 items vs 1–50 normal)

---

## 🛠️ Feature Engineering

Six new features were engineered to enhance model performance:

| Feature | Formula | Intuition |
|---|---|---|
| `velocity_ratio` | `txn_7d / (txn_30d + 1)` | Detects burst activity |
| `trust_score` | `feedback * 0.6 + age_normalized * 0.4` | Combined seller trustworthiness |
| `is_night_login` | `1 if login_hour <= 5` | Flags suspicious login hours |
| `price_to_feedback_ratio` | `avg_price / (feedback + 1)` | High price, low trust signal |
| `risk_score` | `dispute_rate * 0.5 + return_rate * 0.5` | Combined risk indicator |
| `listing_intensity` | `items_listed / (account_age + 1)` | Mass listing on new accounts |

---

## 🤖 Models

### 1. Z-Score (Statistical Baseline)
Flags transactions where any feature exceeds 3 standard deviations from the mean.

### 2. Isolation Forest
Tree-based ensemble that isolates anomalies by randomly partitioning the feature space. Anomalies require fewer splits to isolate. Contamination set to 10% (matching known fraud rate).

### 3. DBSCAN
Density-based clustering that identifies points in low-density regions as anomalies (label = -1). Parameters: `eps=1.5`, `min_samples=5`.

---

## 📈 Results

| Model | Precision | Recall | F1-Score | Accuracy | False Positives |
|---|---|---|---|---|---|
| Z-Score | 0.81 | 0.99 | 0.89 | 0.98 | 237 |
| **Isolation Forest** | **1.00** | **1.00** | **1.00** | **1.00** | **0** ✅ |
| DBSCAN | 0.92 | 1.00 | 0.96 | 0.99 | 90 |

**Isolation Forest** achieved perfect separation, confirmed by the anomaly score distribution showing complete separation between normal (scores 0.05–0.20) and fraudulent transactions (scores -0.18 to -0.02).

---

## 🔑 Top Features by Fraud Separation Power

1. `risk_score` — strongest signal (dispute + return combined)
2. `seller_feedback_score` — clear gap between fraud and normal
3. `dispute_rate` — high dispute activity flags fraud
4. `num_transactions_last_7d` — velocity spikes
5. `unique_items_listed` — mass listing behaviour
6. `is_night_login` — engineered feature performing strongly

---

## 🚀 How to Run

1. Open the notebook on Kaggle: [fraud_anomaly_detection.ipynb](#)
2. Add the dataset: `marketplace_transactions.csv`
3. Run all cells in order

**Or locally:**
```bash
git clone https://github.com/azar2020/fraud-anomaly-detection.git
cd fraud-anomaly-detection
pip install -r requirements.txt
jupyter notebook notebooks/fraud_anomaly_detection.ipynb
```

---

## 🧰 Tech Stack

- **Python 3.12**
- **pandas**, **numpy** — data manipulation
- **scikit-learn** — Isolation Forest, DBSCAN, preprocessing
- **scipy** — Z-score statistical baseline
- **matplotlib**, **seaborn** — visualization
- **Kaggle Notebooks** — development environment

---

## 👤 Author

**Azar Taheri**  
Data Analyst | Predictive Analytics | Anomaly Detection  
[LinkedIn](https://linkedin.com/in/azar-taheri) | [GitHub](https://github.com/azar2020) | [Kaggle](https://www.kaggle.com/azartaheri)
