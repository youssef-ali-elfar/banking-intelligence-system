# 🏦 Banking Transactions Intelligence System

> **DS 405 — Big Data Analysis**
> Pharos University In Alexandria | Faculty of Computer Science and AI

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PySpark](https://img.shields.io/badge/PySpark-3.5.0-orange)
![ML](https://img.shields.io/badge/AI-Fraud%20Detection-red)
![Status](https://img.shields.io/badge/Status-Complete-green)

---

## 📌 Project Overview

A full Big Data analytics pipeline for banking transactions using **Apache Spark (PySpark)**.
Includes data generation, cleaning, feature engineering, window functions, SQL analytics,
and an **AI-powered Fraud Detection Model** (Bonus).

---

## 🗂️ Project Structure

```
banking_project/
│
├── generate_data.py        # Phase 1 — Synthetic data generation (Faker)
├── banking_analytics.py    # Phase 2-8 — Full PySpark pipeline
├── ai_model.py             # 🤖 BONUS — Fraud Detection ML Model
├── visualize.py            # 📊 Dashboard & Visualizations
├── main.py                 # ▶️  Run everything in order
│
├── requirements.txt        # Dependencies
├── README.md               # This file
│
└── data/
    ├── raw/                # Generated CSV files
    └── processed/          # Cleaned Parquet files (partitioned)
```

---

## ✅ Phases Implemented

| Phase | Description | Tool |
|-------|-------------|------|
| 1 | Data Generation (200 customers, 313 accounts, 2000+ transactions) | Faker + Pandas |
| 2 | Data Cleaning (nulls, negatives, duplicates) | PySpark |
| 3 | Feature Engineering (time_of_day, amount_category, is_suspicious) | PySpark |
| 4 | Joins across 3 datasets | PySpark |
| 5 | Aggregations (spend per customer, per city, per category) | PySpark |
| 6 | Window Functions (LAG, RANK, Running Total, PERCENT_RANK) | PySpark |
| 7 | SQL vs DataFrame Comparison | Spark SQL |
| 8 | Partitioned Save by account_type | Parquet |
| 🎁 BONUS | Fraud Detection Model (Random Forest) | MLlib + Sklearn |

---

## 🤖 AI Model — Fraud Detection (Bonus)

Trained a **Random Forest Classifier** to detect suspicious/fraudulent transactions.

**Features used:**
- `amount`, `hour`, `day_of_week`, `month`
- `is_weekend`, `account_type` (encoded)
- `tx_type` (encoded), `amount_category` (encoded)

**Results:**
- Accuracy, Precision, Recall, F1-Score
- Feature Importance Chart
- ROC Curve

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run everything at once
```bash
python main.py
```

### Or run step by step
```bash
python generate_data.py       # Generate data
python banking_analytics.py   # PySpark pipeline
python ai_model.py            # Fraud detection model
python visualize.py           # Dashboard
```

---

## ⚠️ Prerequisites

- Python 3.10+
- Java 8 or 11 (required for PySpark)
  - Check: `java -version`
  - Download: https://www.java.com

---

## 👥 Team Members

| Name | ID |
|------|----|
|      |    |
|      |    |
|      |    |

---

## 📊 Sample Output

The dashboard generates 12 visualizations including:
- Spending by category and city
- Transaction type distribution
- Fraud/suspicious transaction heatmap
- ML model performance metrics
