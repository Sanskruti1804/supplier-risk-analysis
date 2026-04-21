# Supplier Risk Prediction & Procurement Intelligence System

##  Overview

This project is an end-to-end data analytics and machine learning solution designed to identify **high-risk suppliers in procurement operations**.

It simulates a real-world business scenario where companies need to reduce **delays, defects, and supply chain disruptions** using data-driven decision-making.

The system predicts whether a purchase order is risky and provides **interactive insights through a dashboard**.

---

## Business Problem

Procurement teams face:

* Delayed deliveries from suppliers
* Poor supplier performance visibility
* High operational costs due to defects and inefficiencies
* Lack of real-time risk monitoring

This project solves these problems using **data analytics, machine learning, and visualization**.

---

## Solution Approach

**1. Data Engineering**

* Generated synthetic procurement dataset (2000+ records)
* Cleaned and transformed data
* Created business features:

  * `delay_days`
  * `delivery_risk`
  * `fill_rate`
  * `cost_impact`

---

**2. Machine Learning Model**

* Built classification model to predict supplier risk
* Algorithm used: **Random Forest**
* Target variable: `delivery_risk`

---

**3. Model Evaluation**

* Accuracy, Precision, Recall, F1-score
* Handled class imbalance
* Focused on improving **recall for risky suppliers**

---

**4. Explainability (SHAP)**

* Identified key drivers of supplier risk
* Feature importance visualization
* Transparent model insights

---

**5. Interactive Dashboard (Streamlit)**

* Filter by:

  * Region
  * Supplier
  * Category

* KPIs:

  * Total Orders
  * Risk Rate
  * Risky Suppliers

* Charts:

  * Risk by Region
  * Risk by Category

* Visual outputs:

  * Feature importance
  * SHAP analysis

---

**Tech Stack**

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* SHAP
* Streamlit
* Matplotlib

---

**Key Insights**

* Supplier delays strongly impact risk classification
* Low fill rate correlates with higher risk
* Poor supplier rating increases risk probability
* Defect rate directly impacts cost inefficiency

---

**Business Impact**

This system can help companies:

* Reduce supplier delays by ~20–30%
* Improve procurement decision-making
* Identify high-risk suppliers early
* Optimize supply chain performance

---

**Project Structure**

```
supplier-risk-analysis/
│
├── data/
│   └── procurement_data_cleaned.csv
│
├── outputs/
│   ├── feature_importance.png
│   └── shap_summary.png
│
├── src/
│   ├── data_generation.py
│   ├── train_model.py
│   ├── feature_importance.py
│   ├── shap_analysis.py
│   └── dashboard.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

**How to Run**

```bash
pip install -r requirements.txt
python src/train_model.py
python src/feature_importance.py
python src/shap_analysis.py
python -m streamlit run src/dashboard.py
```

---

 **Future Improvements**

* Integrate with SAP / ERP systems
* Real-time API-based prediction
* Advanced ML models (XGBoost tuning)
* Deployment on cloud (AWS / Azure)

