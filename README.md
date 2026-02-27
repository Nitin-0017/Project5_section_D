# Customer Churn Prediction

**Traditional Machine Learning | Customer Analytics**

---

## Project Overview

This project develops an end-to-end Machine Learning pipeline to predict customer churn using historical behavioral and transactional data.

The system performs data preprocessing, feature engineering, model training, evaluation, and deployment through an interactive Streamlit web application.

**Dataset Size:** ~440K records  
**Target Variable:** Churn (0 = Retained, 1 = Churned)  
**Best Model:** Random Forest (Accuracy ≈ 0.99)

---

## Problem Statement

Customer churn is a major challenge for subscription-based businesses. Identifying customers who are likely to discontinue services enables companies to take proactive retention measures.

Churn behavior depends on multiple factors such as:

- Customer engagement (usage frequency)  
- Service experience (support calls)  
- Payment behavior (payment delay)  
- Subscription characteristics (contract length)

The objective of this project is to build a reliable classification model that predicts churn risk and highlights the key drivers behind customer attrition.

---

## Dataset Description

The dataset contains customer demographic, behavioral, and subscription information.

### Important Features

- `Age`  
- `Tenure`  
- `Usage Frequency`  
- `Support Calls`  
- `Payment Delay`  
- `Total Spend`  
- `Last Interaction`  
- `Gender`  
- `Subscription Type`  
- `Contract Length`  

**Target Column:** `Churn`

---

## Machine Learning Pipeline

### Data Cleaning

- Removed missing values  
- Removed duplicate records  
- Dropped non-informative identifier column (`CustomerID`)

---

### Encoding

- One-Hot Encoding applied to categorical variables  
- Train–test alignment ensured consistent feature space  

---

### Scaling

- StandardScaler applied to numerical features  
- Scaler fitted only on training data to prevent data leakage  

---

### Train-Test Split

- Stratified 80/20 split  
- Maintained class distribution  
- Prevented information leakage  

---

## Models Implemented

| Model                 | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|----------|--------|----------|
| Logistic Regression  | 0.79     | 0.52     | 0.99   | 0.68     |
| Decision Tree        | 0.97     | 0.95     | 0.98   | 0.96     |
| **Random Forest**    | **0.99** | **0.99** | **0.99** | **0.99** |

Random Forest achieved the best performance and was selected for deployment.

**Note:** High performance is attributed to strong feature separability present in the dataset.

---

## Key Insights

- Support call frequency is the strongest churn indicator  
- Customers with higher payment delays show elevated churn risk  
- Lower tenure customers are more likely to churn  
- Subscription type and contract length significantly influence retention  

The model demonstrates strong generalization as training and testing performances are closely aligned.

---

## Model Deployment

The best model and scaler were serialized using `joblib`:

```python
joblib.dump(rf, "churn_model_final.pkl")
joblib.dump(scaler, "scaler.pkl")
