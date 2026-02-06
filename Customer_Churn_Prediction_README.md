# Customer Churn Prediction using Machine Learning

## Problem Statement
Customer churn refers to customers discontinuing a service or subscription.
The objective of this project is to build a machine learning model that predicts whether a customer is likely to churn based on behavioral and subscription data.

---

## Dataset Overview
- Total Records: 64,374
- Features: 11 input features + 1 target variable
- Target Variable: Churn (0 = No churn, 1 = Churn)

Key Features:
- Age
- Tenure
- Usage Frequency
- Support Calls
- Payment Delay
- Total Spend
- Last Interaction
- Gender
- Subscription Type
- Contract Length

Dataset is balanced:
- 52.6% Non-Churn
- 47.3% Churn

---

## Project Workflow

### 1. Data Cleaning
Removed identifier column:
df = df.drop("CustomerID", axis=1)

Reason: CustomerID does not provide predictive value.

---

### 2. Feature & Target Separation
X = df.drop("Churn", axis=1)
y = df["Churn"]

---

### 3. Encoding Categorical Variables
X = pd.get_dummies(X, drop_first=True)

drop_first=True avoids multicollinearity (dummy variable trap).

---

### 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

Parameter Explanation:
- test_size=0.2 → 20% data used for testing
- random_state=42 → ensures reproducibility
- stratify=y → maintains class distribution

---

### 5. Feature Scaling (For Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Scaling ensures features contribute proportionally.

---

## Models Used

### Logistic Regression
A linear classification algorithm that models probability using the sigmoid function.

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

max_iter=1000 ensures convergence.

---

### Random Forest
An ensemble model that builds multiple decision trees and averages predictions.

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

random_state ensures reproducibility.

---

## Model Evaluation

Metrics Used:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## Cross Validation

scores = cross_val_score(
    LogisticRegression(max_iter=1000),
    X,
    y,
    cv=5,
    scoring="roc_auc"
)

Mean ROC-AUC ≈ 0.90

---

## Results Summary

| Model | Accuracy | F1 Score | ROC-AUC |
|--------|----------|----------|----------|
| Logistic Regression | ~0.83 | ~0.82 | ~0.90 |
| Random Forest | ~0.99 | ~0.99 | ~0.999 |

---

## Feature Importance (Random Forest)

Top Influential Features:
1. Payment Delay
2. Support Calls
3. Tenure
4. Usage Frequency

---

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn

---

## Conclusion
This project demonstrates end-to-end ML pipeline development, model comparison, cross-validation, and business insight extraction.
