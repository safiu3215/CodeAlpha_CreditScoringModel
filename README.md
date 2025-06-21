# Credit Scoring Model

This project builds a machine learning model to predict an individual's creditworthiness using classification algorithms and synthetic financial data.

---

## 🔍 Objective

Predict whether an individual will default on a loan based on:
- Income
- Debts
- Loan Amount
- Payment History

---

## 🧠 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Jupyter / Google Colab

---

## 🛠️ Project Workflow

### 1. Data Simulation
A synthetic dataset of 500 individuals is generated using `NumPy` to mimic real-world financial behavior.

### 2. Feature Engineering
We derive meaningful financial ratios:
- `debt_to_income = debts / income`
- `loan_to_income = loan_amount / income`

We also convert categorical `payment_history` into numeric values.

### 3. Preprocessing
- Handle missing/infinite values
- Standardize data using `StandardScaler`

### 4. Model Training
Three machine learning models are trained:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

### 5. Evaluation
Each model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score (if applicable)

---

## ✅ Sample Output (Varies Per Run)

```text
Logistic Regression Results:
Accuracy: 0.84
Precision: 0.76
Recall: 0.68
F1 Score: 0.72
ROC AUC: 0.88