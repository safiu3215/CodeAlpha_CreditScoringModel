import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings

# ----------------------------
# Step 1: Simulate a Realistic Dataset
# ----------------------------
np.random.seed(42)
df = pd.DataFrame({
    'income': np.random.randint(20000, 100000, size=500),
    'debts': np.random.randint(1000, 50000, size=500),
    'loan_amount': np.random.randint(500, 30000, size=500),
    'payment_history': np.random.choice(['on_time', 'late', 'very_late', 'no_history'], size=500),
    'defaulted': np.random.choice([0, 1], size=500, p=[0.7, 0.3])
})

# ----------------------------
# Step 2: Feature Engineering
# ----------------------------
df['debt_to_income'] = df['debts'] / df['income']
df['loan_to_income'] = df['loan_amount'] / df['income']

# Convert categorical feature into ordinal numeric
df['payment_history'] = df['payment_history'].map({
    'on_time': 0,
    'late': 1,
    'very_late': 2,
    'no_history': 3
})

# ----------------------------
# Step 3: Prepare X and y
# ----------------------------
X = df.drop('defaulted', axis=1)
y = df['defaulted']

# Replace missing or infinite values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.mean(), inplace=True)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# Step 4: Split Dataset
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Step 5: Train Models
# ----------------------------
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

forest_model = RandomForestClassifier()
forest_model.fit(X_train, y_train)

# ----------------------------
# Step 6: Evaluate Models
# ----------------------------
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))
    print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))
    if hasattr(model, "predict_proba"):
        try:
            roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            print("ROC AUC:", roc)
        except ValueError:
            print("ROC AUC: Cannot compute (only one class present in y_test)")

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Evaluate all models
evaluate_model("Logistic Regression", log_model, X_test, y_test)
evaluate_model("Decision Tree", tree_model, X_test, y_test)
evaluate_model("Random Forest", forest_model, X_test, y_test)
