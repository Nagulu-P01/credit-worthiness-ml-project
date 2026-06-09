# =========================
# BASELINE CREDIT MODEL
# SIMPLE VERSION (NO OPTIMIZATION)
# =========================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

# =========================
# 1. LOAD DATA
# =========================

df = pd.read_csv("data/loan.csv")

# =========================
# 2. BASIC CLEANING
# =========================

df.drop("Loan_ID", axis=1, inplace=True)

# Fill missing values
num_cols = ["LoanAmount", "Loan_Amount_Term", "Credit_History"]
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

cat_cols = ["Gender", "Married", "Dependents", "Self_Employed"]
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# =========================
# 3. ENCODING (SIMPLE LABEL ENCODER)
# =========================

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# =========================
# 4. SPLIT FEATURES & TARGET
# =========================

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# =========================
# 5. TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================
# 6. MODEL TRAINING (DEFAULT RF)
# =========================

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# =========================
# 7. PREDICTION
# =========================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# =========================
# 8. EVALUATION
# =========================

print("\n===== BASELINE MODEL RESULTS =====")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# 9. SAVE MODEL
# =========================

joblib.dump(model, "baseline_model.pkl")

print("\nModel saved successfully!")