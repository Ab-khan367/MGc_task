"""
Lead scoring model for MGC CRM data.

Trains a Random Forest classifier to predict whether a lead will convert.
Data prep steps include city normalization, median imputation for missing values,
and dropping columns that would cause data leakage (especially token payments).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "leads.csv"))
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

# Drop duplicate records sharing the same crm_record_hash
before = len(df)
df = df.drop_duplicates(subset="crm_record_hash", keep="first")
print(f"Removed {before - len(df)} duplicate rows (same crm_record_hash)")

# Columns dropped:
# - lead_id, crm_record_hash: arbitrary identifiers
# - created_at: timestamps might lead to leakage or temporal bias
# - token_amount_received_pkr: target leakage! paying a token happens AFTER converting.
DROP_COLS = [
    "lead_id",
    "created_at",
    "crm_record_hash",
    "token_amount_received_pkr",
]
df = df.drop(columns=DROP_COLS)
print(f"Dropped columns: {DROP_COLS}")

# Fix typos and standardize city names across the dataset
CITY_MAP = {
    "ISLAMABAD": "Islamabad",
    "ISB":       "Islamabad",
    "RAWALPINDI":"Rawalpindi",
    "Rwp":       "Rawalpindi",
    "LAHORE":    "Lahore",
    "KARACHI":   "Karachi",
    "khi":       "Karachi",
    "PESHAWAR":  "Peshawar",
    "FAISALABAD":"Faisalabad",
    "MULTAN":    "Multan",
    "GUJRANWALA":"Gujranwala",
    "ABBOTTABAD":"Abbottabad",
}
df["city"] = df["city"].replace(CITY_MAP)
print(f"Normalized {len(CITY_MAP)} city name variants")

# Impute missing values (Unknown for area, median for numerical fields)
df["area"] = df["area"].fillna("Unknown")

for col in ["bedrooms", "first_response_minutes", "agent_experience_years", "budget_pkr_lac"]:
    median_val = df[col].median()
    missing_count = df[col].isna().sum()
    df[col] = df[col].fillna(median_val)
    if missing_count > 0:
        print(f"  Filled {missing_count} missing '{col}' with median={median_val}")

# Encode strings to integers for scikit-learn
CATEGORICAL_COLS = ["source", "city", "area", "property_type"]
label_encoders = {}

for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Separate features from target
X = df.drop(columns=["converted"])
y = df["converted"]

print(f"\nFeature columns ({len(X.columns)}): {list(X.columns)}")
print(f"Class balance: {y.value_counts().to_dict()}  "
      f"({y.mean()*100:.1f}% positive)")

# Stratified split to preserve the ~93/7 class ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Baseline Random Forest model using balanced class weights
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# Evaluate using positive-class F1 score due to class imbalance
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

f1 = f1_score(y_test, y_pred, pos_label=1)
auc = roc_auc_score(y_test, y_proba)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"\nPrimary metric — F1-score (positive class): {f1:.4f}")
print(f"Secondary metric — AUC-ROC:                 {auc:.4f}")
print(f"\nWhy F1-score?")
print(f"  Class balance is {(1-y.mean())*100:.0f}/{y.mean()*100:.0f} "
      f"(negative/positive).")
print(f"  Accuracy would be {(1-y.mean())*100:.0f}% by always predicting 0 — useless.")
print(f"  F1 balances precision and recall on the minority (converted) class,")
print(f"  which is what the sales team actually cares about: finding the leads")
print(f"  most likely to convert without wasting time on false positives.")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Converted", "Converted"]))

# Top features ranked by feature importance
importances = pd.Series(model.feature_importances_, index=X.columns)
print("Top 10 Feature Importances:")
print(importances.sort_values(ascending=False).head(10).to_string())

# Save serialized model artifact and encoders for the web interface
model_path = os.path.join(BASE_DIR, "lead_model.joblib")
joblib.dump({
    "model": model,
    "label_encoders": label_encoders,
    "feature_columns": list(X.columns),
    "categorical_columns": CATEGORICAL_COLS,
}, model_path)
print(f"\nModel saved to {model_path}")
