# ============================================================
# CREDIT SCORING MODEL — Logistic Regression
# Group 16 | Supervised Machine Learning Project
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, RocCurveDisplay
)

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv("german_credit_data.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# ============================================================
# 2. FILL MISSING VALUES FIRST (before any logic)
# ============================================================
df['Saving accounts']  = df['Saving accounts'].fillna('unknown')
df['Checking account'] = df['Checking account'].fillna('unknown')

# ============================================================
# 3. CREATE TARGET COLUMN
# High default risk = little/unknown checking account
# AND credit amount above median
# ============================================================
median_credit = df['Credit amount'].median()

df['default'] = (
    (df['Checking account'].isin(['little', 'unknown'])) &
    (df['Credit amount'] > median_credit)
).astype(int)

print(f"Default distribution:\n{df['default'].value_counts()}")
print(f"Default rate: {df['default'].mean():.1%}\n")

# Confirm column exists before proceeding
print("Columns after target creation:", df.columns.tolist())

# ============================================================
# 4. ENCODE CATEGORICAL COLUMNS
# ============================================================
df = pd.get_dummies(df, columns=['Sex', 'Housing', 'Saving accounts',
                                  'Checking account', 'Purpose'],
                    drop_first=True)

print(f"Shape after encoding: {df.shape}")
print("Columns after encoding:", df.columns.tolist())

# ============================================================
# 5. SPLIT FEATURES AND TARGET
# ============================================================
X = df.drop('default', axis=1)
y = df['default']

# ============================================================
# 6. TRAIN/TEST SPLIT — 70% train, 30% test
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}\n")

# ============================================================
# 7. NORMALIZE FEATURES
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ============================================================
# 8. TRAIN THE MODEL
# ============================================================
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
print("Model training complete.\n")

# ============================================================
# 9. PREDICTIONS
# ============================================================
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# ============================================================
# 10. PERFORMANCE REPORT
# ============================================================
print("========================================")
print("       MODEL PERFORMANCE REPORT         ")
print("========================================")
print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"  F1-Score  : {f1_score(y_test, y_pred):.4f}")
print(f"  AUC-ROC   : {roc_auc_score(y_test, y_prob):.4f}")
print("========================================\n")
print(classification_report(y_test, y_pred,
      target_names=["Good Borrower", "Defaulter"]))

# ============================================================
# 11. CROSS-VALIDATION
# ============================================================
cv_scores = cross_val_score(model, X_train_scaled, y_train,
                             cv=5, scoring='accuracy')
print(f"Cross-Val Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

# ============================================================
# 12. CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Good Borrower", "Defaulter"],
            yticklabels=["Good Borrower", "Defaulter"])
plt.title("Confusion Matrix — Logistic Regression")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
print("Saved: confusion_matrix.png")

# ============================================================
# 13. ROC CURVE
# ============================================================
RocCurveDisplay.from_predictions(y_test, y_prob,
                                  name="Logistic Regression")
plt.title("ROC Curve — Credit Scoring Model")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()
print("Saved: roc_curve.png")

# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================
importance = pd.Series(
    model.coef_[0], index=X.columns
).sort_values(key=abs, ascending=False)

print("\nTop 10 Most Influential Features:")
print(importance.head(10))

plt.figure(figsize=(10, 6))
importance.head(10).plot(kind='barh', color='steelblue')
plt.title("Top 10 Feature Importances — Logistic Regression")
plt.xlabel("Coefficient Value")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
print("Saved: feature_importance.png")