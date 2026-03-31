import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_validate, cross_val_predict

from src.data_loader import Load_Data

# Create results folder
os.makedirs("results", exist_ok=True)

# Load data
df = Load_Data()
X = df.drop(['Class', 'SampleID'], axis=1)
y = df['Class']

# Load models
rf_model = joblib.load("models/rf_model.pkl")
svm_model = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")

scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

# ================= RF =================
rf_scores = cross_validate(rf_model, X, y, cv=5, scoring=scoring)
rf_pred = cross_val_predict(rf_model, X, y, cv=5)

rf_text = f"""
===== RANDOM FOREST =====
Accuracy: {rf_scores['test_accuracy'].mean()}
Precision: {rf_scores['test_precision_weighted'].mean()}
Recall: {rf_scores['test_recall_weighted'].mean()}
F1: {rf_scores['test_f1_weighted'].mean()}

{classification_report(y, rf_pred)}
Confusion Matrix:
{confusion_matrix(y, rf_pred)}
"""

print(rf_text)

with open("results/rf_results.txt", "w") as f:
    f.write(rf_text)

# ================= SVM =================
X_scaled = scaler.transform(X)

svm_scores = cross_validate(svm_model, X_scaled, y, cv=5, scoring=scoring)
svm_pred = cross_val_predict(svm_model, X_scaled, y, cv=5)

svm_text = f"""
===== SVM =====
Accuracy: {svm_scores['test_accuracy'].mean()}
Precision: {svm_scores['test_precision_weighted'].mean()}
Recall: {svm_scores['test_recall_weighted'].mean()}
F1: {svm_scores['test_f1_weighted'].mean()}

{classification_report(y, svm_pred)}
Confusion Matrix:
{confusion_matrix(y, svm_pred)}
"""

print(svm_text)

with open("results/svm_results.txt", "w") as f:
    f.write(svm_text)