import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import Load_Data
from src.model import GeneClassifier

# Create folders
os.makedirs("models", exist_ok=True)

# Load data
df = Load_Data()

X = df.drop(['Class', 'SampleID'], axis=1)
y = df['Class']

# Split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- Random Forest --------
rf_model = GeneClassifier("rf")
rf_model.train(x_train, y_train)

# -------- SVM (with scaling) --------
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

svm_model = GeneClassifier("svm")
svm_model.train(x_train_scaled, y_train)

# -------- Save models --------
joblib.dump(rf_model.model, "models/rf_model.pkl")
joblib.dump(svm_model.model, "models/svm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Models saved in /models/")