import os
import joblib
import matplotlib.pyplot as plt
import matplotx as mpx

from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import cross_val_score

from src.data_loader import Load_Data

# Create results folder
os.makedirs("results", exist_ok=True)

# Load data
df = Load_Data()
X = df.drop(['Class', 'SampleID'], axis=1)
y = df['Class']

# Load trained model
rf_model = joblib.load("models/rf_model.pkl")

# Remove constant features
X_clean = VarianceThreshold(0).fit_transform(X)

k_values = list(range(1, 101, 2))
scores = []

for k in k_values:
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(X_clean, y)

    score = cross_val_score(rf_model, X_new, y, cv=5).mean()
    scores.append(score)

# Plot
with plt.style.context(mpx.styles.dracula):
    plt.plot(k_values, scores, marker='o')
    plt.xlabel("Number of Genes")
    plt.ylabel("Accuracy")
    plt.title("Feature Selection vs Accuracy")
    plt.tight_layout()
    plt.savefig("results/feature_vs_accuracy.png", dpi=300)
    plt.show()