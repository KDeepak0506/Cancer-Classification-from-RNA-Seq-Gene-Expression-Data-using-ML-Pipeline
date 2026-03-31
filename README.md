# 🧬 Gene Expression Cancer Classification

## 📌 Overview

This project focuses on classifying different types of cancer using gene expression (RNA-Seq) data.
Machine learning models are trained on high-dimensional biological data to identify patterns that distinguish cancer types.

---
## 📂 Dataset

📂 Dataset: UCI Gene Expression Cancer RNA-Seq Dataset

The dataset used in this project is too large to be stored in the repository.

You can download it from:
https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq

After downloading, place it in:

dataset/raw/

## 🎯 Objectives

* Classify cancer types based on gene expression
* Compare performance of multiple ML models
* Perform feature selection to identify important genes
* Build a clean, scalable ML pipeline

---

## 🚀 Features

* ✅ Random Forest & Support Vector Machine (SVM)
* ✅ Cross-validation for robust evaluation
* ✅ Feature importance analysis
* ✅ Feature selection (SelectKBest + VarianceThreshold)
* ✅ High-accuracy classification (~99%)
* ✅ Industry-style project structure

---

## 🧠 Models Used

### 🌲 Random Forest

* Ensemble learning method
* Handles high-dimensional data effectively
* Provides feature importance

### 📈 Support Vector Machine (SVM)

* Effective in high-dimensional spaces
* Linear kernel used for classification
* Requires feature scaling

---

## 📊 Results

| Model         | Accuracy | Precision | Recall | F1 Score |
| ------------- | -------- | --------- | ------ | -------- |
| Random Forest | ~0.99    | ~0.99     | ~0.99  | ~0.99    |
| SVM           | ~0.99    | ~0.99     | ~0.99  | ~0.99    |

* Strong performance across all classes
* Minimal misclassification

---

## 📊 Visualizations

### 📉 Feature Selection vs Accuracy

This plot shows how model performance changes with the number of selected genes.

<p align="center">
  <img src="results/feature_vs_accuracy.png" width="600">
</p>

---

### 📊 Class Distribution

Distribution of different cancer types in the dataset.

<p align="center">
  <img src="results/Class_Distribution.png" width="600">
</p>

---

## 🔍 Interpretation

* Accuracy remains high even with fewer genes → strong signal in data
* Feature selection reduces dimensionality significantly
* Model generalizes well across classes
* Slight class imbalance handled effectively

---

## 📁 Project Structure

```
project/
│
├── src/
│   ├── model.py              # Model definitions
│   ├── train.py              # Training pipeline
│   ├── evaluate.py           # Evaluation metrics
│   ├── feature_selection.py  # Feature selection experiments
│   └── data_loader.py        # Dataset loader
│
├── models/                   # Saved models (joblib)
├── results/                  # Metrics & plots
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🔹 Train Models

```bash
python -m src.train
```

### 🔹 Evaluate Models

```bash
python -m src.evaluate
```

### 🔹 Feature Selection Analysis

```bash
python -m src.feature_selection
```

---

## 📂 Outputs

All outputs are saved in the `results/` folder:

* 📄 Model evaluation reports (`.txt`)
* 📊 Confusion matrices
* 📉 Feature selection plots

---

## 🔍 Key Learnings

* High-dimensional biological data requires feature selection
* Even simple models perform well on structured datasets
* Cross-validation ensures reliable performance evaluation
* Clean project structure improves scalability and maintainability

---

## 🚀 Future Improvements

* Add XGBoost / LightGBM
* Use SHAP for model explainability
* Perform advanced hyperparameter tuning
* Deploy as a web API (Flask/FastAPI)
* Use TCGA dataset for more complexity

---

## 👨‍💻 Author

**Deepak**

---

## ⭐ Acknowledgements

* UCI Machine Learning Repository
* scikit-learn documentation
