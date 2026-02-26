# 🧠 ML Binary & Multi-Class Classification Project

A comprehensive Machine Learning classification pipeline covering **6 datasets** with **6 ML models** each, totaling **36 trained models**.

## 📊 Datasets

| # | Dataset | Type | Samples | Features |
|---|---------|------|---------|----------|
| 1 | Breast Cancer | Binary | 569 | 30 |
| 2 | Pima Diabetes | Binary | 768 | 8 |
| 3 | Titanic | Binary | 891 | 11 |
| 4 | Iris | Multi-Class | 150 | 4 |
| 5 | Wine Quality | Multi-Class | 1,599 | 11 |
| 6 | Dry Bean | Multi-Class | 13,611 | 16 |

## 🤖 Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost

## 📋 Complete ML Pipeline

| Step | Description |
|------|-------------|
| EDA | Distributions, correlations, pairplots, boxplots |
| Outlier Handling | IQR detection + Winsorization (capping) |
| Missing Values | Median imputation for biologically impossible zeros |
| Feature Engineering | Title extraction, FamilySize, IsAlone (Titanic) |
| Feature Selection | ANOVA F-test + Mutual Information scoring |
| Class Imbalance | SMOTE (Synthetic Minority Over-sampling) |
| Feature Scaling | StandardScaler (mean=0, std=1) |
| Cross-Validation | 5-Fold Stratified CV for model stability |
| Evaluation | Accuracy, Precision, Recall, F1, ROC-AUC |
| Classification Reports | Per-class precision, recall, F1 |
| Feature Importance | Tree-based model feature ranking |
| Learning Curves | Overfitting/underfitting visualization |
| Hyperparameter Tuning | GridSearchCV with 5-fold CV |
| Model Saving | Best models exported with joblib |

## 🚀 How to Run

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `ML_Classification_Complete.ipynb`
3. Upload all dataset files when prompted
4. Run all cells (~10-15 min)

## 🛠 Tech Stack

- Python 3.x
- scikit-learn
- XGBoost
- imbalanced-learn (SMOTE)
- pandas, numpy
- matplotlib, seaborn

## 📁 Project Structure

```
├── ML_Classification_Complete.ipynb   # Main notebook (Google Colab)
├── ML_Classification_Complete.py      # Python script version
├── data.csv                           # Breast Cancer dataset
├── diabetes.csv                       # Pima Diabetes dataset
├── titanic/                           # Titanic dataset
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
├── Iris/
│   └── Iris.csv                       # Iris dataset
├── winequality-red.csv                # Wine Quality dataset
├── Dry_Bean_Dataset/
│   ├── Dry_Bean_Dataset.xlsx          # Dry Bean dataset
│   └── Dry_Bean_Dataset.arff
└── README.md
```
