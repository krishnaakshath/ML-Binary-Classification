# %% [markdown]
# # 🧠 Comprehensive ML Classification Project
# ## Binary & Multi-Class Classification Pipeline
# ---
# **Datasets:**
# - Binary: Breast Cancer, Diabetes (Pima), Titanic
# - Multi-Class: Iris, Wine Quality, Dry Bean
#
# **Models:** Logistic Regression, Decision Tree, Random Forest, SVM, KNN, XGBoost
#
# **Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix

# %% [markdown]
# ## 📦 Part 1: Setup & Imports

# %%
# Install required packages (for Colab)
!pip install -q xgboost openpyxl imbalanced-learn

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     cross_val_score, learning_curve, StratifiedKFold)
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, auc)
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import SMOTE
from scipy import stats

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
print("✅ All libraries imported successfully!")

# %% [markdown]
# ## 📥 Part 2: Data Loading
# **Upload all dataset files to Colab** using the file upload cell below.

# %%
from google.colab import files
print("📂 Upload ALL dataset files:")
print("  1. data.csv (Breast Cancer)")
print("  2. diabetes.csv (Pima Diabetes)")
print("  3. train.csv (Titanic)")
print("  4. Iris.csv")
print("  5. winequality-red.csv")
print("  6. Dry_Bean_Dataset.xlsx")
uploaded = files.upload()

# %%
# Load all datasets
df_cancer = pd.read_csv('data.csv')
df_diabetes = pd.read_csv('diabetes.csv')
df_titanic = pd.read_csv('train.csv')
df_iris = pd.read_csv('Iris.csv')
df_wine = pd.read_csv('winequality-red.csv')
df_beans = pd.read_excel('Dry_Bean_Dataset.xlsx')

datasets = {
    'Breast Cancer': df_cancer, 'Diabetes': df_diabetes,
    'Titanic': df_titanic, 'Iris': df_iris,
    'Wine Quality': df_wine, 'Dry Bean': df_beans
}

for name, df in datasets.items():
    print(f"\n{'='*50}")
    print(f"📊 {name}: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Missing: {df.isnull().sum().sum()} | Duplicates: {df.duplicated().sum()}")

# %% [markdown]
# ---
# # 🔍 Part 3: Exploratory Data Analysis (EDA)
# ---

# %% [markdown]
# ### 3.1 — Breast Cancer Dataset EDA

# %%
print("📊 BREAST CANCER DATASET")
print(f"Shape: {df_cancer.shape}")
print(f"\nTarget Distribution:\n{df_cancer['diagnosis'].value_counts()}")
print(f"\nData Types:\n{df_cancer.dtypes.value_counts()}")
df_cancer.describe().round(2)

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.countplot(data=df_cancer, x='diagnosis', ax=axes[0], palette='coolwarm')
axes[0].set_title('Diagnosis Distribution (M=Malignant, B=Benign)', fontsize=12)
for p in axes[0].patches:
    axes[0].annotate(f'{int(p.get_height())}', (p.get_x()+p.get_width()/2., p.get_height()),
                     ha='center', va='bottom', fontsize=11)
sns.histplot(data=df_cancer, x='radius_mean', hue='diagnosis', kde=True, ax=axes[1], palette='coolwarm')
axes[1].set_title('Radius Mean by Diagnosis')
sns.histplot(data=df_cancer, x='concavity_mean', hue='diagnosis', kde=True, ax=axes[2], palette='coolwarm')
axes[2].set_title('Concavity Mean by Diagnosis')
plt.tight_layout(); plt.show()

# %%
mean_cols = [c for c in df_cancer.columns if '_mean' in c]
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df_cancer[mean_cols].corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
ax.set_title('Correlation Heatmap — Mean Features (Breast Cancer)', fontsize=14)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 3.2 — Diabetes Dataset EDA

# %%
print("📊 DIABETES DATASET")
print(f"Shape: {df_diabetes.shape}")
print(f"\nTarget Distribution:\n{df_diabetes['Outcome'].value_counts()}")
zero_cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
print(f"\nZero values in biological features (likely missing):")
for col in zero_cols:
    print(f"  {col}: {(df_diabetes[col]==0).sum()} zeros ({(df_diabetes[col]==0).mean()*100:.1f}%)")

# %%
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for i, col in enumerate(df_diabetes.columns[:-1]):
    ax = axes[i//4, i%4]
    sns.histplot(data=df_diabetes, x=col, hue='Outcome', kde=True, ax=ax, palette='Set1')
    ax.set_title(col, fontsize=11)
plt.suptitle('Diabetes Dataset — Feature Distributions by Outcome', fontsize=14, y=1.02)
plt.tight_layout(); plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(df_diabetes.corr(), annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
ax.set_title('Correlation Heatmap — Diabetes', fontsize=14)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 3.3 — Titanic Dataset EDA

# %%
print("📊 TITANIC DATASET")
print(f"Shape: {df_titanic.shape}")
print(f"\nTarget Distribution:\n{df_titanic['Survived'].value_counts()}")
print(f"\nMissing Values:\n{df_titanic.isnull().sum()[df_titanic.isnull().sum()>0]}")

# %%
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
sns.countplot(data=df_titanic, x='Survived', ax=axes[0], palette='Set2')
axes[0].set_title('Survival Distribution')
sns.countplot(data=df_titanic, x='Pclass', hue='Survived', ax=axes[1], palette='Set2')
axes[1].set_title('Survival by Class')
sns.countplot(data=df_titanic, x='Sex', hue='Survived', ax=axes[2], palette='Set2')
axes[2].set_title('Survival by Sex')
sns.histplot(data=df_titanic, x='Age', hue='Survived', kde=True, ax=axes[3], palette='Set2')
axes[3].set_title('Survival by Age')
plt.tight_layout(); plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df_titanic, x='Pclass', y='Fare', hue='Survived', ax=axes[0], palette='Set2')
axes[0].set_title('Fare by Class & Survival')
sns.countplot(data=df_titanic, x='Embarked', hue='Survived', ax=axes[1], palette='Set2')
axes[1].set_title('Survival by Embarkation Port')
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 3.4 — Iris Dataset EDA

# %%
print("📊 IRIS DATASET")
print(f"Shape: {df_iris.shape}")
print(f"\nTarget Distribution:\n{df_iris['Species'].value_counts()}")

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, col in enumerate(['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']):
    ax = axes[i//2, i%2]
    sns.boxplot(data=df_iris, x='Species', y=col, ax=ax, palette='viridis')
    ax.set_title(f'{col} by Species', fontsize=12)
plt.suptitle('Iris Dataset — Feature Distributions', fontsize=14, y=1.02)
plt.tight_layout(); plt.show()

# %%
sns.pairplot(df_iris, hue='Species', vars=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'],
             palette='viridis', diag_kind='kde')
plt.suptitle('Iris Pairplot', y=1.02, fontsize=14); plt.show()

# %% [markdown]
# ### 3.5 — Wine Quality Dataset EDA

# %%
print("📊 WINE QUALITY DATASET")
print(f"Shape: {df_wine.shape}")
print(f"\nQuality Distribution:\n{df_wine['quality'].value_counts().sort_index()}")

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, col in enumerate(['fixed acidity','volatile acidity','citric acid','alcohol','sulphates','pH']):
    ax = axes[i//3, i%3]
    sns.boxplot(data=df_wine, x='quality', y=col, ax=ax, palette='magma')
    ax.set_title(f'{col} by Quality')
plt.suptitle('Wine Quality — Key Features', fontsize=14, y=1.02)
plt.tight_layout(); plt.show()

# %%
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df_wine.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
ax.set_title('Correlation Heatmap — Wine Quality', fontsize=14)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 3.6 — Dry Bean Dataset EDA

# %%
print("📊 DRY BEAN DATASET")
print(f"Shape: {df_beans.shape}")
print(f"\nClass Distribution:\n{df_beans['Class'].value_counts()}")

# %%
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
num_cols = df_beans.select_dtypes(include=np.number).columns[:8]
for i, col in enumerate(num_cols):
    ax = axes[i//4, i%4]
    sns.boxplot(data=df_beans, x='Class', y=col, ax=ax, palette='Set3')
    ax.tick_params(axis='x', rotation=45)
    ax.set_title(col, fontsize=11)
plt.suptitle('Dry Bean — Feature Distributions by Class', fontsize=14, y=1.02)
plt.tight_layout(); plt.show()

# %% [markdown]
# ---
# # 🧹 Part 4: Data Cleaning & Preprocessing
# ---

# %% [markdown]
# ### 4.1 — Outlier Detection & Handling (IQR Method)

# %%
def detect_outliers_iqr(df, columns):
    """Detect outliers using IQR method and return summary."""
    outlier_info = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outlier_info[col] = {'count': len(outliers), 'pct': len(outliers)/len(df)*100,
                             'lower': lower, 'upper': upper}
    return outlier_info

def cap_outliers(df, columns):
    """Cap outliers at IQR boundaries (Winsorization)."""
    df_capped = df.copy()
    for col in columns:
        Q1 = df_capped[col].quantile(0.25)
        Q3 = df_capped[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_capped[col] = df_capped[col].clip(lower, upper)
    return df_capped

# Show outlier detection for each dataset
for name, df in [('Breast Cancer', df_cancer), ('Diabetes', df_diabetes), ('Wine Quality', df_wine)]:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    info = detect_outliers_iqr(df, num_cols)
    total_outliers = sum(v['count'] for v in info.values())
    top_outliers = sorted(info.items(), key=lambda x: x[1]['count'], reverse=True)[:3]
    print(f"\n📊 {name} — Total outlier instances: {total_outliers}")
    for col, v in top_outliers:
        print(f"   {col}: {v['count']} outliers ({v['pct']:.1f}%)")

print("\n✅ Outlier detection complete! Using IQR capping (Winsorization) during preprocessing.")

# %% [markdown]
# ### 4.2 — Data Cleaning

# %%
# ============ BREAST CANCER ============
df_cancer_clean = df_cancer.drop(['id'], axis=1)
df_cancer_clean = df_cancer_clean.loc[:, ~df_cancer_clean.columns.str.contains('^Unnamed')]
df_cancer_clean['diagnosis'] = df_cancer_clean['diagnosis'].map({'M': 1, 'B': 0})
# Cap outliers
num_cols_cancer = df_cancer_clean.select_dtypes(include=np.number).columns.drop('diagnosis').tolist()
df_cancer_clean = cap_outliers(df_cancer_clean, num_cols_cancer)
print(f"✅ Breast Cancer cleaned + outliers capped: {df_cancer_clean.shape}")

# %%
# ============ DIABETES ============
df_diabetes_clean = df_diabetes.copy()
zero_cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
df_diabetes_clean[zero_cols] = df_diabetes_clean[zero_cols].replace(0, np.nan)
imputer = SimpleImputer(strategy='median')
df_diabetes_clean[zero_cols] = imputer.fit_transform(df_diabetes_clean[zero_cols])
# Cap outliers
num_cols_diab = df_diabetes_clean.columns.drop('Outcome').tolist()
df_diabetes_clean = cap_outliers(df_diabetes_clean, num_cols_diab)
print(f"✅ Diabetes cleaned + outliers capped: {df_diabetes_clean.shape}")

# %%
# ============ TITANIC ============
df_titanic_clean = df_titanic.copy()
df_titanic_clean['Age'].fillna(df_titanic_clean['Age'].median(), inplace=True)
df_titanic_clean['Embarked'].fillna(df_titanic_clean['Embarked'].mode()[0], inplace=True)
df_titanic_clean['FamilySize'] = df_titanic_clean['SibSp'] + df_titanic_clean['Parch'] + 1
df_titanic_clean['IsAlone'] = (df_titanic_clean['FamilySize'] == 1).astype(int)
df_titanic_clean['Title'] = df_titanic_clean['Name'].str.extract(r' ([A-Za-z]+)\.')
df_titanic_clean['Title'] = df_titanic_clean['Title'].replace(
    ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
df_titanic_clean['Title'] = df_titanic_clean['Title'].replace(['Mlle','Ms'], 'Miss')
df_titanic_clean['Title'] = df_titanic_clean['Title'].replace('Mme', 'Mrs')
df_titanic_clean['Sex'] = df_titanic_clean['Sex'].map({'male': 0, 'female': 1})
df_titanic_clean = pd.get_dummies(df_titanic_clean, columns=['Embarked', 'Title'], drop_first=True)
df_titanic_clean.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)
print(f"✅ Titanic cleaned: {df_titanic_clean.shape}")

# %%
# ============ IRIS ============
df_iris_clean = df_iris.drop('Id', axis=1)
le_iris = LabelEncoder()
df_iris_clean['Species_encoded'] = le_iris.fit_transform(df_iris_clean['Species'])
print(f"✅ Iris cleaned: {df_iris_clean.shape}")
print(f"   Classes: {dict(zip(le_iris.classes_, le_iris.transform(le_iris.classes_)))}")

# %%
# ============ WINE QUALITY ============
df_wine_clean = df_wine.copy()
num_cols_wine = df_wine_clean.columns.drop('quality').tolist()
df_wine_clean = cap_outliers(df_wine_clean, num_cols_wine)
print(f"✅ Wine Quality cleaned + outliers capped: {df_wine_clean.shape}")

# %%
# ============ DRY BEAN ============
df_beans_clean = df_beans.copy()
df_beans_clean.dropna(inplace=True)
le_beans = LabelEncoder()
df_beans_clean['Class_encoded'] = le_beans.fit_transform(df_beans_clean['Class'])
num_cols_beans = df_beans_clean.select_dtypes(include=np.number).columns.drop('Class_encoded').tolist()
df_beans_clean = cap_outliers(df_beans_clean, num_cols_beans)
print(f"✅ Dry Bean cleaned + outliers capped: {df_beans_clean.shape}")

# %% [markdown]
# ---
# # 🎯 Part 5: Feature Selection (SelectKBest)
# ---

# %%
def feature_selection_analysis(X, y, feature_names, dataset_name, k='all'):
    """Perform feature selection using ANOVA F-test and Mutual Information."""
    # ANOVA F-test
    selector_f = SelectKBest(score_func=f_classif, k=k)
    selector_f.fit(X, y)
    f_scores = pd.DataFrame({'Feature': feature_names, 'F-Score': selector_f.scores_,
                             'p-value': selector_f.pvalues_}).sort_values('F-Score', ascending=False)

    # Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({'Feature': feature_names, 'MI-Score': mi_scores}).sort_values('MI-Score', ascending=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    top_n = min(15, len(feature_names))
    sns.barplot(data=f_scores.head(top_n), x='F-Score', y='Feature', ax=axes[0], palette='viridis')
    axes[0].set_title(f'{dataset_name} — Top Features (ANOVA F-Score)')
    sns.barplot(data=mi_df.head(top_n), x='MI-Score', y='Feature', ax=axes[1], palette='magma')
    axes[1].set_title(f'{dataset_name} — Top Features (Mutual Information)')
    plt.tight_layout(); plt.show()

    # Print top features
    print(f"\n🏆 {dataset_name} — Top 5 Features:")
    for i, row in f_scores.head(5).iterrows():
        print(f"   {row['Feature']}: F={row['F-Score']:.2f}, p={row['p-value']:.2e}")
    return f_scores

# %%
# Breast Cancer
X_cancer = df_cancer_clean.drop('diagnosis', axis=1)
fs_cancer = feature_selection_analysis(X_cancer.values, df_cancer_clean['diagnosis'],
                                       X_cancer.columns.tolist(), 'Breast Cancer')

# %%
# Diabetes
X_diab = df_diabetes_clean.drop('Outcome', axis=1)
fs_diabetes = feature_selection_analysis(X_diab.values, df_diabetes_clean['Outcome'],
                                          X_diab.columns.tolist(), 'Diabetes')

# %%
# Iris
X_iris_fs = df_iris_clean[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
fs_iris = feature_selection_analysis(X_iris_fs.values, df_iris_clean['Species_encoded'],
                                      X_iris_fs.columns.tolist(), 'Iris')

# %%
# Wine Quality
X_wine_fs = df_wine_clean.drop('quality', axis=1)
fs_wine = feature_selection_analysis(X_wine_fs.values, df_wine_clean['quality'],
                                      X_wine_fs.columns.tolist(), 'Wine Quality')

print("\n✅ Feature selection analysis complete for all datasets!")
# %% [markdown]
# ---
# # 🤖 Part 6: Enhanced Model Training & Evaluation Pipeline
# ---
# Includes: Cross-Validation, SMOTE, Classification Reports, Feature Importance, Learning Curves

# %%
def get_models():
    """Return dictionary of all models to train."""
    return {
        'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)
    }

def train_evaluate(X_train, X_test, y_train, y_test, dataset_name, task='binary', use_smote=False):
    """Train all models with optional SMOTE, cross-validation & return results."""
    models = get_models()
    results = []
    trained = {}
    cv_results = {}

    # Apply SMOTE if requested (for imbalanced datasets)
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_fit, y_train_fit = smote.fit_resample(X_train, y_train)
        print(f"  📊 SMOTE applied: {len(y_train)} → {len(y_train_fit)} samples")
    else:
        X_train_fit, y_train_fit = X_train, y_train

    for name, model in models.items():
        # Cross-validation (5-fold)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = 'f1' if task == 'binary' else 'f1_weighted'
        cv_scores = cross_val_score(model, X_train_fit, y_train_fit, cv=cv, scoring=scoring)
        cv_results[name] = cv_scores

        # Train on full training set
        model.fit(X_train_fit, y_train_fit)
        y_pred = model.predict(X_test)
        trained[name] = model

        avg = 'binary' if task == 'binary' else 'weighted'
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
        rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)

        try:
            if task == 'binary':
                y_prob = model.predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test, y_prob)
            else:
                y_prob = model.predict_proba(X_test)
                n_classes = len(np.unique(y_test))
                y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
                if y_test_bin.shape[1] == 1:
                    y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])
                auc_score = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='weighted')
        except:
            auc_score = np.nan

        results.append({
            'Model': name, 'Accuracy': acc, 'Precision': prec,
            'Recall': rec, 'F1-Score': f1, 'ROC-AUC': auc_score,
            'CV-Mean': cv_scores.mean(), 'CV-Std': cv_scores.std()
        })
        print(f"  ✅ {name}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc_score:.4f}, CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

    return pd.DataFrame(results), trained, cv_results

# %%
def plot_results(results_df, dataset_name):
    """Plot model comparison charts."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    results_melted = results_df[['Model','Accuracy','Precision','Recall','F1-Score','ROC-AUC']].melt(
        id_vars='Model', var_name='Metric', value_name='Score')
    sns.barplot(data=results_melted, x='Model', y='Score', hue='Metric', ax=axes[0], palette='viridis')
    axes[0].set_title(f'{dataset_name} — Model Comparison', fontsize=13)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha='right')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    heatmap_data = results_df.set_index('Model')[['Accuracy','Precision','Recall','F1-Score','ROC-AUC']]
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGn', ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title(f'{dataset_name} — Scores Heatmap', fontsize=13)
    plt.tight_layout(); plt.show()

def plot_cv_comparison(cv_results, dataset_name):
    """Plot cross-validation score distributions."""
    fig, ax = plt.subplots(figsize=(10, 5))
    cv_data = pd.DataFrame(cv_results)
    ax.boxplot(cv_data.values, labels=cv_data.columns)
    ax.set_title(f'{dataset_name} — 5-Fold Cross-Validation Scores', fontsize=13)
    ax.set_ylabel('F1-Score')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout(); plt.show()

def plot_confusion_matrices(X_test, y_test, trained_models, dataset_name):
    """Plot confusion matrices for all models."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, (name, model) in enumerate(trained_models.items()):
        ax = axes[idx//3, idx%3]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{name}', fontsize=11)
        ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
    plt.suptitle(f'{dataset_name} — Confusion Matrices', fontsize=14, y=1.02)
    plt.tight_layout(); plt.show()

def plot_roc_curves(X_test, y_test, trained_models, dataset_name):
    """Plot ROC curves for binary classification."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in trained_models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
    ax.plot([0,1], [0,1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{dataset_name} — ROC Curves', fontsize=14)
    ax.legend(loc='lower right'); plt.tight_layout(); plt.show()

def print_classification_reports(X_test, y_test, trained_models, dataset_name, target_names=None):
    """Print detailed classification reports for all models."""
    print(f"\n{'='*60}")
    print(f"📋 CLASSIFICATION REPORTS — {dataset_name}")
    print(f"{'='*60}")
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        print(f"\n🔹 {name}:")
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

def plot_feature_importance(trained_models, feature_names, dataset_name):
    """Plot feature importance for tree-based models."""
    tree_models = {n: m for n, m in trained_models.items()
                   if hasattr(m, 'feature_importances_')}
    if not tree_models:
        return
    n = len(tree_models)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 6))
    if n == 1: axes = [axes]
    for idx, (name, model) in enumerate(tree_models.items()):
        imp = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_})
        imp = imp.sort_values('Importance', ascending=True).tail(15)
        axes[idx].barh(imp['Feature'], imp['Importance'], color=plt.cm.viridis(np.linspace(0.3, 0.9, len(imp))))
        axes[idx].set_title(f'{name} — Feature Importance', fontsize=12)
        axes[idx].set_xlabel('Importance')
    plt.suptitle(f'{dataset_name} — Feature Importance (Top 15)', fontsize=14, y=1.02)
    plt.tight_layout(); plt.show()

def plot_learning_curves(model, model_name, X_train, y_train, dataset_name, task='binary'):
    """Plot learning curves to check overfitting/underfitting."""
    scoring = 'f1' if task == 'binary' else 'f1_weighted'
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42, n_jobs=-1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training Score', color='#2196F3')
    ax.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15, color='#2196F3')
    ax.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation Score', color='#E91E63')
    ax.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                    val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.15, color='#E91E63')
    ax.set_xlabel('Training Set Size'); ax.set_ylabel('F1-Score')
    ax.set_title(f'{dataset_name} — Learning Curve ({model_name})', fontsize=13)
    ax.legend(); plt.tight_layout(); plt.show()

print("✅ Enhanced pipeline functions defined!")

# %% [markdown]
# ---
# # 🎯 Part 7: Binary Classification — Training & Evaluation
# ---

# %% [markdown]
# ### 7.1 — Breast Cancer Classification

# %%
print("="*60)
print("🔬 BREAST CANCER — Binary Classification")
print("="*60)
X = df_cancer_clean.drop('diagnosis', axis=1)
y = df_cancer_clean['diagnosis']
feature_names_cancer = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

results_cancer, models_cancer, cv_cancer = train_evaluate(
    X_train_sc, X_test_sc, y_train, y_test, 'Breast Cancer', 'binary')
print(f"\n{results_cancer.to_string(index=False)}")

# %%
plot_results(results_cancer, 'Breast Cancer')
plot_cv_comparison(cv_cancer, 'Breast Cancer')
plot_confusion_matrices(X_test_sc, y_test, models_cancer, 'Breast Cancer')
plot_roc_curves(X_test_sc, y_test, models_cancer, 'Breast Cancer')
print_classification_reports(X_test_sc, y_test, models_cancer, 'Breast Cancer', ['Benign', 'Malignant'])
plot_feature_importance(models_cancer, feature_names_cancer, 'Breast Cancer')

# %%
# Learning curve for best model
best_name = results_cancer.loc[results_cancer['F1-Score'].idxmax(), 'Model']
plot_learning_curves(get_models()[best_name], best_name, X_train_sc, y_train, 'Breast Cancer')

# %% [markdown]
# ### 7.2 — Diabetes Classification (with SMOTE for class imbalance)

# %%
print("="*60)
print("🩺 DIABETES — Binary Classification (with SMOTE)")
print("="*60)
X = df_diabetes_clean.drop('Outcome', axis=1)
y = df_diabetes_clean['Outcome']
feature_names_diab = X.columns.tolist()
print(f"Class imbalance: {dict(y.value_counts())} — applying SMOTE")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

results_diabetes, models_diabetes, cv_diabetes = train_evaluate(
    X_train_sc, X_test_sc, y_train, y_test, 'Diabetes', 'binary', use_smote=True)
print(f"\n{results_diabetes.to_string(index=False)}")

# %%
plot_results(results_diabetes, 'Diabetes')
plot_cv_comparison(cv_diabetes, 'Diabetes')
plot_confusion_matrices(X_test_sc, y_test, models_diabetes, 'Diabetes')
plot_roc_curves(X_test_sc, y_test, models_diabetes, 'Diabetes')
print_classification_reports(X_test_sc, y_test, models_diabetes, 'Diabetes', ['No Diabetes', 'Diabetes'])
plot_feature_importance(models_diabetes, feature_names_diab, 'Diabetes')

# %%
best_name = results_diabetes.loc[results_diabetes['F1-Score'].idxmax(), 'Model']
plot_learning_curves(get_models()[best_name], best_name, X_train_sc, y_train, 'Diabetes')

# %% [markdown]
# ### 7.3 — Titanic Survival Prediction (with SMOTE)

# %%
print("="*60)
print("🚢 TITANIC — Binary Classification (with SMOTE)")
print("="*60)
X = df_titanic_clean.drop('Survived', axis=1)
y = df_titanic_clean['Survived']
feature_names_titanic = X.columns.tolist()
print(f"Class imbalance: {dict(y.value_counts())} — applying SMOTE")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

results_titanic, models_titanic, cv_titanic = train_evaluate(
    X_train_sc, X_test_sc, y_train, y_test, 'Titanic', 'binary', use_smote=True)
print(f"\n{results_titanic.to_string(index=False)}")

# %%
plot_results(results_titanic, 'Titanic')
plot_cv_comparison(cv_titanic, 'Titanic')
plot_confusion_matrices(X_test_sc, y_test, models_titanic, 'Titanic')
plot_roc_curves(X_test_sc, y_test, models_titanic, 'Titanic')
print_classification_reports(X_test_sc, y_test, models_titanic, 'Titanic', ['Died', 'Survived'])
plot_feature_importance(models_titanic, feature_names_titanic, 'Titanic')

# %%
best_name = results_titanic.loc[results_titanic['F1-Score'].idxmax(), 'Model']
plot_learning_curves(get_models()[best_name], best_name, X_train_sc, y_train, 'Titanic')

# %% [markdown]
# ---
# # 🌈 Part 8: Multi-Class Classification — Training & Evaluation
# ---

# %% [markdown]
# ### 8.1 — Iris Classification

# %%
print("="*60)
print("🌸 IRIS — Multi-Class Classification")
print("="*60)
X = df_iris_clean[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = df_iris_clean['Species_encoded']
feature_names_iris = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

results_iris, models_iris, cv_iris = train_evaluate(
    X_train_sc, X_test_sc, y_train, y_test, 'Iris', 'multi')
print(f"\n{results_iris.to_string(index=False)}")

# %%
plot_results(results_iris, 'Iris')
plot_cv_comparison(cv_iris, 'Iris')
plot_confusion_matrices(X_test_sc, y_test, models_iris, 'Iris')
print_classification_reports(X_test_sc, y_test, models_iris, 'Iris', le_iris.classes_.tolist())
plot_feature_importance(models_iris, feature_names_iris, 'Iris')

# %%
best_name = results_iris.loc[results_iris['F1-Score'].idxmax(), 'Model']
plot_learning_curves(get_models()[best_name], best_name, X_train_sc, y_train, 'Iris', 'multi')

# %% [markdown]
# ### 8.2 — Wine Quality Classification (with SMOTE)

# %%
print("="*60)
print("🍷 WINE QUALITY — Multi-Class Classification (with SMOTE)")
print("="*60)
X = df_wine_clean.drop('quality', axis=1)
y = df_wine_clean['quality']
le_wine = LabelEncoder()
y = le_wine.fit_transform(y)
feature_names_wine = X.columns.tolist()
print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))} — applying SMOTE")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

results_wine, models_wine, cv_wine = train_evaluate(
    X_train_sc, X_test_sc, y_train, y_test, 'Wine Quality', 'multi', use_smote=True)
print(f"\n{results_wine.to_string(index=False)}")

# %%
plot_results(results_wine, 'Wine Quality')
plot_cv_comparison(cv_wine, 'Wine Quality')
plot_confusion_matrices(X_test_sc, y_test, models_wine, 'Wine Quality')
print_classification_reports(X_test_sc, y_test, models_wine, 'Wine Quality')
plot_feature_importance(models_wine, feature_names_wine, 'Wine Quality')

# %% [markdown]
# ### 8.3 — Dry Bean Classification

# %%
print("="*60)
print("🫘 DRY BEAN — Multi-Class Classification")
print("="*60)
X = df_beans_clean.drop(['Class', 'Class_encoded'], axis=1)
y = df_beans_clean['Class_encoded']
feature_names_beans = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

results_beans, models_beans, cv_beans = train_evaluate(
    X_train_sc, X_test_sc, y_train, y_test, 'Dry Bean', 'multi')
print(f"\n{results_beans.to_string(index=False)}")

# %%
plot_results(results_beans, 'Dry Bean')
plot_cv_comparison(cv_beans, 'Dry Bean')
plot_confusion_matrices(X_test_sc, y_test, models_beans, 'Dry Bean')
print_classification_reports(X_test_sc, y_test, models_beans, 'Dry Bean', le_beans.classes_.tolist())
plot_feature_importance(models_beans, feature_names_beans, 'Dry Bean')
# %% [markdown]
# ---
# # ⚡ Part 9: Hyperparameter Tuning (GridSearchCV)
# ---

# %%
param_grids = {
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
}

def tune_best_model(X_train, X_test, y_train, y_test, results_df, dataset_name, task='binary'):
    """Find best model and tune it with GridSearchCV."""
    best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
    print(f"\n🏆 Best model for {dataset_name}: {best_model_name}")

    if best_model_name in param_grids:
        tune_name = best_model_name
    else:
        tune_name = 'Random Forest'
        print(f"   → Tuning Random Forest instead (as proxy)")

    if tune_name == 'Random Forest':
        base_model = RandomForestClassifier(random_state=42)
    elif tune_name == 'XGBoost':
        base_model = XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)
    else:
        base_model = SVC(probability=True, random_state=42)

    scoring = 'f1' if task == 'binary' else 'f1_weighted'
    grid = GridSearchCV(base_model, param_grids[tune_name], cv=5, scoring=scoring, n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    print(f"   Best params: {grid.best_params_}")
    print(f"   Best CV Score: {grid.best_score_:.4f}")

    y_pred = grid.best_estimator_.predict(X_test)
    avg = 'binary' if task == 'binary' else 'weighted'
    tuned_acc = accuracy_score(y_test, y_pred)
    tuned_f1 = f1_score(y_test, y_pred, average=avg)
    print(f"   Test Accuracy: {tuned_acc:.4f}")
    print(f"   Test F1-Score: {tuned_f1:.4f}")

    return grid.best_estimator_, grid.best_params_, grid.best_score_, tuned_acc, tuned_f1

print("✅ Hyperparameter tuning function defined!")

# %% [markdown]
# ### 9.1 — Tune Binary Classification Models

# %%
print("="*60 + "\n🔬 Tuning BREAST CANCER")
X = df_cancer_clean.drop('diagnosis', axis=1); y = df_cancer_clean['diagnosis']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
best_cancer, params_cancer, cv_cancer_t, acc_c, f1_c = tune_best_model(X_tr_s, X_te_s, y_tr, y_te, results_cancer, 'Breast Cancer')

# %%
print("="*60 + "\n🩺 Tuning DIABETES")
X = df_diabetes_clean.drop('Outcome', axis=1); y = df_diabetes_clean['Outcome']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
best_diabetes, params_diab, cv_diab_t, acc_d, f1_d = tune_best_model(X_tr_s, X_te_s, y_tr, y_te, results_diabetes, 'Diabetes')

# %%
print("="*60 + "\n🚢 Tuning TITANIC")
X = df_titanic_clean.drop('Survived', axis=1); y = df_titanic_clean['Survived']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
best_titanic, params_tit, cv_tit_t, acc_t, f1_t = tune_best_model(X_tr_s, X_te_s, y_tr, y_te, results_titanic, 'Titanic')

# %% [markdown]
# ### 9.2 — Tune Multi-Class Classification Models

# %%
print("="*60 + "\n🌸 Tuning IRIS")
X = df_iris_clean[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = df_iris_clean['Species_encoded']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
best_iris, params_iris_t, cv_iris_t, acc_i, f1_i = tune_best_model(X_tr_s, X_te_s, y_tr, y_te, results_iris, 'Iris', 'multi')

# %%
print("="*60 + "\n🍷 Tuning WINE QUALITY")
X = df_wine_clean.drop('quality', axis=1); y_raw = df_wine_clean['quality']
le_w = LabelEncoder(); y = le_w.fit_transform(y_raw)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
best_wine, params_wine_t, cv_wine_t, acc_w, f1_w = tune_best_model(X_tr_s, X_te_s, y_tr, y_te, results_wine, 'Wine Quality', 'multi')

# %%
print("="*60 + "\n🫘 Tuning DRY BEAN")
X = df_beans_clean.drop(['Class','Class_encoded'], axis=1); y = df_beans_clean['Class_encoded']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
best_beans, params_beans_t, cv_beans_t, acc_b, f1_b = tune_best_model(X_tr_s, X_te_s, y_tr, y_te, results_beans, 'Dry Bean', 'multi')

# %% [markdown]
# ---
# # 💾 Part 10: Save Best Models (joblib)
# ---

# %%
import joblib

best_models_dict = {
    'breast_cancer_model': best_cancer,
    'diabetes_model': best_diabetes,
    'titanic_model': best_titanic,
    'iris_model': best_iris,
    'wine_quality_model': best_wine,
    'dry_bean_model': best_beans
}

for name, model in best_models_dict.items():
    filename = f'{name}.joblib'
    joblib.dump(model, filename)
    print(f"💾 Saved: {filename}")

print("\n✅ All 6 tuned models saved! You can download them from the Files panel.")
print("   To load: model = joblib.load('breast_cancer_model.joblib')")

# %% [markdown]
# ---
# # 📊 Part 11: Grand Comparison & Final Summary
# ---

# %%
all_results = {
    'Breast Cancer': results_cancer, 'Diabetes': results_diabetes,
    'Titanic': results_titanic, 'Iris': results_iris,
    'Wine Quality': results_wine, 'Dry Bean': results_beans
}

# Grand comparison chart
fig, axes = plt.subplots(2, 3, figsize=(24, 12))
for idx, (name, res) in enumerate(all_results.items()):
    ax = axes[idx//3, idx%3]
    x = np.arange(len(res))
    w = 0.13
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i*w, res[metric], w, label=metric, color=color, alpha=0.85)
    ax.set_xticks(x + 2*w)
    ax.set_xticklabels(res['Model'], rotation=35, ha='right', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title(f'📊 {name}', fontsize=13, fontweight='bold')
    if idx == 0: ax.legend(fontsize=8, loc='lower right')
plt.suptitle('🏆 Grand Model Comparison Across All Datasets', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout(); plt.show()

# %%
# Cross-Validation stability comparison
all_cv = {'Breast Cancer': cv_cancer, 'Diabetes': cv_diabetes, 'Titanic': cv_titanic,
           'Iris': cv_iris, 'Wine Quality': cv_wine, 'Dry Bean': cv_beans}
fig, axes = plt.subplots(2, 3, figsize=(22, 10))
for idx, (name, cv_res) in enumerate(all_cv.items()):
    ax = axes[idx//3, idx%3]
    cv_data = pd.DataFrame(cv_res)
    ax.boxplot(cv_data.values, labels=cv_data.columns)
    ax.set_title(f'{name}', fontsize=12, fontweight='bold')
    ax.set_ylabel('CV F1-Score')
    ax.tick_params(axis='x', rotation=35)
plt.suptitle('📦 Cross-Validation Score Distributions', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(); plt.show()

# %%
# Best model per dataset summary
print("\n" + "="*70)
print("🏆 FINAL SUMMARY — BEST MODELS PER DATASET")
print("="*70)
summary_data = []
for name, res in all_results.items():
    best_idx = res['F1-Score'].idxmax()
    best = res.iloc[best_idx]
    summary_data.append({
        'Dataset': name, 'Best Model': best['Model'],
        'Accuracy': f"{best['Accuracy']:.4f}", 'F1-Score': f"{best['F1-Score']:.4f}",
        'ROC-AUC': f"{best['ROC-AUC']:.4f}", 'CV-Mean': f"{best['CV-Mean']:.4f}",
        'CV-Std': f"±{best['CV-Std']:.4f}"
    })
summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# %%
# Visual summary
fig, ax = plt.subplots(figsize=(14, 6))
datasets_list = summary_df['Dataset'].tolist()
f1_scores = [float(x) for x in summary_df['F1-Score']]
acc_scores = [float(x) for x in summary_df['Accuracy']]
auc_scores = [float(x) for x in summary_df['ROC-AUC']]
cv_scores = [float(x) for x in summary_df['CV-Mean']]

x = np.arange(len(datasets_list))
w = 0.2
ax.bar(x - 1.5*w, acc_scores, w, label='Accuracy', color='#2196F3', alpha=0.85)
ax.bar(x - 0.5*w, f1_scores, w, label='F1-Score', color='#E91E63', alpha=0.85)
ax.bar(x + 0.5*w, auc_scores, w, label='ROC-AUC', color='#9C27B0', alpha=0.85)
ax.bar(x + 1.5*w, cv_scores, w, label='CV-Mean', color='#4CAF50', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(datasets_list, rotation=20, ha='right')
ax.set_ylim(0, 1.15)
ax.set_title('🏆 Best Model Performance Across All Datasets', fontsize=14, fontweight='bold')
ax.legend()
for i in range(len(datasets_list)):
    ax.annotate(summary_df['Best Model'].iloc[i],
                (x[i], max(acc_scores[i], f1_scores[i], auc_scores[i], cv_scores[i]) + 0.02),
                ha='center', fontsize=8, fontweight='bold', color='green')
plt.tight_layout(); plt.show()

# %%
print("\n" + "="*70)
print("✅ PROJECT COMPLETE!")
print("="*70)
print("""
📋 Complete ML Pipeline Steps Covered:
  ✅ EDA (Exploratory Data Analysis) — 6 datasets
  ✅ Outlier Detection & Handling (IQR Capping/Winsorization)
  ✅ Data Cleaning & Missing Value Imputation
  ✅ Feature Engineering (Titanic: Title, FamilySize, IsAlone)
  ✅ Feature Selection (ANOVA F-test + Mutual Information)
  ✅ Class Imbalance Handling (SMOTE)
  ✅ Feature Scaling (StandardScaler)
  ✅ 6 ML Models × 6 Datasets = 36 Models Trained
  ✅ 5-Fold Stratified Cross-Validation
  ✅ Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC
  ✅ Detailed Classification Reports (per-class)
  ✅ Confusion Matrices & ROC Curves
  ✅ Feature Importance Plots (tree-based models)
  ✅ Learning Curves (overfitting/underfitting check)
  ✅ Model Comparison Charts & Heatmaps
  ✅ Hyperparameter Tuning (GridSearchCV)
  ✅ Model Saving (joblib)
  ✅ Final Summary Report

📊 Datasets processed:
  Binary:     Breast Cancer | Diabetes | Titanic
  Multi-Class: Iris | Wine Quality | Dry Bean
""")
