# 🏦 Bank Customer Churn Prediction

> End-to-end machine learning classification project using Python and Scikit-learn to predict bank customer churn.

## Overview

Customer churn is an important business problem for banks because retaining existing customers can be more efficient than acquiring new ones. This project builds a classification pipeline to predict whether a bank customer is likely to exit.

The project covers the full workflow:

**Data inspection → EDA → Feature Engineering → Preprocessing → Model Comparison → Cross-Validation → Test Evaluation → Prediction**

## Dataset

The original analysis used a **10,000-row, 13-column** banking customer dataset. The target variable is `Exited`.

- `0` = No churn
- `1` = Churn
- Churned customers in the original analysis: **2,038 / 10,000 (20.38%)**

The raw dataset is not included in this repository. See `data/README.md` for how to add it locally.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Machine Learning Workflow

### 1. Exploratory Data Analysis

The analysis investigates customer characteristics and their relationship with churn, including:

- Credit score
- Age
- Geography
- Gender
- Tenure
- Balance
- Number of products
- Active membership
- Estimated salary

### 2. Feature Engineering

The project creates:

- `Balance per product`
- `Salary to Balance ratio`
- `AgeGroup`

### 3. Preprocessing

A Scikit-learn `ColumnTransformer` handles numerical and categorical variables.

**Numerical pipeline**
- Median imputation
- Standardization with `StandardScaler`

**Categorical pipeline**
- Most-frequent imputation
- One-hot encoding

### 4. Models Compared

Five classification algorithms were evaluated:

| Model | Mean 5-Fold ROC-AUC | Std. Dev. |
|---|---:|---:|
| Logistic Regression | 0.7813 | 0.0098 |
| Random Forest | 0.8501 | 0.0096 |
| **Gradient Boosting** | **0.8611** | **0.0078** |
| SVM | 0.8224 | 0.0166 |
| KNN | 0.7821 | 0.0154 |

### 5. Final Test Evaluation

The Gradient Boosting pipeline was selected based on the highest mean cross-validated ROC-AUC.

On the held-out 20% test set, the original notebook reported:

| Metric | Score |
|---|---:|
| Accuracy | **86.60%** |
| Precision | **75.93%** |
| Recall | **50.25%** |
| F1 Score | **60.47%** |
| ROC-AUC | **87.75%** |

The project uses multiple evaluation metrics because the churn class represents only 20.38% of the dataset.

## Repository Structure

```text
bank-customer-churn-prediction/
├── data/
│   └── README.md
├── images/
├── models/
├── notebooks/
│   └── bank_customer_churn_prediction_clean.ipynb
├── src/
│   └── README.md
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Julzthecoder/bank-customer-churn-prediction.git
cd bank-customer-churn-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Place your CSV in:

```text
data/CUSTOMER CHURN RECORDS.csv
```

### 4. Open the notebook

```bash
jupyter notebook
```

Then open:

```text
notebooks/bank_customer_churn_prediction_clean.ipynb
```

## Results Visuals

### Model Comparison

![Model comparison](images/model_comparison_roc_auc.png)

### Confusion Matrix

![Confusion matrix](images/confusion_matrix.png)

## Skills Demonstrated

**Python:** Pandas, NumPy, Matplotlib, Seaborn

**Machine Learning:** classification, feature engineering, preprocessing pipelines, cross-validation, model comparison, evaluation

**Predictive Analytics:** probability prediction, ROC-AUC, precision, recall, F1

**Data Science:** EDA, data preparation, reproducible workflows, model persistence

## Future Improvements

- Hyperparameter tuning
- Threshold optimization
- Feature importance and model interpretation
- Experiment tracking
- Deployment as a small web application/API
- Monitoring model performance on new data

## Author

**Julz Aumah**

Junior Data Scientist | Python | Machine Learning | SQL | Predictive Analytics

GitHub: https://github.com/Julzthecoder
