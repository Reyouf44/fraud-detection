# Fraud Detection in Financial Transactions

Analyzing credit card transaction patterns to detect fraud with machine learning.

Big Data course project (ICS 474) at King Fahd University of Petroleum & Minerals (KFUPM), by **Riuof Albogmi**.

## Overview

Fraud is rare but costly. In this dataset only about **0.58%** of transactions are fraudulent, so a model that always predicts "not fraud" would look over 99% accurate while catching nothing. This project explores the data, prepares it for modeling, and compares models using metrics that matter for rare events: precision, recall and F1-score.

## Dataset

- **Source:** [Credit Card Transactions Dataset](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset) on Kaggle
- **Size:** about 1.3 million transactions, 24 columns
- **Target:** `is_fraud` (1 = fraud, 0 = legitimate)
- **Fraud rate:** about 0.58%, a highly imbalanced dataset

The dataset is not included in this repository. Download it from Kaggle to run the notebook.

## Approach

1. **Exploratory analysis:** missing values, duplicates, summary statistics, distributions, correlations and outliers
2. **Cleaning:** dropped personal and redundant fields (names, card number, street, merchant zip code), filled missing values, and clipped extreme transaction amounts to the 1st and 99th percentiles
3. **Encoding and scaling:** label encoding for high-cardinality categories, one-hot encoding for gender, and `RobustScaler` for numeric features because of extreme outliers in transaction amounts
4. **Feature importance:** a Random Forest to find which features drive fraud predictions
5. **Modeling:** a stratified 80/20 train/test split, comparing three models:
   - Logistic Regression with balanced class weights
   - Random Forest with all features
   - Random Forest with the top 10 features only

## Key findings

- **Random Forest clearly outperformed Logistic Regression**, which had very low precision and F1-score because a linear model can't capture the complex patterns in fraud.
- **Random Forest achieved high precision, with recall of about 55%.** It rarely flags legitimate transactions, but it still misses some fraud, so improving recall is the main next step.
- **Transaction amount and category drive about 78% of the model's predictive power**, with amount alone at about 65%.
- **A Random Forest using only the top 10 features performed almost as well** as the full model, so a smaller, faster and easier-to-explain model is possible.

## Future work

- Improve recall with resampling (for example SMOTE) or by tuning the decision threshold
- Engineer time-based features, such as hour of day and spending velocity
- Try gradient boosting models such as XGBoost or LightGBM

## Repository contents

| File | Description |
|---|---|
| `fraud_detection.ipynb` | Full analysis and modeling notebook |

## Tools

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
