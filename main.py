# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load datasets
train_df = pd.read_csv('/Users/andreduran/Documents/School/CS471/FinalProject/train.csv')
test_df = pd.read_csv('/Users/andreduran/Documents/School/CS471/FinalProject/test.csv')

# Data Preprocessing on training data
# 1. Handle missing values
train_df.fillna({
    'Gender': train_df['Gender'].mode()[0],
    'Married': train_df['Married'].mode()[0],
    'Dependents': train_df['Dependents'].mode()[0],
    'Self_Employed': train_df['Self_Employed'].mode()[0],
    'LoanAmount': train_df['LoanAmount'].mean(),
    'Loan_Amount_Term': train_df['Loan_Amount_Term'].mode()[0],
    'Credit_History': train_df['Credit_History'].mode()[0]
}, inplace=True)

# Encode categorical variables
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
for col in categorical_cols:
    train_df[col] = LabelEncoder().fit_transform(train_df[col])

# Feature Engineering
train_df['Income_Loan_Ratio'] = (train_df['ApplicantIncome'] + train_df['CoapplicantIncome']) / train_df['LoanAmount']

# Split into features and target variable
X_train = train_df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y_train = train_df['Loan_Status'].apply(lambda x: 1 if x == 'Y' else 0)

# Preprocess test data (same steps as training data)
test_df.fillna({
    'Gender': test_df['Gender'].mode()[0],
    'Married': test_df['Married'].mode()[0],
    'Dependents': test_df['Dependents'].mode()[0],
    'Self_Employed': test_df['Self_Employed'].mode()[0],
    'LoanAmount': test_df['LoanAmount'].mean(),
    'Loan_Amount_Term': test_df['Loan_Amount_Term'].mode()[0],
    'Credit_History': test_df['Credit_History'].mode()[0]
}, inplace=True)

for col in categorical_cols:
    test_df[col] = LabelEncoder().fit_transform(test_df[col])

test_df['Income_Loan_Ratio'] = (test_df['ApplicantIncome'] + test_df['CoapplicantIncome']) / test_df['LoanAmount']
X_test = test_df.drop(['Loan_ID'], axis=1)

# Standardize numerical features
scaler = StandardScaler()
X_train[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Income_Loan_Ratio']] = scaler.fit_transform(
    X_train[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Income_Loan_Ratio']]
)
X_test[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Income_Loan_Ratio']] = scaler.transform(
    X_test[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Income_Loan_Ratio']]
)

# Rule-based approach
def rule_based_model(row):
    if row['Credit_History'] == 1 and row['Income_Loan_Ratio'] > 0.5:
        return 1  # Approve
    else:
        return 0  # Decline

# Machine Learning Models
# Model 1: Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Model 2: Logistic Regression with further increased max_iter and alternative solver
lr_model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')  # Increased max_iter and changed solver
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Model 3: Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Compile predictions into a DataFrame for submission or further analysis
predictions_df = pd.DataFrame({
    'Loan_ID': test_df['Loan_ID'],
    'DecisionTree_Prediction': y_pred_dt,
    'LogisticRegression_Prediction': y_pred_lr,
    'RandomForest_Prediction': y_pred_rf
})

# Save predictions
predictions_df.to_csv('loan_predictions.csv', index=False)
print("Predictions saved to 'loan_predictions.csv'")
