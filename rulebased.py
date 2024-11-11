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
X = train_df.drop(columns=['Loan_Status'])
y = train_df['Loan_Status']

# Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Rule-based approach function
def rule_based_loan_approval(row):
    # Example rule for loan approval
    if row['LoanAmount'] < 200 and row['Self_Employed'] == 'No' and int(row['Dependents']) <= 2:
        return 'Approved'
    else:
        return 'Rejected'

# Applying rule-based approach to the dataset
train_df['Rule_Based_Prediction'] = train_df.apply(rule_based_loan_approval, axis=1)

# Displaying first few results of the rule-based prediction
print(train_df[['Rule_Based_Prediction']].head())

# Machine Learning Models
# 1. Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_val)

# 2. Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_val)

# 3. Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_val)

# Evaluation
print("Decision Tree Metrics:")
print(f"Accuracy: {accuracy_score(y_val, y_pred_dt)}")
print(f"Precision: {precision_score(y_val, y_pred_dt, pos_label='Y')}")
print(f"Recall: {recall_score(y_val, y_pred_dt, pos_label='Y')}")
print(f"F1 Score: {f1_score(y_val, y_pred_dt, pos_label='Y')}")

print("\nLogistic Regression Metrics:")
print(f"Accuracy: {accuracy_score(y_val, y_pred_lr)}")
print(f"Precision: {precision_score(y_val, y_pred_lr, pos_label='Y')}")
print(f"Recall: {recall_score(y_val, y_pred_lr, pos_label='Y')}")
print(f"F1 Score: {f1_score(y_val, y_pred_lr, pos_label='Y')}")

print("\nRandom Forest Metrics:")
print(f"Accuracy: {accuracy_score(y_val, y_pred_rf)}")
print(f"Precision: {precision_score(y_val, y_pred_rf, pos_label='Y')}")
print(f"Recall: {recall_score(y_val, y_pred_rf, pos_label='Y')}")
print(f"F1 Score: {f1_score(y_val, y_pred_rf, pos_label='Y')}")
