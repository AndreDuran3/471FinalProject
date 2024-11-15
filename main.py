# Import necessary libraries
import pandas as pd  # Used for data manipulation and analysis
import numpy as np  # Used for numerical operations
from sklearn.model_selection import train_test_split  # Used to split data into training and test sets
from sklearn.preprocessing import StandardScaler, LabelEncoder  # StandardScaler standardizes features; LabelEncoder encodes categorical variables
from sklearn.tree import DecisionTreeClassifier  # Decision Tree classifier
from sklearn.linear_model import LogisticRegression  # Logistic Regression classifier
from sklearn.ensemble import RandomForestClassifier  # Random Forest classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # Evaluation metrics

# Load datasets
train_df = pd.read_csv('/Users/andreduran/Documents/School/CS471/FinalProject/train.csv')  # Load training data
test_df = pd.read_csv('/Users/andreduran/Documents/School/CS471/FinalProject/test.csv')  # Load test data

# Data Preprocessing on training data
# 1. Handle missing values
train_df.fillna({
    'Gender': train_df['Gender'].mode()[0],  # Fill missing 'Gender' values with the mode
    'Married': train_df['Married'].mode()[0],  # Fill missing 'Married' values with the mode
    'Dependents': train_df['Dependents'].mode()[0],  # Fill missing 'Dependents' values with the mode
    'Self_Employed': train_df['Self_Employed'].mode()[0],  # Fill missing 'Self_Employed' values with the mode
    'LoanAmount': train_df['LoanAmount'].mean(),  # Fill missing 'LoanAmount' values with the mean
    'Loan_Amount_Term': train_df['Loan_Amount_Term'].mode()[0],  # Fill missing 'Loan_Amount_Term' values with the mode
    'Credit_History': train_df['Credit_History'].mode()[0]  # Fill missing 'Credit_History' values with the mode
}, inplace=True)

# Encode categorical variables
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']  # Define columns to be encoded
for col in categorical_cols:
    train_df[col] = LabelEncoder().fit_transform(train_df[col])  # Encode each categorical column as numerical

# Feature Engineering
train_df['Income_Loan_Ratio'] = (train_df['ApplicantIncome'] + train_df['CoapplicantIncome']) / train_df['LoanAmount']  # Create new feature as a ratio of total income to loan amount

# Split into features and target variable
X_train = train_df.drop(['Loan_ID', 'Loan_Status'], axis=1)  # Separate features (exclude 'Loan_ID' and 'Loan_Status')
y_train = train_df['Loan_Status'].apply(lambda x: 1 if x == 'Y' else 0)  # Convert target variable to binary (1 for 'Y', 0 for 'N')

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
)  # Standardize training set features
X_test[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Income_Loan_Ratio']] = scaler.transform(
    X_test[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Income_Loan_Ratio']]
)  # Apply the same scaling to the test set

# Rule-based approach
def rule_based_model(row):
    if row['Credit_History'] == 1 and row['Income_Loan_Ratio'] > 0.5:
        return 1  # Approve loan if good credit history and sufficient income-loan ratio
    else:
        return 0  # Decline otherwise

# Machine Learning Models
# Model 1: Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)  # Initialize Decision Tree with a random seed for reproducibility
dt_model.fit(X_train, y_train)  # Train the model on training data
y_pred_dt = dt_model.predict(X_test)  # Predict on test data

# Model 2: Logistic Regression with further increased max_iter and alternative solver
lr_model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')  # Logistic Regression with increased max iterations and alternative solver
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Model 3: Random Forest
rf_model = RandomForestClassifier(random_state=42)  # Initialize Random Forest
rf_model.fit(X_train, y_train)  # Train the model on training data
y_pred_rf = rf_model.predict(X_test)  # Predict on test data

# Compile predictions into a DataFrame for submission or further analysis
predictions_df = pd.DataFrame({
    'Loan_ID': test_df['Loan_ID'],  # Include 'Loan_ID' for identification in submission
    'DecisionTree_Prediction': y_pred_dt,  # Predictions from Decision Tree
    'LogisticRegression_Prediction': y_pred_lr,  # Predictions from Logistic Regression
    'RandomForest_Prediction': y_pred_rf  # Predictions from Random Forest
})

# Save predictions
predictions_df.to_csv('loan_predictions.csv', index=False)  # Save predictions to CSV file
print("Predictions saved to 'loan_predictions.csv'")
