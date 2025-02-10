# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import time

# Load dataset
df = pd.read_csv(r"c:\Users\surya\Downloads\Crop and fertilizer dataset.csv")

# Separate features and target variable
X = df.drop('Crop', axis=1)  # Replace 'Crop' with your actual target column name
y = df['Crop']

# Encode categorical target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Handle categorical variables in features using OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), list(X.select_dtypes(include=['object']).columns))], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the number of training samples and testing samples
print(f"Number of training samples: {X_train.shape[0]}")
print(f"Number of testing samples: {X_test.shape[0]}")

# Define models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Train Random Forest model and measure training time
start_time = time.time()
rf_model.fit(X_train, y_train)
rf_training_time = time.time() - start_time
print(f"Random Forest training time: {rf_training_time:.2f} seconds")

# Train XGBoost model and measure training time
start_time = time.time()
xgb_model.fit(X_train, y_train)
xgb_training_time = time.time() - start_time
print(f"XGBoost training time: {xgb_training_time:.2f} seconds")

# Make predictions using both models
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

# Calculate and display individual model accuracies
rf_accuracy = accuracy_score(y_test, rf_pred)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f'Random Forest accuracy: {rf_accuracy:.4f}')
print(f'XGBoost accuracy: {xgb_accuracy:.4f}')

# Combine predictions using voting (simple average)
stacked_pred = (rf_pred + xgb_pred) / 2

# Convert to integer predictions
stacked_pred = np.round(stacked_pred).astype(int)

# Evaluate final predictions
stacked_accuracy = accuracy_score(y_test, stacked_pred)
print(f'Final stacked model accuracy: {stacked_accuracy:.4f}')
