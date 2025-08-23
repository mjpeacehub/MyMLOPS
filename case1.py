import pandas as pd 

# read the csv file

filepath = '/Users/mj_peace/Desktop/MLOPS/Datasets/diabetes.csv'
df = pd.read_csv(filepath)
#print(df.head())


#preprocessing the data
df = df.dropna()
df = df.drop_duplicates()
# check for null values
if df.isnull().values.any():
    print("Data contains null values.")
else:
    print("Data does not contain null values.")

# EDA : Exploratory data analysis
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
# Visualizing the distribution of the target variable
# plt.figure(figsize=(10, 6))
# sns.countplot(x='Outcome', data=df) 
# plt.title('Distribution of Diabetes Outcome')
# plt.xlabel('Diabetes Outcome (0 = No, 1 = Yes)')
# plt.ylabel('Count')
# plt.show()

# Visualizing the correlation between features
# plt.figure(figsize=(12, 8))
# correlation_matrix = df.corr()
# sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
# plt.title('Correlation Matrix of Features')
# plt.show()  

# Splitting the data into features and target variable
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Building a machine learning model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluating the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")  
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Saving the model using joblib
import joblib
filepath = '/Users/mj_peace/Desktop/MLOPS/models/diabetes.pkl'
model_filename = filepath  
# model_filename = r"C:\Users\ankit_aj\Desktop\MLOPS-case_studies\Demo_050725_DVC\SKILLFY_21_JUNE_25\MLFLOW_demo\diabetes_model.pkl"  # Ensure the path is correct and does not have extra quotes'
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")

import mlflow
import mlflow.sklearn
# Set the tracking URI to the local file system
mlflow.set_tracking_uri("file:./mlruns")
# Set the experiment name
mlflow.set_experiment("1708_rf_diabetes_prediction_experiment")
# Start an MLflow run
with mlflow.start_run():
    # Log the model
    mlflow.sklearn.log_model(model, "rf_diabetes_model")
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    # Log the model file
    mlflow.log_artifact(model_filename, artifact_path="model_files")
# Print the run ID
    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")


import subprocess
import time

# Start the MLflow UI on port 5000
subprocess.Popen(["mlflow", "ui", "--port", "5000"])

# Optional: Give it time to start
time.sleep(3)

print("MLflow UI running at http://localhost:5000")

