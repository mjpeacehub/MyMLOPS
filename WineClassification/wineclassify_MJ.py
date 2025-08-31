import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression as LR
import xgboost as xgb
import mlflow
import dagshub
import pickle


dataset = '/Users/mj_peace/Desktop/MyMLOPS/WineClassification/winequality-red.csv'
df = pd.read_csv(dataset) 


# if df.duplicated().any():
#     print("Data contains duplicate values.")
# else:
#     print("Data does not contain duplicate values.")

if df.isnull().values.any():
    print("Data contains null values.")
else:
    print("Data does not contain null values.")

# df = df.dropna()
# df = df.drop_duplicates()

print("Data after removing duplicates:",df.shape)

X = df.drop('quality',axis=1) # predictor feature coloumns
y = df.quality
print(y)
# import numpy as np
# y = y - y.min()
# print(y)
# exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
print('Training Set :',len(X_train))
print('Test Set :',len(X_test))
print('Training labels :',len(y_train))
print('Test Labels :',len(y_test))


# from sklearn.impute import KNNImputer
# imputer = KNNImputer(n_neighbors=5)
# X_train = imputer.fit_transform(X_train)
# X_test = imputer.transform(X_test)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("X_train and X_test scaled, and scaler saved to scaler.pkl")

# from sklearn.impute import SimpleImputer
# #impute with mean all 0 readings
# imputer = SimpleImputer(missing_values = 0 , strategy ="mean") #fill = Imputer(missing_values = 0 , strategy ="mean", axis=0)
# X_train = imputer.fit_transform(X_train_scaled)
# X_test = imputer.transform(X_test_scaled)

from sklearn.model_selection import GridSearchCV

mlflow.set_experiment("WineClassification")

print("Starting model training..." )

# Grid Search for Random Forest
param_grid = {
    'n_estimators': [10, 35, 50, 100],
    'max_depth': [None, 5, 10, 20],
    'criterion': ['gini', 'entropy'],
    'random_state': [42]
}
grid_search = GridSearchCV(RF(), param_grid, cv=3, n_jobs=1, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters found by GridSearchCV:", grid_search.best_params_)


dagshub.init(repo_owner='edurekajuly24gcp', repo_name='skillfy_morn_2707', mlflow=True)

# import subprocess
# import time

# # Start the MLflow UI on port 7000
# subprocess.Popen(["mlflow", "ui", "--port", "7001"])

# # Optional: Give it time to start
# time.sleep(3)

# print("MLflow UI running at http://localhost:7001")
mlflow.set_experiment("WineClassification")

with mlflow.start_run(run_name="random_forest_gridsearch"):
    mlflow.set_tag("author", "MJPeace")
    mlflow.set_tag("model_type", "random_forest_gridsearch")
    model = grid_search.best_estimator_
    print(model)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(model)
    print(report)
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    print("dictionary keys are:", report_dict.keys())
    # mlflow.log_params(grid_search.best_params_)
    mlflow.log_metrics({
        'accuracy': report_dict['accuracy'],
        # 'recall_class_0': report_dict['recall'],
        # 'recall_class_1': report_dict['recall'],
        'f1_score_macro': report_dict['macro avg']['f1-score']
    })
    mlflow.log_artifacts("rfmodelheart.pkl")
      # Save and log model
    filename = f'{model}_model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    # mlflow.log_artifact(filename, model)
    run_id = mlflow.active_run().info.run_id
    mlflow.end_run()

print(run_id)
print("Training Complete")

