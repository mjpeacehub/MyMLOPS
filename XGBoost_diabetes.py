import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RandomForest
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle

data = '/Users/mj_peace/Desktop/ML/skillfy_morn_2707/Dagshub_demo/diabetes.csv'
df = pd.read_csv(data)
print(df.columns)
print("--------")

df.shape

df.head()
X = df.drop('Outcome',axis=1) # predictor feature coloumns
y = df.Outcome

X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.10, random_state = 42)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('Training Set :',len(X_train))
print('Test Set :',len(X_test))
print('Training labels :',len(y_train))
print('Test Labels :',len(y_test))

# KNN Imputer
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# model = XGBClassifier(
#     n_estimators=300,
#     max_depth=6,
#     learning_rate=0.05,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     use_label_encoder=False,
#     eval_metric='logloss'
# )


param_grid = {
    'n_estimators': [250, 300, 200],
    'max_depth': [10, 7, 9],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8]
}

xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
model = GridSearchCV(xgb, param_grid, scoring='f1', cv=3, verbose=1, n_jobs=-1)
model.fit(X_train, y_train)

print("Best parameters:", model.best_params_)
print("Best F1 score:", model.best_score_)

#model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

report = classification_report(y_test, y_pred)
print(report)

report_dict = classification_report(y_test, y_pred, output_dict=True)
print(report_dict)



# exit()



import dagshub
dagshub.init(repo_owner='edurekajuly24gcp', repo_name='skillfy_morn_2707', mlflow=True)

import mlflow
mlflow.set_experiment("XGBOOST 23_08")

with mlflow.start_run():
    mlflow.set_tag("author", "MJPeace")  # Replace with your actual name
    # Log all RandomForest parameters
    # for param, value in model.get_params().items():
    #     mlflow.log_param(param, value)
    mlflow.log_metrics({
        'accuracy': report_dict['accuracy'],
        'recall_class_0': report_dict['0']['recall'],
        'recall_class_1': report_dict['1']['recall'],
        'f1_score_macro': report_dict['macro avg']['f1-score']
    })
    # Save the model to a file
    filename = 'xbg_diabetes_gridsearch.pkl'
    pickle.dump(model, open(filename, 'wb'))
    # Log the model file as an artifact
    mlflow.log_artifact(filename, "xgboost-model")



    

