import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.metrics import classification_report
import pickle

data = '/Users/mj_peace/Desktop/ML/skillfy_morn_2707/Dagshub_demo/diabetes.csv'
df = pd.read_csv(data)
print(df.columns)
df.shape

df.head()
X = df.drop('Outcome',axis=1) # predictor feature coloumns
y = df.Outcome

print(X.head())

X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.10, random_state = 42)



# KNN Imputer
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)



from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



model = RandomForest(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features="sqrt",
    class_weight="balanced",
    criterion="log_loss",
    random_state=10
)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)



report = classification_report(y_test, y_pred)
print(report)

report_dict = classification_report(y_test, y_pred, output_dict=True)
print(report_dict)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")  
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import dagshub
dagshub.init(repo_owner='edurekajuly24gcp', repo_name='skillfy_morn_2707', mlflow=True)


import mlflow

mlflow.set_experiment("RF experiments 23_08")

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
    filename = 'random_forest_model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    # Log the model file as an artifact
    mlflow.log_artifact(filename, "random-forest-model")