import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LogRegression
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.metrics import classification_report
import xgboost as xgb
import pickle
import mlflow


data = '/Users/mj_peace/Desktop/MyMLOPS/Datasets/heart.csv'
df = pd.read_csv(data)
print(df.columns)

df.shape
df.head()
X = df.drop('target',axis=1) # predictor feature coloumns
y = df.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
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

from sklearn.impute import SimpleImputer
#impute with mean all 0 readings
imputer = SimpleImputer(missing_values = 0 , strategy ="mean") #fill = Imputer(missing_values = 0 , strategy ="mean", axis=0)
X_train = imputer.fit_transform(X_train_scaled)
X_test = imputer.transform(X_test_scaled)


models = {
    'logistic_regression': {
        'model': LogRegression(),
        'params': {
            "solver": "lbfgs",
            "max_iter": 45,
            "multi_class": "auto",
            "random_state": 123
        }
        },
    'random_forest': {
        'model': RandomForest(),
        'params': {
            "n_estimators": 35,
            # "max_depth": 10,
            "criterion": "gini",
            "random_state": 42
        }
        },
    'xgboost': {
        'model': xgb.XGBClassifier(),
        'params': {
            'n_estimators': 100,
            # 'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 123
        }
        }
            
}

# Train and log each model
mlflow.set_experiment("Multi_Classifier_Heart_Experiment")

for model_name, model_info in models.items():
    print(f"\nTraining {model_name}...")
    
    with mlflow.start_run(run_name=model_name):
        # Set tags
        mlflow.set_tag("author", "MJPeace")
        mlflow.set_tag("model_type", model_name)
        
        # Train model
        model = model_info['model']
        model.set_params(**model_info['params'])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)  

        report_dict = classification_report(y_test, y_pred,output_dict=True)
        print(report_dict)

        # report_dict = classification_report(y_test, y_pred, output_dict=True)
        # print(report_dict)
         
        # Log parameters
        mlflow.log_params(model_info['params'])
        
        # Log metrics
        metrics = {
            'accuracy': report_dict['accuracy'],
            'recall_class_0': report_dict['0']['recall'],
            'recall_class_1': report_dict['1']['recall'],
            'f1_score_macro': report_dict['macro avg']['f1-score']
        }
        mlflow.log_metrics(metrics)
        
        # Save and log model
        filename = f'{model_name}_model.pkl'
        pickle.dump(model, open(filename, 'wb'))
        mlflow.log_artifact(filename, model_name)
        
        print(f"{model_name} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score (macro): {metrics['f1_score_macro']:.4f}")


print("All models trained and logged.")        
