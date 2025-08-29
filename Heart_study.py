import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.metrics import classification_report
import pickle

data = '/Users/mj_peace/Desktop/MyMLOPS/Datasets/heart.csv'
df = pd.read_csv(data)
print(df.columns)

if df.isnull().values.any():
    print("Data contains null values.")
else:
    print("Data does not contain null values.") 

X = df.drop('target',axis=1) # feature columns  
y = df['target']


X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.3, random_state = 100,shuffle=True)

print('Training Set :',len(X_train))
print('Test Set :',len(X_test))
print('Training labels :',len(y_train))
print('Test Labels :',len(y_test))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)    
print('Scaled Training Set :',len(X_train_scaled))
print('Scaled Test Set :',len(X_test_scaled))

from sklearn.impute import SimpleImputer
#impute with mean all 0 readings
imputer = SimpleImputer(missing_values = 0 , strategy ="mean") #fill = Imputer(missing_values = 0 , strategy ="mean", axis=0)

X_train = imputer.fit_transform(X_train_scaled)
X_test = imputer.transform(X_test_scaled)   
print('Imputed Training Set :',len(X_train))
print('Imputed Test Set :',len(X_test))

params = {
    "n_estimators":35,
    "criterion": "gini",
    "random_state": 42}
model = RandomForest(**params)

model.fit(X_train, y_train) 
y_pred = model.predict(X_test)

report_dict = classification_report(y_test, y_pred,output_dict=True)

print(report_dict)

filename = 'HeartStudy_RF.pkl'
pickle.dump(model, open(filename, 'wb'))
print("Model dumped!")  

# report_dict_test = classification_report(y_test, y_test)

# print(report_dict_test)

# exit()
# ---------------------------AUC/ROC curve-------------------------------------
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc

# y_pred=model.predict(X_test)
# fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# roc_auc = auc(fpr, tpr)
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC)')
# plt.legend(loc="lower right")
# plt.show()

exit()

import dagshub
dagshub.init(repo_owner='edurekajuly24gcp', repo_name='skillfy_morn_2707', mlflow=True)

import mlflow   

mlflow.set_experiment("RF experiments 24_08")

with mlflow.start_run():
    mlflow.set_tag("author","MJ_Peace   ")
    mlflow.log_params(params)
    mlflow.log_metrics({
        'accuracy': report_dict['accuracy'],
        'recall_class_0': report_dict['0']['recall'],
        'recall_class_1': report_dict['1']['recall'],
        'f1_score_macro': report_dict['macro avg']['f1-score']
    })
    mlflow.log_artifacts("rfmodelheart.pkl")

