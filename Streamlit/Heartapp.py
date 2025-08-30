import numpy as np
import pickle
import streamlit as st
import pandas as pd


file = '/Users/mj_peace/Desktop/MyMLOPS/Streamlit/heart.csv'
df = pd.read_csv(file)  
print(df.nunique())

categorical_cols = []
for col in df.columns:
    if df[col].dtype == 'object' or df[col].nunique() < 10: # Heuristic: consider columns with less than 20 unique values as potentially categorical
        categorical_cols.append(col)

print("Categorical columns:", categorical_cols)

# for col in df.columns:
#     print(f"Value counts for {col}:")
#     print(df[col].value_counts())
#     print("\n") 

# exit()

st.title('Heart Disease Prediction Web App')
st.write('Enter patient measurements to predict the presence of heart disease.')
# Input fields for the features

age = st.number_input('Age', min_value=1, max_value=120, value=30)
sex = st.selectbox('Please select your gender', ("Male","Female"))
cp = st.slider('Chest Pain Type (0-3)', 0, 3, 1 )
trestbps = st.number_input('Resting Blood Pressure', min_value=50, max_value=250, value=120)
chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ("Yes","No"))
restecg = st.selectbox('Resting Electrocardiographic Results (0-2)', [0, 1, 2])
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
exang = st.selectbox('Exercise Induced Angina', [0, 1])
oldpeak = st.number_input('ST Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=10.0, value=1.0)
slope = st.number_input('Slope of the Peak Exercise ST Segment (0-2)', min_value=0, max_value=2, value=1)
ca = st.number_input('Number of Major Vessels Colored by Fluoroscopy (0-4)', min_value=0, max_value=4, value=0)
thal = st.number_input('Thalassemia (0-3)', min_value=0, max_value=3, value=2)  


sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0

features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

pikl_file ='/Users/mj_peace/Desktop/MyMLOPS/Streamlit/HeartStudy_RF.pkl'
with open(pikl_file, 'rb') as f:
    rf_model = pickle.load(f)   

prediction = rf_model.predict(features)

if prediction[0] == 1:
    st.error(f"⚠️ High Risk of Heart Disease!")
    st.write("Please consult with a healthcare professional.")
else:
    st.success(f"✅ Low Risk of Heart Disease!")    