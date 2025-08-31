import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load scaler and model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Find the model file (assuming only one *_model.pkl)
import glob
model_files = glob.glob('*_model.pkl')
if model_files:
    with open(model_files[0], 'rb') as f:
        model = pickle.load(f)
else:
    st.error('Model file not found!')
    st.stop()

st.title('Wine Quality Classification')
st.write('Predict wine quality using a trained Random Forest model.')

# Feature names from winequality-red.csv (excluding 'quality')
feature_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                 'pH', 'sulphates', 'alcohol']

def user_input_features():
    data = {}
    for feature in feature_names:
        data[feature] = st.number_input(f'{feature}', min_value=0.0, format='%f')
    return pd.DataFrame([data])

input_df = user_input_features()

if st.button('Predict Quality'):
    # Scale input
    input_scaled = scaler.transform(input_df)
    # Predict
    prediction = model.predict(input_scaled)
    result_df = input_df.copy()
    result_df['Predicted Quality'] = prediction
    st.success(f'Predicted wine quality: {prediction[0]}')
    st.write('Result DataFrame:')
    st.dataframe(result_df)

st.write('---')
st.write('Upload a CSV file for batch prediction:')
uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if set(feature_names).issubset(df.columns):
        X = df[feature_names]
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        df['Predicted Quality'] = preds
        st.write(df)
        st.download_button('Download Results', df.to_csv(index=False), file_name='wine_quality_predictions.csv')
    else:
        st.error('CSV must contain all required features: ' + ', '.join(feature_names))
