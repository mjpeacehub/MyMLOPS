import numpy as np
import pickle
import streamlit as st
import pandas as pd


file = '/Users/mj_peace/Desktop/MyMLOPS/Streamlit/heart.csv'
df = pd.read_csv(file)  
df.describe()

exit()

st.title('Flower Classification Web App')
st.write('Enter flower measurements to predict the species.')

sepal_length = st.slider('Sepal Length (cm)', 0.0, 10.0, 5.0)
sepal_width = st.slider('Sepal Width (cm)', 0.0, 10.0, 3.5)
petal_length = st.slider('Petal Length (cm)', 0.0, 10.0, 1.5)
petal_width = st.slider('Petal Width (cm)', 0.0, 10.0, 0.2)


# Load the pre-trained model
with open('random_forest_model.pkl', 'rb') as f:  
    model = pickle.load(f)  


features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Make a prediction
prediction = model.predict(features)


species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
predicted_species = species_mapping[prediction[0]]

# Display the prediction
st.write(f'The predicted species is: **{predicted_species}**')

exit()

df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df

option = st.selectbox(
    'Which number do you like best?',
     df['first column'])

'You selected: ', option

import streamlit as st
import time

'Starting a long computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'