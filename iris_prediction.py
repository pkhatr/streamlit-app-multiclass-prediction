# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np 
import pickle 
from tensorflow.keras import models
from PIL import Image


# Load KNN model
with open('knn.pkl', 'rb') as input_file:
    knn = pickle.load(input_file)

# Load Random Forest model
with open('rf.pkl', 'rb') as input_file:
    rf = pickle.load(input_file)

# Load ANN
ann = models.load_model('ann.h5')

# Load fitted Scaling object
with open('sc.pkl', 'rb') as input_file:
    sc = pickle.load(input_file)

# Load fitted label encoder object
with open('le.pkl', 'rb') as input_file:
    le = pickle.load(input_file)


st.title('Iris Flower Species Prediction')

st.sidebar.header('Input Parameters')
sepal_length = st.sidebar.number_input('Sepal Length', min_value=4.0, max_value=8.5, value=5.5)
sepal_width = st.sidebar.number_input('Sepal Width', min_value=1.0, max_value=5.0, value=3.0)
petal_length = st.sidebar.number_input('Petal Length', min_value=0.1, max_value=7.5, value=3.75)
petal_width = st.sidebar.number_input('Petal Width', min_value=0.1, max_value=3.5, value=1.25)

# Create array of input
x_values = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Update x_values using sc
x_values = sc.transform(x_values)

image_iris = Image.open('Iris_flower.jpg')
st.image(image_iris, caption='Downloaded from Pixabay https://pixabay.com/photos/iris-flowers-purple-red-purple-1669973/')

st.header('Application Information')
st.write('Iris is a beautiful flower and it symbolizes wisdom, courage, hope and faith. \
    This application helps in predicting the Iris species based on information about petal and sepal.')

st.header('Prediction - Output')

# Predictions using different models
y_knn = knn.predict(x_values)
prob_knn = knn.predict_proba(x_values)
y_rf = rf.predict(x_values)
prob_rf = rf.predict_proba(x_values)
y_ann = np.argmax(ann.predict(x_values), axis=1)
y_ann = le.inverse_transform(y_ann)
prob_ann = np.max(ann.predict(x_values), axis=1)


df = pd.DataFrame({'Algorithm': ['K Nearest Neighbor', 'Random Forest', 'Artificial Neural Network'], 
                    'Predicted Species': [y_knn[0], y_rf[0], y_ann[0]],
                    'Probability': [np.max(prob_knn[0]), np.max(prob_rf[0]), prob_ann[0]]
                    })

st.write(df)