import streamlit as st
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('regression_model.h5')

# Load the scaler and encoders
with open('Reg_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('Reg_one_hot_encoding_geography.pkl', 'rb') as f:
    one_hot_encoding_geography = pickle.load(f)

with open('Reg_label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

# Define the Streamlit app
st.title("Customer Churn Prediction")

# User input
credit_score = st.number_input('CreditScore')
geography = st.selectbox('Geography', one_hot_encoding_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
tenure = st.slider('Tenure', 0, 10)
balance = st.number_input('Balance')
number_of_products = st.slider('NumOfProducts', 1, 4)
has_cr_card = st.selectbox('HasCrCard', [0, 1])
is_active_member = st.selectbox('IsActiveMember', [0, 1])
exited = st.selectbox('Exited', [0, 1])

# Prepare the input data for the model
input_data = {
    'CreditScore': credit_score,
    'Gender': label_encoder_gender.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': number_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'Exited': 0
}

geography_encoded = one_hot_encoding_geography.transform([[geography]]).toarray()
geography_encoded_df = pd.DataFrame(geography_encoded, columns=one_hot_encoding_geography.get_feature_names_out(['Geography']))

# combine one-hot encoded features with the original input data
input_data_df = pd.DataFrame([input_data])
input_data_df = pd.concat([input_data_df, geography_encoded_df], axis=1)

# scale the input data
input_data_scaled = scaler.transform(input_data_df)

# Make predictions
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

# display the prediction probability
st.write(f"Predicted salary: {prediction_probability:.2f}")

