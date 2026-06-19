import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pickle

model=tf.keras.models.load_model('model.h5')
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


    ## Streamlit app


st.title("Customer Churn / Exit Prediction")

# User input

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)

age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')

credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')

tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)

has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

encoded_geo=onehot_encoder_geo.transform([[geography]])
col_geo=pd.DataFrame(encoded_geo.toarray(),columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data=pd.concat([input_data,col_geo],axis=1)   

input_scaled=scaler.transform(input_data)

prediction=model.predict(input_scaled)
pred_prob=prediction[0][0]

st.write(f"The probability output was {pred_prob}")
if pred_prob>=0.5:
    st.write('The customer will most likely Churn / exit')

else:
    st.write('The customer will not Churn / exit')