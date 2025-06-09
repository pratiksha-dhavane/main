import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle

# Load trained model
model = tf.keras.models.load_model("ann-regression/model.h5")

with open("ann-regression/label_encoder_gender.pkl","rb") as file:
  label_encoder_gn = pickle.load(file)

with open("ann-regression/onehot_encoder_geo.pkl","rb") as file:
  ohe_geo = pickle.load(file)

with open("ann-regression/scaler.pkl","rb") as file:
  scaler= pickle.load(file)

# Streamlit app

st.title("Estimated Salary Prediction")

geography = st.selectbox('Geography', ohe_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gn.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure',0, 10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is active member', [0,1])
exited = st.selectbox('Exited',[0,1])

input_data = pd.DataFrame({
  'CreditScore' : [credit_score],
  'Gender' : [label_encoder_gn.transform([gender])[0]],
  'Age' : [age],
  'Tenure' : [tenure],
  'Balance' : [balance],
  'NumOfProducts' : [num_of_products],
  'HasCrCard' : [has_cr_card],
  'IsActiveMember' : [is_active_member],
  'Exited' : [exited]
})

# One Hot enconding the Geography column
geo_encoded = ohe_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded.toarray(), columns=ohe_geo.get_feature_names_out(['Geography']))

# Combining one hot encoded columns with input data
input_df = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

# Using Scaling on the data
input_data_scaled = scaler.transform(input_df)

# Predict Estimated Salary
prediction = model.predict(input_data_scaled)
predict_proba = prediction[0][0]

st.write(f"Estimated Salary : {predict_proba:.2f}")

