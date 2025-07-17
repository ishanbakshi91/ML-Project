import streamlit as st
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open('model.pkl','rb'))
encoder = pickle.load(open('target_encoder.pkl','rb'))
transformer = pickle.load(open('transformer.pkl','rb'))

st.title("Insurance Premium Prediction")
age = st.text_input('Enter Age', 18)
age = int(age) 

sex = st.selectbox('Please select gender', ('Male', 'Female'))

bmi = st.text_input('Enter BMI', 18)
bmi = float(bmi)

children = st.selectbox('Please select number of children ', (0,1,2,3,4,5))
children = int(children)


smoker = st.selectbox('Please select smoker category ', ("Yes","No"))

region = st.selectbox('Please select your region ', ("southwest", "southeast", "northeast", "northwest"))


l = {}
l['age'] = age
l['sex'] = sex
l['bmi'] = bmi
l['children'] = children
l['smoker'] = smoker
l['region'] = region

df = pd.DataFrame(l, index=[0])

df['region'] = encoder.transform(df['region'])
df['sex'] = df['sex'].map({'Male':1, 'Female':0})
df['smoker'] = df['smoker'].map({'Yes':1, 'No':0})

df = transformer.transform(df)
y_pred = model.predict(df)

if st.button("Submit"):
    st.header(f" Premium Prediction is {round(y_pred[0],2)} INR")
