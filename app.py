import streamlit as st
import pickle
import numpy as np


def load_model():
  with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
  return data


data = load_model()


model = data['model']
country_encoder = data['country_encoder']
edlevel_encoder = data['edlevel_encoder']

def show_predict_page():
  st.title('Software Developer Salary Prediction')
  st.write('We need some information to predict the salary')
  
  countries = (
    'United States of America', 'Canada', 'United Kingdom of Great Britain and Northern Ireland', 'Germany', 'France',
    'Netherlands', 'Spain', 'Italy', 'Australia', 'Brazil', 'Russia',
    'Poland', 'Switzerland', 'Sweden', 'Ukraine', 'Belgium', 'Austria', 'India', 'Norway', 'Denmark'
  )
  
  education = (
    'Less than a Bachelors', 'Bachelors degree', 'Masters degree', 'Post grad'
  )
  
  country = st.selectbox('Country', countries)
  education = st.selectbox('Education Level', education)
  
  experience = st.slider('Years of Experience', 0, 50, 3)
  
  ok = st.button('Predict Salary')
  
  if ok:
    X = np.array([[country, education, experience]])
    X[:, 0] = country_encoder.transform(X[:, 0])
    X[:, 1] = edlevel_encoder.transform(X[:, 1])
    X = X.astype(float)
    salary = model.predict(X)
    st.subheader(f'The estimated salary is ${salary[0]:.2f}')
  
  

show_predict_page()