#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import streamlit as st
import pickle


# In[2]:


loaded_model = pickle.load(open("pred.sav", "rb"))


# In[3]:


st.title("Diabetes Predictor")
Pregnancies = st.number_input("Number of Pregnancies")
Glucose = st.number_input("Glucose level")
BloodPressure = st.number_input("Blood Pressure value")
SkinThickness = st.number_input("Skin Thickness value")
Insulin = st.number_input("Insulin level")
BMI = st.number_input("Body Mass Index")
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function value")
Age = st.number_input("Age(in years)")


# In[4]:


def diabetes_prediction(input_data):
    A = np.asarray(input_data)
    B = A.reshape(1, -1)
    prediction = loaded_model.predict(B)
    print("prediction")
    if(prediction[0]==0):
        return"You are Non-Diabetic"
    else:
        return"You are Diabetic"


# In[5]:


diagnosis = ""
if(st.button("Predict")):
    diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
st.success(diagnosis)
st.write("Accuracy Score = ",(75.97402597402598))

