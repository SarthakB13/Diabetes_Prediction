import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from plotly import graph_objs as go
from sklearn.preprocessing import StandardScaler
import pickle
import os



data_file = 'E:\\ML Projects with web app\\Streamlit notes\\train.csv.csv'
data = pd.read_csv(data_file)

SX = data.drop(columns = 'Outcome', axis=1)
scaler = StandardScaler()
scaler.fit(SX)

filename = 'diabetes_pred_model'
Loaded_model = pickle.load(open(filename, 'rb'))

st.title("Diabetes predictor")
nav = st.sidebar.radio("Home", ["Data", "Prediction", "Contribute"])
if nav == "Data":
    if st.checkbox("Show The Dataset"):
        st.dataframe(data, width = 1200)  


if nav == "Prediction":
    c1 , c2, c3, c4 = st.columns(4)

    v1 = c1.number_input("No. of Pregnancies",0.00,20.00, step = 1.00)
    v2 = c2.text_input("Glucose level")
    v3 = c3.text_input("Blood Pressure")
    v4 = c4.text_input("Skin Thickness")
    
    c5 , c6, c7, c8 = st.columns(4)
    
    v5 = c5.text_input("Insulin level") 
    v6 = c6.text_input("BMI")
    v7 = c7.text_input("Diabetes Pedigree Function")
    v8 = c8.text_input("Age")
    if st.button("Predict"):
        input_data = (v1, v2, v3, v4, v5, v6, v7, v8)
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        std_data = scaler.transform(input_data_reshaped)
        prediction = Loaded_model.predict(std_data)
        if(prediction == 1):
            st.error('the patient is diabitic')
        else:
            st.success('the patient is not diabitic')
    
    
if nav == "Contribute":
    st.header("Contribute to our dataset")
    pe = st.number_input("Pregnancies",0.00,20.00, step = 1.00)
    gl = st.text_input("Glucose level")
    bp = st.text_input("Blood Pressure")
    ST = st.text_input("Skin Thickness")
    IL = st.text_input("Insulin level")
    bmi = st.text_input("BMI")
    dpf = st.text_input("Diabetes Pedigree Function")
    ag = st.text_input("Age")
    D = st.text_input("Diabetic(0 or 1)")
    if st.button("submit"):
        to_add = {"Pregnancies":[pe],"Glucose":[gl], "BloodPressure":[bp],"SkinThickness":[ST],"Insulin":[IL],"BMI":[bmi],"DiabetesPedigreeFunction":[dpf],"Age":[ag],"Outcome":[D]}
        to_add = pd.DataFrame(to_add)
        to_add.to_csv(data_file ,mode='a',header = False,index= False)
        st.success("Submitted")
        
        
        