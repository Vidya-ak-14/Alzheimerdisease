# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 17:26:07 2023

@author: tmsha
"""


import pandas as pd
import numpy as np
import pickle
import streamlit as st
loaded_model=pickle.load(open('VotingClassifier.pkl', 'rb'))
def alzheimer_prediction(input_data):
    # Convert input data to a pandas DataFrame
    input_data_as_dataframe = pd.DataFrame(input_data, columns=['M_F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF'])

    # Predict using the DataFrame
    prediction = loaded_model.predict(input_data_as_dataframe)

    # Print the prediction
    print(prediction)

    # Return the appropriate message based on the prediction
    if prediction[0] == 0:
       return 'The Person does not have Alzheimer Disease'
    else:
       return 'The person has Alzheimer Disease'
  
    
def main():
    
    
    # giving a title
    st.title('Alzheimer Prediction Web App')
    
    
    # getting the input data from the user
    
    
    M_F = st.text_input('Gender')
    Age=st.text_input('Age')
    EDUC= st.number_input('Years of Education')
    SES= st.number_input('Socioeconomic Status')
    MMSE= st.number_input('Mini Mental State Examination')
    eTIV= st.number_input('Estimated Total Intracranial Volume')
    nWBV=st.number_input('Normalize Whole Brain Volume')
    ASF= st.number_input('Atlas Scaling Factor')
    
    
   
              
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Alzheimer Test Result'):
        #diagnosis = alzheimer_prediction([['M_F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']])
        
        diagnosis = alzheimer_prediction([M_F, Age, EDUC, SES, MMSE, eTIV, nWBV, ASF])

    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()


