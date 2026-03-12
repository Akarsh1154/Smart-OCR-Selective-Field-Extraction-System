import streamlit as st
import cv2 as cv 
def get_user_input():
    input_image= st.file_uploader("Upload Image File")
    options=st.selectbox("Choose the output" ,options=['total','customer info','date'])
    return input_image,options
