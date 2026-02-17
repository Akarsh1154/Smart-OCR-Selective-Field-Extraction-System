from core.preprocessing import clean_image, rescale_image, to_grayscale, remove_noise, binarize_image, deskew_image
from core.userinput import get_user_input
import cv2 as cv 
import streamlit as st

st.title("Smart OCR - Selective Field Extraction System")
st.write("Upload an image to see the cleaned version ready for OCR processing.")

if __name__ == "__main__":
    image = get_user_input()
    
    cleaned_image = clean_image( cv.imread(image)) 
    st.image(cleaned_image, caption="Cleaned Image")


    