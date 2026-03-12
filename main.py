from core.preprocessing import clean_image
from core.userinput import get_user_input
import cv2 as cv 
import numpy as np
import streamlit as st

st.title("Smart OCR - Selective Field Extraction System")
st.write("Upload an image to see the cleaned version ready for OCR processing.")
image,requires = get_user_input()
if image is not None:
    st.success("image is there")
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    cleaned_image = clean_image(cv.imdecode(file_bytes, cv.IMREAD_COLOR))
    st.image(cleaned_image, caption="Cleaned Image")
else:
    st.error("image ille")
