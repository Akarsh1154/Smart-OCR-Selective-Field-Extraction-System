import cv2 as cv
import numpy as np
import streamlit as st  
from PIL import Image

def rescale_image(image, scale_factor=2):
    # Get original dimensions
    height, width = image.shape[:2]

    # Calculate new dimensions
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    dimensions = (new_width, new_height)

    # FIXED: Use CUBIC for upscaling (zooming in) to keep text sharp
    rescaled_image = cv.resize(image, dimensions, interpolation=cv.INTER_CUBIC)
    return rescaled_image

def to_grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def remove_noise(image):
    return cv.medianBlur(image, 3)

def deskew_image(image):
    # FIXED: Find coordinates of BLACK pixels (text), not negative values
    # We invert the image so text becomes White (255) for detection
    inverted = cv.bitwise_not(image)
    coords = np.column_stack(np.where(inverted > 0))
    
    # Safety check: if image is blank, return as is
    if len(coords) == 0:
        return image

    angle = cv.minAreaRect(coords)[-1]

    # FIXED: Logic to handle OpenCV's angle range (-90 to 0)
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv.getRotationMatrix2D(center, angle, 1.0)
    
    # FIXED: Use borderValue=255 (White) to fill corners
    rotated = cv.warpAffine(
        image, M, (w, h), 
        flags=cv.INTER_CUBIC, 
        borderMode=cv.BORDER_CONSTANT, 
        borderValue=255
    )

    return rotated


def binarize_image(image):
    # Generates White Background (255) and Black Text (0)
    binary_image = cv.THRESH_BINARY + cv.THRESH_OTSU
    _, binary_image = cv.threshold(image, 0, 255, binary_image)
    return binary_image

def clean_image(image,display=False):
    image = np.array(image)
    # Pipeline: Rescale -> Gray -> Noise -> Binary -> Deskew
    resimg = rescale_image(image)
    if display:                                    
        st.image(resimg, caption="Rescaled Image")
    grayimg = to_grayscale(resimg)
    if display:                                    
        st.image(grayimg, caption="Grayscale Image")
    removenoise = remove_noise(grayimg)
    if display:                                    
        st.image(removenoise, caption="Denoised Image")
    deskewedimg = deskew_image(removenoise)
    if display:                                    
        st.image(deskewedimg, caption="Deskewed Image")
    binaryimg = binarize_image(deskewedimg)
    if display:                                   
        st.image(binaryimg, caption="Binarized Image")
    binaryimg = Image.fromarray(binaryimg) 
    return binaryimg