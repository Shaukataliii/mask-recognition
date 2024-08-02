# from src import module
import streamlit as st
# from camera_input_live import camera_input_live

try:
    import os
    st.write("os module imported successfully")
    
    import cv2
    st.write("cv2 module imported successfully")
    
    import numpy as np
    st.write("numpy module imported successfully")
    
    import pandas as pd
    st.write("pandas module imported successfully")
    
    import tensorflow as tf
    st.write("tensorflow module imported successfully")
    
    from tensorflow import keras
    st.write("keras module imported successfully")
    
    import matplotlib.pyplot as plt
    st.write("matplotlib module imported successfully")
    
except ImportError as e:
    st.write("ImportError: %s", e)
except Exception as e:
    st.write("Error: %s", e)


import streamlit as st
from src import module
from camera_input_live import camera_input_live

st.title("Mask Recognition App")

# Your application logic here

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# making prediction on the live input
st.title("Face Mask Detector")
features = ['Process single image', 'Process camera live']
session = st.session_state

selected_feat = st.selectbox("Select an option", features)
process_btn = st.button("Process Image", type="primary")

if selected_feat==features[0]:
    session['picture'] = st.camera_input("Take Picture")

    if process_btn and session['picture'] is not None:
        st.write("Image in not none.")
        st.image(session['picture'])

        with st.spinner("Processing..."):
            image = module.convert_st_image_to_cv_image(session['image'])
            is_mask_present = module.detect_save_predict_face_give_prediction(image)
            st.write(is_mask_present)

    else:
        st.error("Process btn is not pressed or picture is None.")
        st.write("Process btn", process_btn)
        st.write("Picture", session['picture'])

else:
    image = camera_input_live()
    st.write(type(image))
    st.image(image)