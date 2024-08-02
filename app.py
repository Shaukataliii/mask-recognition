import os
import streamlit as st
from camera_input_live import camera_input_live
from src import module

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
        image = session['picture']
        with st.spinner("Processing..."):
            is_mask_present = module.detect_save_predict_face_give_prediction(image)
            st.write(is_mask_present)

else:
    image = camera_input_live()
    st.image(image)
    # with st.spinner("Processing..."):
    is_mask_present = module.detect_save_predict_face_give_prediction(image)
    st.write(is_mask_present)