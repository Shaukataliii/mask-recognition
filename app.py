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
        st.write("Image in not none.")
        st.image(image)

        with st.spinner("Processing..."):
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



# import streamlit as st
# import numpy as np
# import cv2, os

# # Load Haar Cascade for face detection
# cascade_path = os.path.join(os.getcwd(), "src", "resources", "haarcascade_frontalface_default.xml")
# haarcascade_detector = cv2.CascadeClassifier(cascade_path)

# if haarcascade_detector.empty():
#     st.error("Error loading Haar Cascade classifier")

# # Capture image using streamlit camera input
# st.title("Face Detection App")
# st.subheader("Capture an image and detect faces")

# image_data = st.camera_input("Capture Image")

# if image_data:
#     # Convert the image to OpenCV format
#     image_bytes = image_data.getvalue()
#     cv_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

#     st.subheader("Image")
#     st.image(cv_image, channels="GRAY")

#     # Detect faces in the image
#     faces = haarcascade_detector.detectMultiScale(cv_image, scaleFactor=1.1, minNeighbors=3)

#     # Draw rectangles around detected faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(cv_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

#     # Display the result
#     st.subheader("Detected Faces")
#     st.image(cv_image, channels="GRAY")
