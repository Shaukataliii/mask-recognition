import os, cv2
import streamlit as st
import numpy as np
from keras.models import load_model

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
cwd = os.getcwd()
model_filepath = f"{cwd}/src/resources/model.h5"
face_haarcascade_filepath = f"{cwd}/src/resources/haarcascade_frontalface_default.xml"


def detect_save_predict_face_give_prediction(st_image):
    cv_image = convert_st_image_to_cv_image(st_image)
    face_images = get_all_face_images(cv_image)
    results = []

    for i,face in enumerate(face_images):
        face_mask_bool = is_face_mask_present(face)
        results.append(face_mask_bool)
        # save_cv_image(i, face, face_mask_bool)
    return results

def convert_st_image_to_cv_image(st_image):
    image_bytes = st_image.getvalue()
    cv_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return cv_image

def get_all_face_images(cv_image):
    face_images = []
    faces_coors = detect_all_faces(cv_image)
    if faces_coors.size != 0:
        for x,y,w,h in faces_coors:
            face_image = cv_image[y:y+h, x:x+w]
            st.image(face_image, caption="Extracted face on which perdiction will be performed")
            face_images.append(face_image)
        return face_images
    else:
        stop_app_with_warning("No face detected")

def detect_all_faces(cv_image):
    detector = load_harcascade_classifier()
    return detector.detectMultiScale(image=cv_image, scaleFactor=1.1, minNeighbors=3)

def load_harcascade_classifier():
    haarcascade_detector = cv2.CascadeClassifier(face_haarcascade_filepath)
    if haarcascade_detector.empty():
        stop_app_with_warning(f"Haarcascade classifier is empty. Path used: {face_haarcascade_filepath}")
    else:
        return haarcascade_detector

def is_face_mask_present(captured_image):
    encoded_image = encode_cvimage(captured_image)
    global model
    prediction = model.predict(encoded_image)
    prediction = map_probability(prediction[0])
    return prediction

def encode_cvimage(cv_image):
    image = cv2.resize(cv_image, dsize=(224,224))
    image = np.array(image)
    image = np.reshape(image, (1,224,224,3))
    return image

def map_probability(value):
    return False if value < 0.5 else True

def stop_app_with_warning(warning):
    st.warning(warning)
    st.stop()

def load_mlmodel():
    if os.path.exists(model_filepath):
        return load_model(filepath=model_filepath)
    else:
        stop_app_with_warning(f"Model path doesn't exist: {model_filepath}")

model = load_mlmodel()

# def save_cv_image(image_num, cv_image, result):
#     cv2.imwrite(f"results/detected-face-{image_num}-{result}.jpg", cv_image)