import os, cv2, time
import streamlit as st
import numpy as np
from keras.models import load_model

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class Main:
    def __init__(self):
        self.cwd = os.getcwd()
        self.model_filepath = f"{self.cwd}/src/resources/model.h5"
        self.face_haarcascade_filepath = f"{self.cwd}/src/resources/haarcascade_frontalface_default.xml"

    def detect_faces_and_predict(self, cv_image):
        face_images = self.get_all_face_images(cv_image)
        results = []

        for face in face_images:
            is_mask_present_bool = self.is_face_mask_present(face)
            results.append(is_mask_present_bool)
        return results

    def get_all_face_images(self, cv_image):
        face_images = []
        faces_coors = self.detect_all_faces(cv_image)
        if faces_coors.size != 0:
            for x,y,w,h in faces_coors:
                face_image = cv_image[y:y+h, x:x+w]
                face_images.append(face_image)
            return face_images
        else:
            self.stop_app_with_warning("No face detected")

    def detect_all_faces(self, cv_image):
        detector = self.load_harcascade_classifier()
        return detector.detectMultiScale(image=cv_image, scaleFactor=1.1, minNeighbors=4)   # handle if no face was detected

    def load_harcascade_classifier(self):
        haarcascade_detector = cv2.CascadeClassifier(self.face_haarcascade_filepath)
        if haarcascade_detector.empty():
            self.stop_app_with_warning(f"Haarcascade classifier is empty. Path used: {self.face_haarcascade_filepath}")
        else:
            return haarcascade_detector

    def is_face_mask_present(self, captured_image):
        global model
        encoded_image = self.encode_cvimage(captured_image)
        prediction = model.predict(encoded_image)
        prediction = self.map_probability(prediction[0])
        return prediction

    def encode_cvimage(self, cv_image):
        image = cv2.resize(cv_image, dsize=(224,224))
        image = np.array(image)
        image = np.reshape(image, (1,224,224,3))
        return image

    def map_probability(self, value):
        return False if value < 0.5 else True

    def stop_app_with_warning(self, warning):
        st.warning(warning)
        st.stop()


class Utils:
    def __init__(self):
        self.last_frame_processed_at = 0
    def convert_st_image_to_cv_image(self, st_image):
        image_bytes = st_image.getvalue()
        cv_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        return cv_image
    
    def calc_time_difference_from_last_frame(self):
        current_time = time.time()
        time_difference = current_time - self.last_frame_processed_at
        self.last_frame_processed_at = current_time
        return np.round(time_difference,1)
    
    def format_result(self, result, frame_counter):
        formatted_result = f"Frame no: {frame_counter} \
                            \nResult: {result} \
                            \nFrame processed after {utils.calc_time_difference_from_last_frame()} s."
        return formatted_result

@st.cache_resource
def load_mlmodel():
    if os.path.exists(main.model_filepath):
        return load_model(filepath=main.model_filepath)
    else:
        main.stop_app_with_warning(f"Model path doesn't exist: {main.model_filepath}")







utils = Utils()
main = Main()
model = load_mlmodel()