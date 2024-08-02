import os, cv2, logging
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
model_filepath = r"src\resources\model.h5"
face_haarcascade_filepath = r"src\resources\haarcascade_frontalface_default.xml"


def convert_st_image_to_cv_image(st_image):
    image_bytes = st_image.getvalue()
    cv_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    return cv_image

def detect_camera_till_interupt():
    camera = cv2.VideoCapture(0)
    while True:
        success, cv_image = camera.read()
        cv2.imshow("Camera", cv_image)
        detect_save_predict_face_give_prediction(cv_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

def detect_camera_once(camera_port = 0):
    cv_image = capture_one_from_camera(camera_port)
    return detect_save_predict_face_give_prediction(cv_image)

def detect_save_predict_face_give_prediction(cv_image):
    face_images = get_all_face_images(cv_image)

    for i,face in enumerate(face_images):
        save_cv_image(i, face)
        face_mask_bool = is_face_mask_present(face)
        print(face_mask_bool)
        return face_mask_bool

def save_cv_image(image_num, cv_image):
    cv2.imwrite(f"results/detected-face-{image_num}.jpg", cv_image)

def capture_one_from_camera(camera_port = 0):
    camera = cv2.VideoCapture(camera_port)
    result, cv_image = camera.read()
    return cv_image

def detect_using_path():
    image = cv2.imread("results/original.jpg", 0)
    face_images = get_all_face_images(image)
    for i,face in enumerate(face_images):
        # face_mask = module.is_face_mask_present(face)
        cv2.imwrite(f"results/detected-face-{i}.jpg", face)

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

def get_all_face_images(cv_image):
    face_images = []
    faces_coors = detect_all_faces(cv_image)
    for x,y,w,h in faces_coors:
        face_image = cv_image[y:y+h, x:x+w]
        face_images.append(face_image)

    return face_images

def detect_all_faces(cv_image):
    detector = load_harcascade_classifier()
    return detector.detectMultiScale(cv_image, 1.1, 3)

def load_model():
    from keras.models import load_model
    if os.path.exists(model_filepath):
        return load_model(model_filepath)
    else:
        raise Exception(f"Model path doesn't exist: {model_filepath}")
    
def detect_faces_draw_rectangle(cv_image):
    detected_faces = detect_all_faces(cv_image)
    for x,y,w,h in detected_faces:
        cv2.rectangle(cv_image, (x,y), (x+w, y+h), (0,255,0), 3)
    return cv_image

def load_harcascade_classifier():
    return cv2.CascadeClassifier(face_haarcascade_filepath)

model = load_model()