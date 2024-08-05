import streamlit as st
st.set_page_config("Face Mask Detector", page_icon=":movie_camera:")
from src.module import Main, Utils

session = st.session_state
utils = Utils()
main = Main()

with st.sidebar:
    st.title("Face Mask Detector :camera:")
    st.caption("Capture an image and click process.")
    process_btn = st.button("Process Image", type="primary")
    result_area = st.empty()

picture = st.camera_input("Capture an image for processing.")

if process_btn and picture is not None:
    # image = session['picture']
    with st.spinner("Processing..."):
        cv_image = utils.convert_st_image_to_cv(picture)
        is_mask_present = Main.detect_faces_and_predict(cv_image)
        result_area.text(is_mask_present)