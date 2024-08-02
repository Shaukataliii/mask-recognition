from src import module
import streamlit as st
from camera_input_live import camera_input_live


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