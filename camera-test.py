import streamlit as st

picture = st.camera_input("Take a picture")

if picture:
    st.write("Image is not none")
    st.image(picture)

else:
    st.error("Image is none.")