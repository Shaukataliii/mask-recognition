import streamlit as st
from queue import Queue
from streamlit_webrtc import webrtc_streamer
from src.module import Main, Utils


main = Main()
utils = Utils()
frames_queue = Queue()

def put_cvframe_to_queue(frame):
    cv_frame = frame.to_ndarray(format='bgr24')
    frames_queue.put(cv_frame)

with st.sidebar:
    st.title("Live Video Processor :movie_camera:")
    st.caption("Start capturing video and click process. To stop processing, stop capturing by clicking on the stop button.")
    process_frames = st.button("Process", type="primary")
    result_area = st.empty()

video_capturer = webrtc_streamer(
    key = "videoframes",
    video_frame_callback=put_cvframe_to_queue,
    media_stream_constraints={'video': True, 'audio': False}
)

if process_frames:
    frame_counter = 0
    while video_capturer.state.playing:
        try:
            frame = frames_queue.get()
            result = main.detect_faces_and_predict(frame)
            result = utils.format_result(result, frame_counter)
            result_area.text(result)
            
            frame_counter += 1

        except Exception as e:
            st.error(f"Failed to process frame. {e}")
            st.stop()