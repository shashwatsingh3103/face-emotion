
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import draw_text, draw_bounding_box, apply_offsets
from utils.preprocessor import preprocess_input
from PIL import Image
import cv2

# Streamlit app setup
st.set_page_config(page_title="Real-time Emotion Recognition", layout="wide")
st.title("Real-time Emotion Recognition")
st.write("This application recognizes emotions in real-time from a webcam feed or video file.")

# Parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# Hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# Loading models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

# Getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# Starting lists for calculating modes
emotion_window = []

# Streamlit placeholders for buttons and video display
allow_camera = st.checkbox("Allow Camera Access", value=True)

# Conditionally display the video uploader based on checkbox state
upload_video_placeholder = st.empty()
if not allow_camera:
    upload_video = upload_video_placeholder.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
else:
    upload_video_placeholder.empty()
    upload_video = None

start_button = st.sidebar.button("Start")
stop_button = st.sidebar.button("Stop")
frame_placeholder = st.empty()

# Function to start capturing from video source
def start_capturing(source):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        st.error("Error: Could not open video source.")
        return

    while cap.isOpened():
        ret, bgr_image = cap.read()

        if not ret:
            st.warning("No frames to capture.")
            break

        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
                                              minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for face_coordinates in faces:
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except Exception as e:
                st.warning(f"Error resizing face image: {e}")
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)

            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except Exception as e:
                st.warning(f"Error calculating mode: {e}")
                continue

            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int)
            color = color.tolist()

            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -45, 1, 1)

        # Convert to PIL Image and update the Streamlit image display
        frame_image = Image.fromarray(rgb_image)
        frame_placeholder.image(frame_image, caption="Emotion Recognition", use_column_width=True)

        # Check the stop button state
        if st.session_state.get("stop"):
            cap.release()
            st.write("Stopped capturing.")
            st.session_state["stop"] = False
            break

    cap.release()

# Handle button actions
if start_button:
    st.session_state["stop"] = False
    if allow_camera:
        # To handle webcam directly, use `0` as the source
        start_capturing(0)  # Start with webcam
    elif upload_video is not None:
        # Use the uploaded video file directly
        start_capturing(upload_video)  # Start with uploaded video file

if stop_button:
    st.session_state["stop"] = True
