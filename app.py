# Import necessary libraries
import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO

# Set the title of the web app
st.title("Face detection with YOLOv8 Object Tracking")

# Load YOLOv8 face detection model
model = YOLO('models/face_det.pt')

# Initialize button state variable
b = 0

# Create sidebar for user input
with st.sidebar:
    # Allow user to upload a video file
    file = st.file_uploader("Upload video", type=["mp4"])
    # Allow user to select grid width for displaying detected faces
    n = st.number_input("Select Grid Width", 4, 8)
    # Create buttons for starting and stopping face detection
    col_sid = st.columns(2)
    if col_sid[0].button('Detect faces!'):
        b = 1
    if col_sid[1].button('Stop!'):
        b = 0

# Create a placeholder to display video frames
stframe = st.empty()
# Create tabs for displaying video frames and detected faces
tab1, tab2 = st.tabs(["Video", "Faces"])

# Code for displaying video frames
with tab1:
    if file:
        # Create a temporary file to store the uploaded video
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        vf = cv2.VideoCapture(tfile.name)
        
        st.write('Last detected face')
        example = st.empty()
        # Create grid columns for displaying detected faces
        with tab2:
            cols = st.columns(n)

            if b:
                idn, x = 0, 0
                while vf.isOpened():
                    success, frame = vf.read()

                    if success:
                        frc = frame.copy()
                        # Detect and display faces in the video frames
                        for box in model.track(frame, persist=True, verbose=False)[0].boxes:
                            if box.conf < 0.5:
                                continue

                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            id = int(box.id) if box.id else 0
                            if id > idn:
                                idn = id
                                cols[x % n].image(cv2.resize(frc[y1:y2, x1:x2], (100, 100)), channels="BGR", caption=str(x + 1))
                                x += 1
                                example.image(cv2.resize(frc[y1:y2, x1:x2], (150, 150)), channels="BGR",
                                              caption=str(float(box.conf) * 100)[:4] + "%")

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(frame, f"id: {id}", (int(x1), int(y1) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (255, 255, 255), 2)

                        stframe.image(frame, channels="BGR", width=700)
                    else:
                        break
