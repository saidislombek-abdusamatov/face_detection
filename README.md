# Face Detection with YOLOv8 and Export to ONNX, TFLite, TensorRT

This repository contains a real-time face detection and tracking Streamlit application powered by YOLOv8. The application allows users to upload video files, detect faces in the frames, and display tracked faces. Additionally, the face detection model has been exported to ONNX, TFLite, and TensorRT formats for efficient deployment and inference on various platforms.

## Features

- **Real-time Face Detection:** Utilizes YOLOv8 for real-time face detection in video frames.
- **Face Tracking:** Implements object tracking to maintain consistency across frames.
- **Export to Multiple Formats:** The face detection model has been exported to ONNX, TFLite, and TensorRT formats for diverse deployment options.
- **Interactive Streamlit App:** Provides a user-friendly interface for uploading videos, adjusting detection parameters, and viewing tracked faces.
- **Efficient Inference:** The exported models allow efficient inference on different platforms, ensuring optimal performance.

## Getting Started

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/saidislombek-abdusamatov/face_detection.git
   cd face_detection
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

   Open your browser and go to `http://localhost:8501` to access the Streamlit app.

## Exported Models

The face detection model has been exported to different formats for deployment flexibility:

- **ONNX:** `models/face_det.onnx`
- **TFLite:** `models/face_det.tflite`
- **TensorRT:** `models/face_det.engine`

## Usage

1. **Upload Video:** Click on the "Upload video" button to upload a video file in `.mp4` format.
2. **Adjust Parameters:** Use the number input fields to adjust the grid width and confidence threshold for face detection.
3. **Start Detection:** Click the "Detect faces!" button to start the face detection process. Detected faces will be displayed in real-time.
4. **Stop Detection:** Click the "Stop!" button to stop the face detection process.

## Export and Inference Models

If you want to export and inference the face detection model to different formats, you can use the following scripts:

- **Export to [ONNX](onnx.ipynb)**

- **Export to [TFLite](tflite.ipynb)**

- **Export to [TensorRT](tensorrt.ipynb)**
