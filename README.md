# faceTrack - Face Recognition Attendance System

A Python-based attendance system using YOLOv8 and DeepFace with a Tkinter GUI for seamless face recognition and attendance tracking.

## Features
- **Register People**: Add individuals using images, videos, or folders.
- **Group Management**: Create, edit, or delete groups and manage members.
- **Attendance Tracking**: Generate annotated videos and Excel reports from video/image inputs.
- **Cross-Platform**: Works on Linux, macOS, and Windows.

## Prerequisites
- Python 3.13 or higher
- Git
- `curl` or `wget` (for downloading model files)
- Internet connection (for downloading models and dependencies)
- Windows users: Ensure `bunzip2` is installed (e.g., via Git Bash) or manually decompress `.bz2` files using tools like 7-Zip.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yuishigam1/faceTrack.git
   cd faceTrack
   ```

2. **Set Up and Run**:
   Run the following command to download required models, install dependencies, and start the application:
   ```bash
   curl -o yolov8x-face.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-face.pt && curl -o shape_predictor_68_face_landmarks.dat.bz2 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && bunzip2 shape_predictor_68_face_landmarks.dat.bz2 && python -m venv face_recognition_env && . face_recognition_env/bin/activate 2>/dev/null || face_recognition_env\Scripts\activate && pip install -r requirements.txt && python gui.py
   ```
   - **What it does**:
     - Downloads `yolov8x-face.pt` (Ultralytics YOLOv8 face detection model).
     - Downloads and decompresses `shape_predictor_68_face_landmarks.dat` (Dlib facial landmarks model).
     - Creates a virtual environment (`face_recognition_env`).
     - Activates the virtual environment (Linux/macOS: `face_recognition_env/bin/activate`; Windows: `face_recognition_env\Scripts\activate`).
     - Installs dependencies from `requirements.txt`.
     - Launches the application (`gui.py`).
   - **Note for Windows users**: If `bunzip2` is unavailable, download `shape_predictor_68_face_landmarks.dat.bz2` manually from [Dlib](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and decompress it using 7-Zip or similar, then place `shape_predictor_68_face_landmarks.dat` in the project root.

## Usage
- **Add Person**: Upload images or videos to register a person in the system.
- **Manage Groups**: Create/delete groups or add/remove members.
- **Predict Attendance**: Process videos/images to generate annotated videos and Excel attendance reports.

## Dependencies
Listed in `requirements.txt`, including:
- `opencv-python`
- `pandas`
- `openpyxl`
- `numpy`
- `torch`
- `ultralytics`
- `deepface`

## Notes
- For improved preprocessing, install `dlib`: `pip install dlib` (requires CMake and libpng).
- Ensure `yolov8x-face.pt` and `shape_predictor_68_face_landmarks.dat` are in the project root before running (handled by the setup command).
- Verify licensing for `yolov8x-face.pt` (Ultralytics) and `shape_predictor_68_face_landmarks.dat` (Dlib) before distribution.