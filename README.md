# faceTrack - Face Recognition Attendance System

A Python-based attendance system using YOLOv8 and DeepFace with a Tkinter GUI for seamless face recognition and attendance tracking.

## Features

- Register people using images, videos, or folders.
- Group management: create, edit, or delete groups and manage members.
- Attendance tracking: generate annotated videos and Excel reports from video/image inputs.
- Cross-platform: works on Linux, macOS, and Windows.

## Prerequisites

- Python 3.13 or higher
- Git
- curl or wget (for downloading model files)
- Internet connection (for downloading models and dependencies)
- Windows users: ensure bunzip2 is installed (e.g., via Git Bash) or manually decompress .bz2 files using tools like 7-Zip.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yuishigam1/faceTrack.git
cd faceTrack
```

2. Set up and run (single-line command does the downloading, environment setup, dependency install, and runs the GUI):

```bash
curl -o yolov8x-face.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-face.pt \
  && curl -o shape_predictor_68_face_landmarks.dat.bz2 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 \
  && bunzip2 shape_predictor_68_face_landmarks.dat.bz2 \
  && python -m venv face_recognition_env \
  && . face_recognition_env/bin/activate 2>/dev/null || face_recognition_env\Scripts\activate \
  && pip install -r requirements.txt \
  && python gui.py
```

### What this does

- Downloads `yolov8x-face.pt` (YOLOv8 face detection model).
- Downloads and decompresses `shape_predictor_68_face_landmarks.dat` (Dlib facial landmarks model).
- Creates a virtual environment named `face_recognition_env`.
- Activates the virtual environment (auto-detects OS).
- Installs dependencies from `requirements.txt`.
- Launches the GUI (`gui.py`).

### Windows note

If `bunzip2` is not available on Windows, download `shape_predictor_68_face_landmarks.dat.bz2` manually from [http://dlib.net/files/shape\_predictor\_68\_face\_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and extract it using 7-Zip. Place the resulting `shape_predictor_68_face_landmarks.dat` file in the project root.

## Usage

- Add Person: upload images or videos to register a person in the system.
- Manage Groups: create/delete groups or add/remove members.
- Predict Attendance: process videos/images to generate annotated videos and Excel attendance reports.

## Dependencies

Dependencies are listed in `requirements.txt`. Core packages include:

```
opencv-python
pandas
openpyxl
numpy
torch
ultralytics
deepface
```

Optional (recommended for improved preprocessing):

```
pip install dlib
```

(Installing `dlib` requires CMake and libpng on the system.)
