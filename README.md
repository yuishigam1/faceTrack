# faceTrack - Face Recognition Attendance System

A Python-based attendance system using YOLOv8 and DeepFace with a Tkinter GUI.

## Features
- Add people via images/videos/folders.
- Manage groups (add/remove members, groups).
- Predict attendance with annotated videos and Excel reports.
- Cross-platform: Linux, macOS, Windows.

## Prerequisites
- Python 3.13+
- Git
- `yolov8x-face.pt` (download from [Ultralytics](https://github.com/ultralytics/ultralytics))
- `shape_predictor_68_face_landmarks.dat` (download from [Dlib](http://dlib.net/files/))

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/faceTrack.git
   cd faceTrack
