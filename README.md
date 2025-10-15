cd ~/repo/faceTrack && printf '%s\n' "# faceTrack - Face Recognition Attendance System

A Python-based attendance system using YOLOv8 and DeepFace with a Tkinter GUI.

## Features
- Add people via images/videos/folders.
- Manage groups (add/remove members, groups).
- Predict attendance with annotated videos and Excel reports.
- Cross-platform: Linux, macOS, Windows.

## Prerequisites
- Python 3.13+
- Git
- \`curl\` or \`wget\` (for downloading model files)
- Internet connection (to download models and dependencies)

## Installation
1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/yuishigam1/faceTrack.git
   cd faceTrack
   \`\`\`
2. Run the following command to download models, install dependencies, and start the application:
   \`\`\`bash
   curl -o yolov8x-face.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt && curl -o shape_predictor_68_face_landmarks.dat.bz2 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && bunzip2 shape_predictor_68_face_landmarks.dat.bz2 && python -m venv face_recognition_env && . face_recognition_env/bin/activate 2>/dev/null || face_recognition_env\\Scripts\\activate && pip install -r requirements.txt && python gui.py
   \`\`\`
   - **Linux/macOS**: The command uses \`. face_recognition_env/bin/activate\` to activate the virtual environment.
   - **Windows**: It uses \`face_recognition_env\\Scripts\\activate\` (the \`2>/dev/null\` ensures compatibility by suppressing errors if one activation fails).
   - Downloads \`yolov8x-face.pt\` (from Ultralytics) and \`shape_predictor_68_face_landmarks.dat.bz2\` (from Dlib, then decompresses it).
   - Creates a virtual environment, installs dependencies from \`requirements.txt\`, and runs \`gui.py\`.

## Usage
- **Add Person**: Upload images/videos to register a person.
- **Groups**: Manage groups (add/remove members, create/delete groups).
- **Predict**: Run face recognition on videos/images, outputs annotated video and Excel report.

## Dependencies
See \`requirements.txt\` for full list, including:
- opencv-python
- pandas
- openpyxl
- numpy
- torch
- ultralytics
- deepface

## Notes
- Install \`dlib\` for better preprocessing: \`pip install dlib\` (requires CMake, libpng).
- Ensure \`yolov8x-face.pt\` and \`shape_predictor_68_face_landmarks.dat\` are in the project root before running (handled by the installation command).
- Check licensing for \`yolov8x-face.pt\` (Ultralytics) and \`shape_predictor_68_face_landmarks.dat\` (Dlib) before distribution." > README.md
