@echo off
if not exist face_recognition_env (
    python -m venv face_recognition_env
)
call face_recognition_env\Scripts\activate
pip install -r requirements.txt
python gui.py
