#!/bin/bash
set -e
if [ ! -d "face_recognition_env" ]; then
    python3 -m venv face_recognition_env
fi
source face_recognition_env/bin/activate
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source face_recognition_env/Scripts/activate
fi
pip install -r requirements.txt
python3 gui.py
