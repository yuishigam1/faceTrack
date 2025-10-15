#!/usr/bin/env python3
"""
Face Recognition Prediction CLI Tool
Predicts faces in an image or video, outputs a processed file with unique name,
and returns frame counts for recognized individuals.
"""

import cv2
import os
import torch
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from ultralytics import YOLO
from deepface import DeepFace
from face_processing_pipeline import preprocess_face_complete, preprocess_face_basic

# Configure logging
logging.basicConfig(level=logging.INFO)

# Model initialization
face_detector = YOLO("yolov8x-face.pt")
MODELS = ["Facenet"]
MODEL_WEIGHTS = {"Facenet": 1.0}
MODEL_THRESHOLDS = {"Facenet": 0.55}
ENSEMBLE_MIN_CONFIDENCE = 0.65
MIN_MODELS_AGREEMENT = 1
MINIMUM_AVERAGE_SIMILARITY = 0.60
MINIMUM_TOP_SIMILARITY = 0.65

# ================ UTILITY FUNCTIONS ================
def get_unique_filename(base_path, filename_stem, file_extension):
    file_counter = 1
    output_path = base_path / f"{filename_stem}{file_extension}"
    while output_path.exists():
        output_path = base_path / f"{filename_stem}_{file_counter}{file_extension}"
        file_counter += 1
    return output_path

def load_face_database(embeddings_dir="face_embeddings", people=None):
    database = {}
    embeddings_dir = Path(embeddings_dir)
    if not embeddings_dir.exists():
        logging.warning("No face embeddings directory found")
        return database

    total_embeddings = 0
    for person_file in embeddings_dir.glob("*.pt"):
        person_name = person_file.stem.replace("_embeddings", "")
        if people is None or person_name in people:
            try:
                data = torch.load(person_file)
                embeddings = data["embeddings"].numpy()
                image_paths = data["image_paths"]
                database[person_name] = {"embeddings": embeddings, "image_paths": image_paths}
                total_embeddings += len(embeddings)
                logging.info(f"Loaded {len(embeddings)} embeddings for {person_name}")
            except Exception as e:
                logging.debug(f"Failed to load embeddings for {person_name}: {e}")
    logging.info(f"Database loaded: {len(database)} people, {total_embeddings} embeddings")
    return database

def calculate_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0
    cosine_sim = dot_product / (norm1 * norm2)
    return (1 + cosine_sim) / 2

def recognize_face_ensemble(face_image, confidence_threshold=0.65):
    if face_image is None or face_image.size == 0:
        return "Unknown", 0

    try:
        face_image = preprocess_face_complete(face_image)
    except Exception as e:
        logging.debug(f"Advanced preprocessing failed, using basic: {e}")
        try:
            face_image = preprocess_face_basic(face_image)
        except Exception as e2:
            logging.debug(f"Basic preprocessing failed, using resize only: {e2}")
            if face_image.shape[:2] != (224, 224):
                face_image = cv2.resize(face_image, (224, 224))

    model_results = {}
    for model_name in MODELS:
        try:
            embedding_result = DeepFace.represent(face_image, model_name=model_name, enforce_detection=False)
            if not embedding_result:
                continue
            face_embedding = np.array(embedding_result[0]["embedding"])

            best_person = "Unknown"
            best_similarity = 0
            model_threshold = MODEL_THRESHOLDS[model_name]

            for person_name, person_data in FACE_DATABASE.items():
                if "embeddings" not in person_data:
                    continue
                similarities = [calculate_similarity(face_embedding, emb) for emb in person_data["embeddings"]]
                if similarities:
                    top_similarities = sorted(similarities, reverse=True)[:3]
                    avg_similarity = np.mean(top_similarities)
                    if avg_similarity >= model_threshold and avg_similarity > best_similarity:
                        best_similarity = avg_similarity
                        best_person = person_name

            model_results[model_name] = {
                "person": best_person,
                "similarity": best_similarity,
                "weight": MODEL_WEIGHTS[model_name]
            }
        except Exception as e:
            logging.debug(f"Model {model_name} failed: {e}")

    if not model_results:
        return "Unknown", 0

    person_scores = {}
    valid_votes = sum(1 for result in model_results.values() if result["person"] != "Unknown")
    if valid_votes < MIN_MODELS_AGREEMENT:
        return "Unknown", 0.0

    for model_name, result in model_results.items():
        person = result["person"]
        similarity = result["similarity"]
        weight = result["weight"]
        if person != "Unknown":
            if person not in person_scores:
                person_scores[person] = {"weighted_score": 0, "votes": 0, "total_similarity": 0, "max_similarity": 0}
            person_scores[person]["weighted_score"] += similarity * weight
            person_scores[person]["votes"] += 1
            person_scores[person]["total_similarity"] += similarity
            person_scores[person]["max_similarity"] = max(person_scores[person]["max_similarity"], similarity)

    if not person_scores:
        return "Unknown", 0.0

    best_candidate = None
    best_score = 0
    for person, scores in person_scores.items():
        if scores["votes"] < MIN_MODELS_AGREEMENT:
            continue
        avg_similarity = scores["total_similarity"] / scores["votes"]
        if avg_similarity < MINIMUM_AVERAGE_SIMILARITY or scores["max_similarity"] < MINIMUM_TOP_SIMILARITY:
            continue
        if scores["weighted_score"] < ENSEMBLE_MIN_CONFIDENCE:
            continue
        if scores["weighted_score"] > best_score:
            best_score = scores["weighted_score"]
            best_candidate = person

    if best_candidate and best_score >= confidence_threshold:
        return best_candidate, best_score
    return "Unknown", 0.0

def draw_enhanced_bbox(frame, bbox, person_name, confidence):
    x1, y1, x2, y2 = map(int, bbox)
    color = (0, 255, 0) if person_name != "Unknown" else (0, 0, 255)
    thickness = 3 if confidence > 0.7 else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    text = f"{person_name} ({confidence:.2f})" if person_name != "Unknown" else f"Unknown ({confidence:.2f})"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_y = y1 - 10 if y1 > 30 else y2 + 30
    cv2.rectangle(frame, (x1, text_y - text_size[1] - 10), (x1 + text_size[0] + 10, text_y + 5), (0, 0, 0), -1)
    cv2.putText(frame, text, (x1 + 5, text_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if person_name != "Unknown":
        bar_width = int((x2 - x1) * confidence)
        cv2.rectangle(frame, (x1, y2 + 5), (x1 + bar_width, y2 + 10), color, -1)

# ================ PREDICTION FUNCTIONS ================
def predict_image(image_path, output_dir="output_videos", confidence_threshold=0.65, people=None):
    global FACE_DATABASE
    FACE_DATABASE = load_face_database(people=people)
    if not os.path.exists(image_path):
        logging.error(f"Image file not found: {image_path}")
        return {}, None

    frame = cv2.imread(image_path)
    if frame is None:
        logging.error(f"Could not read image: {image_path}")
        return {}, None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = get_unique_filename(output_dir, f"result_{Path(image_path).stem}_{timestamp}", ".jpg")

    person_counts = {"Unknown": 0}
    detections = face_detector(frame)
    if detections[0].boxes is not None:
        for detection in detections[0].boxes:
            bbox = detection.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, bbox)
            padding = 15
            x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
            x2, y2 = min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            person_name, confidence = recognize_face_ensemble(face_img, confidence_threshold)
            person_counts[person_name] = person_counts.get(person_name, 0) + 1
            draw_enhanced_bbox(frame, [x1, y1, x2, y2], person_name, confidence)

    cv2.imwrite(str(output_filename), frame)
    logging.info(f"Image processed: {image_path}")
    logging.info(f"Output saved: {output_filename}")
    logging.info("Person counts in image:")
    for person, count in person_counts.items():
        if count > 0:
            logging.info(f"  {person}: {count}")
    return person_counts, str(output_filename)

def predict_video(video_path, output_dir="output_videos", confidence_threshold=0.65, people=None):
    global FACE_DATABASE
    FACE_DATABASE = load_face_database(people=people)
    if not os.path.exists(video_path):
        logging.error(f"Video file not found: {video_path}")
        return {}, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video: {video_path}")
        return {}, None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = get_unique_filename(output_dir, f"result_{Path(video_path).stem}_{timestamp}", ".mp4")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_filename), fourcc, fps, (width, height))

    person_counts = {"Unknown": 0}
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            detections = face_detector(frame)
            if detections[0].boxes is not None:
                for detection in detections[0].boxes:
                    bbox = detection.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, bbox)
                    padding = 15
                    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
                    x2, y2 = min(width, x2 + padding), min(height, y2 + padding)
                    face_img = frame[y1:y2, x1:x2]
                    if face_img.size == 0:
                        continue

                    person_name, confidence = recognize_face_ensemble(face_img, confidence_threshold)
                    person_counts[person_name] = person_counts.get(person_name, 0) + 1
                    draw_enhanced_bbox(frame, [x1, y1, x2, y2], person_name, confidence)

            info_text = f"Frame: {frame_number} | Faces: {sum(c for p, c in person_counts.items())}"
            cv2.rectangle(frame, (10, 10), (400, 50), (0, 0, 0), -1)
            cv2.putText(frame, info_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            out.write(frame)

        except Exception as e:
            logging.debug(f"Error processing frame {frame_number}: {e}")

        frame_number += 1
        if frame_number % 100 == 0:
            progress = (frame_number / total_frames) * 100
            logging.info(f"Progress: {progress:.1f}%")

    cap.release()
    out.release()

    logging.info(f"Video processed: {video_path}")
    logging.info(f"Output saved: {output_filename}")
    logging.info("Person counts in video:")
    for person, count in person_counts.items():
        if count > 0:
            logging.info(f"  {person}: {count}")
    return person_counts, str(output_filename)

# ================ MAIN EXECUTION ================
def main():
    parser = argparse.ArgumentParser(description="Face Recognition Prediction CLI Tool")
    parser.add_argument("--input", required=True, help="Path to input image or video")
    parser.add_argument("--output-dir", default="output_videos", help="Output directory (default: output_videos)")
    parser.add_argument("--confidence", type=float, default=0.65, help="Confidence threshold (default: 0.65)")
    parser.add_argument("--people", nargs="*", help="List of people to detect (e.g., yash_1)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logging.error(f"Input file not found: {args.input}")
        return

    logging.info(f"Processing input: {args.input}")
    logging.info(f"Confidence threshold: {args.confidence}")
    logging.info(f"Available people: {args.people or 'All'}")

    ext = os.path.splitext(args.input)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
        person_counts, output_path = predict_image(args.input, args.output_dir, args.confidence, args.people)
    elif ext in (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"):
        person_counts, output_path = predict_video(args.input, args.output_dir, args.confidence, args.people)
    else:
        logging.error(f"Unsupported file type: {ext}")
        return

    if person_counts:
        logging.info(f"\nFinal attendance report:")
        for person, count in sorted(person_counts.items()):
            if count > 0 and person != "Unknown":
                logging.info(f"{person}: {count}")
        if person_counts.get("Unknown", 0) > 0:
            logging.info(f"Unknown: {person_counts['Unknown']}")
        logging.info(f"Output saved to: {output_path}")
    else:
        logging.error("Processing failed!")

if __name__ == "__main__":
    main()