#!/usr/bin/env python3
"""
Unified Face Processing Pipeline
Extract faces from videos/images/folders, preprocess, augment, and generate/update embeddings incrementally.
"""

import cv2
import os
import torch
import argparse
import random
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from deepface import DeepFace

# Configure logging with file creation tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dlib is optional
try:
    import dlib
    detector = dlib.get_frontal_face_detector()
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if Path(predictor_path).exists():
        predictor = dlib.shape_predictor(predictor_path)
        LANDMARKS_AVAILABLE = True
    else:
        predictor = None
        LANDMARKS_AVAILABLE = False
        logger.info("Landmark predictor not found. Using basic alignment.")
except Exception as e:
    detector = None
    predictor = None
    LANDMARKS_AVAILABLE = False
    logger.info(f"Dlib not available: {e}. Using basic preprocessing.")

# Standard face size
STANDARD_FACE_SIZE = (224, 224)

# Model initialization
face_detector = YOLO("yolov8x-face.pt")

# ================ PREPROCESSING FUNCTIONS ================
def enhance_contrast_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def normalize_lighting_simple(image):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def reduce_blur(image):
    gaussian = cv2.GaussianBlur(image, (9, 9), 10.0)
    return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

def get_face_landmarks(image, face_rect):
    if not LANDMARKS_AVAILABLE or predictor is None:
        return None
    try:
        dlib_rect = dlib.rectangle(face_rect[0], face_rect[1], face_rect[0] + face_rect[2], face_rect[1] + face_rect[3])
        landmarks = predictor(image, dlib_rect)
        return np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
    except Exception as e:
        logger.debug(f"Failed to get landmarks: {e}")
        return None

def align_face_advanced(image, landmarks):
    if landmarks is None or len(landmarks) < 68:
        return image
    try:
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2, (left_eye_center[1] + right_eye_center[1]) // 2)
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1)
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    except Exception as e:
        logger.debug(f"Advanced alignment failed: {e}")
        return image

def align_face_simple(image):
    try:
        h, w = image.shape[:2]
        crop_size = min(h, w)
        y_start = (h - crop_size) // 2
        x_start = (w - crop_size) // 2
        return image[y_start : y_start + crop_size, x_start : x_start + crop_size]
    except Exception as e:
        logger.debug(f"Basic alignment failed: {e}")
        return image

def resize_face_consistent(image, target_size=STANDARD_FACE_SIZE):
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset, x_offset = (target_h - new_h) // 2, (target_w - new_w) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas

def preprocess_face_complete(face_image, enable_alignment=True, enable_lighting=True, enable_deblur=True, enable_resize=True):
    if face_image is None or face_image.size == 0:
        return face_image
    processed = face_image.copy()
    try:
        if enable_alignment:
            processed = align_face_simple(processed)
        if enable_deblur:
            processed = reduce_blur(processed)
        if enable_lighting:
            try:
                processed = enhance_contrast_clahe(processed)
            except Exception as e:
                logger.debug(f"CLAHE failed, using simple normalization: {e}")
                processed = normalize_lighting_simple(processed)
        if enable_resize:
            processed = resize_face_consistent(processed)
        return processed
    except Exception as e:
        logger.debug(f"Preprocessing failed: {e}")
        return resize_face_consistent(face_image) if enable_resize else face_image

def preprocess_face_basic(face_image):
    return preprocess_face_complete(face_image)

# ================ AUGMENTATION AND QUALITY ================
def augment_face(face_image, augmentation_type):
    augmented = face_image.copy()
    if augmentation_type == "brightness_up":
        beta = random.randint(15, 40)
        augmented = cv2.convertScaleAbs(augmented, alpha=1.0, beta=beta)
    elif augmentation_type == "brightness_down":
        beta = random.randint(-40, -15)
        augmented = cv2.convertScaleAbs(augmented, alpha=1.0, beta=beta)
    elif augmentation_type == "contrast_up":
        alpha = random.uniform(1.2, 1.5)
        augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=0)
    elif augmentation_type == "contrast_down":
        alpha = random.uniform(0.6, 0.8)
        augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=0)
    elif augmentation_type == "flip_horizontal":
        augmented = cv2.flip(augmented, 1)
    elif augmentation_type == "gaussian_blur":
        kernel_size = random.choice([3, 5])
        augmented = cv2.GaussianBlur(augmented, (kernel_size, kernel_size), 0)
    elif augmentation_type == "noise":
        noise = np.random.normal(0, 15, augmented.shape).astype(np.uint8)
        augmented = cv2.add(augmented, noise)
    elif augmentation_type == "rotation":
        angle = random.uniform(-10, 10)
        height, width = augmented.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented = cv2.warpAffine(augmented, rotation_matrix, (width, height))
    elif augmentation_type == "lighting_variation":
        height, width = augmented.shape[:2]
        if random.choice([True, False]):
            gradient = np.linspace(1.3, 0.8, width).reshape(1, -1)
        else:
            gradient = np.linspace(0.8, 1.3, width).reshape(1, -1)
        gradient = np.repeat(gradient, height, axis=0)
        if len(augmented.shape) == 3:
            gradient = np.repeat(gradient[:, :, np.newaxis], 3, axis=2)
        augmented = np.clip(augmented * gradient, 0, 255).astype(np.uint8)
    return augmented

def calculate_face_quality(face_crop):
    if face_crop.size == 0:
        return 0
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(laplacian_var * 2, 100)
    brightness = np.mean(gray)
    brightness_score = 100 - abs(brightness - 128) * 0.5
    height, width = face_crop.shape[:2]
    size_score = min((width * height) / (80 * 80) * 100, 100)
    quality = sharpness_score * 0.4 + brightness_score * 0.3 + size_score * 0.3
    return max(0, min(100, quality))

def estimate_pose_type(face_bbox, frame_shape):
    x1, y1, x2, y2 = face_bbox
    face_width = x2 - x1
    face_height = y2 - y1
    face_center_x = (x1 + x2) / 2
    face_center_y = (y1 + y2) / 2
    frame_height, frame_width = frame_shape[:2]
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2
    rel_x = (face_center_x - frame_center_x) / frame_center_x
    rel_y = (face_center_y - frame_center_y) / frame_center_y
    aspect_ratio = face_width / face_height
    if abs(rel_x) < 0.2 and abs(rel_y) < 0.2 and 0.9 <= aspect_ratio <= 1.2:
        return "frontal"
    elif abs(rel_x) > 0.3:
        return "profile"
    elif rel_y < -0.3:
        return "looking_up"
    elif rel_y > 0.3:
        return "looking_down"
    else:
        return "angled"

def should_sample_frame(frame_number, fps, duration_seconds, pose_type="frontal"):
    if fps <= 0:
        return frame_number % 5 == 0
    if pose_type in ["profile", "looking_up", "looking_down"]:
        target_interval = 0.4
    elif pose_type == "angled":
        target_interval = 0.3
    else:
        target_interval = 0.5
    frame_interval = max(1, int(fps * target_interval))
    return frame_number % frame_interval == 0

# ================ PROCESSING FUNCTIONS ================
def process_single_image(image_path, person_name, quality_threshold=35):
    face_images_dir = Path("face_images") / person_name
    face_images_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Creating face images directory: {face_images_dir}")

    frame = cv2.imread(image_path)
    if frame is None:
        logger.error(f"Cannot open image {image_path}")
        return 0, 0, 0, {}

    video_filename = Path(image_path).stem
    faces_saved = 0
    faces_rejected = 0
    augmented_faces = 0
    pose_counts = {"frontal": 0, "profile": 0, "looking_up": 0, "looking_down": 0, "angled": 0}

    augmentation_types = [
        "brightness_up", "brightness_down", "contrast_up", "contrast_down", "flip_horizontal",
        "gaussian_blur", "noise", "rotation", "lighting_variation"
    ]

    detection_results = face_detector(frame)
    if detection_results[0].boxes is not None:
        for face_idx, detection in enumerate(detection_results[0].boxes):
            face_box = detection.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = face_box
            pose_type = estimate_pose_type(face_box, frame.shape)

            height, width = frame.shape[:2]
            padding = int((x2 - x1) * 0.15)
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            quality_score = calculate_face_quality(face_crop)
            if quality_score < quality_threshold:
                faces_rejected += 1
                continue

            try:
                face_processed = preprocess_face_complete(face_crop)
            except Exception as e:
                logger.debug(f"Advanced preprocessing failed, using basic: {e}")
                try:
                    face_processed = preprocess_face_basic(face_crop)
                except Exception as e2:
                    logger.debug(f"Basic preprocessing failed, using resize only: {e2}")
                    face_processed = cv2.resize(face_crop, (224, 224))

            face_resized = face_processed
            pose_counts[pose_type] += 1

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            base_filename = f"{video_filename}_q{quality_score:.0f}_{pose_type}"
            original_filename = f"{base_filename}_original.jpg"
            original_filepath = face_images_dir / original_filename
            logger.info(f"Saving original face: {original_filepath}")
            cv2.imwrite(str(original_filepath), face_resized)
            faces_saved += 1

            if quality_score > 50:
                num_augmentations = random.randint(3, 5)
            else:
                num_augmentations = random.randint(2, 3)

            selected_augmentations = random.sample(augmentation_types, min(num_augmentations, len(augmentation_types)))
            for aug_type in selected_augmentations:
                try:
                    augmented_face = augment_face(face_resized, aug_type)
                    aug_filename = f"{base_filename}_{aug_type}.jpg"
                    aug_filepath = face_images_dir / aug_filename
                    logger.info(f"Saving augmented face: {aug_filepath}")
                    cv2.imwrite(str(aug_filepath), augmented_face)
                    augmented_faces += 1
                except Exception as e:
                    logger.debug(f"Augmentation {aug_type} failed: {e}")

    total_faces = faces_saved + augmented_faces
    logger.info(f"Image processing complete: {image_path}")
    logger.info(f"Original faces: {faces_saved}, Augmented: {augmented_faces}, Rejected: {faces_rejected}")
    logger.info("Pose distribution:")
    for pose, count in pose_counts.items():
        if count > 0:
            logger.info(f"  {pose}: {count}")
    return faces_saved, augmented_faces, faces_rejected, pose_counts

def process_video(video_path, person_name, quality_threshold=35):
    face_images_dir = Path("face_images") / person_name
    face_images_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Creating face images directory: {face_images_dir}")

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        logger.error(f"Could not open video file {video_path}")
        return 0, 0, 0, {}

    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps if fps > 0 else 0

    logger.info(f"Processing video: {total_frames} frames, {fps} FPS, {duration:.1f}s")

    frame_number = 0
    video_filename = Path(video_path).stem
    faces_saved = 0
    faces_rejected = 0
    augmented_faces = 0
    pose_counts = {"frontal": 0, "profile": 0, "looking_up": 0, "looking_down": 0, "angled": 0}

    augmentation_types = [
        "brightness_up", "brightness_down", "contrast_up", "contrast_down", "flip_horizontal",
        "gaussian_blur", "noise", "rotation", "lighting_variation"
    ]

    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break

        try:
            detection_results = face_detector(frame)
            if detection_results[0].boxes is not None:
                for face_idx, detection in enumerate(detection_results[0].boxes):
                    face_box = detection.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = face_box
                    pose_type = estimate_pose_type(face_box, frame.shape)

                    if not should_sample_frame(frame_number, fps, duration, pose_type):
                        continue

                    height, width = frame.shape[:2]
                    padding = int((x2 - x1) * 0.15)
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(width, x2 + padding)
                    y2 = min(height, y2 + padding)

                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue

                    quality_score = calculate_face_quality(face_crop)
                    if quality_score < quality_threshold:
                        faces_rejected += 1
                        continue

                    try:
                        face_processed = preprocess_face_complete(face_crop)
                    except Exception as e:
                        logger.debug(f"Advanced preprocessing failed, using basic: {e}")
                        try:
                            face_processed = preprocess_face_basic(face_crop)
                        except Exception as e2:
                            logger.debug(f"Basic preprocessing failed, using resize only: {e2}")
                            face_processed = cv2.resize(face_crop, (224, 224))

                    face_resized = face_processed
                    pose_counts[pose_type] += 1

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    base_filename = f"{video_filename}_f{frame_number:06d}_q{quality_score:.0f}_{pose_type}"
                    original_filename = f"{base_filename}_original.jpg"
                    original_filepath = face_images_dir / original_filename
                    logger.info(f"Saving original face: {original_filepath}")
                    cv2.imwrite(str(original_filepath), face_resized)
                    faces_saved += 1

                    if quality_score > 50:
                        num_augmentations = random.randint(3, 5)
                    else:
                        num_augmentations = random.randint(2, 3)

                    selected_augmentations = random.sample(augmentation_types, min(num_augmentations, len(augmentation_types)))
                    for aug_type in selected_augmentations:
                        try:
                            augmented_face = augment_face(face_resized, aug_type)
                            aug_filename = f"{base_filename}_{aug_type}.jpg"
                            aug_filepath = face_images_dir / aug_filename
                            logger.info(f"Saving augmented face: {aug_filepath}")
                            cv2.imwrite(str(aug_filepath), augmented_face)
                            augmented_faces += 1
                        except Exception as e:
                            logger.debug(f"Augmentation {aug_type} failed: {e}")

        except Exception as e:
            logger.debug(f"Error processing frame {frame_number}: {e}")

        frame_number += 1
        if frame_number % 100 == 0:
            progress = (frame_number / total_frames) * 100
            logger.info(f"Progress: {progress:.1f}%")

    video_capture.release()
    total_faces = faces_saved + augmented_faces
    logger.info(f"Video processing complete: {video_path}")
    logger.info(f"Original faces: {faces_saved}, Augmented: {augmented_faces}, Rejected: {faces_rejected}")
    logger.info("Pose distribution:")
    for pose, count in pose_counts.items():
        if count > 0:
            logger.info(f"  {pose}: {count}")
    return faces_saved, augmented_faces, faces_rejected, pose_counts

def process_person_folder(folder_path, person_name, quality_threshold=35):
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    folder = Path(folder_path)
    total_faces_saved = 0
    total_augmented = 0
    total_rejected = 0
    all_pose_counts = {"frontal": 0, "profile": 0, "looking_up": 0, "looking_down": 0, "angled": 0}

    files_to_process = []
    for ext in video_extensions + image_extensions:
        files_to_process.extend(folder.glob(f"*{ext}"))
        files_to_process.extend(folder.glob(f"*{ext.upper()}"))

    if not files_to_process:
        logger.warning(f"No media files found in {folder_path}")
        return 0, 0, 0, all_pose_counts

    logger.info(f"Processing folder for {person_name}: {len(files_to_process)} files")
    for file_path in files_to_process:
        logger.info(f"Processing: {file_path.name}")
        if file_path.suffix.lower() in video_extensions:
            faces_saved, augmented_faces, faces_rejected, pose_counts = process_video(str(file_path), person_name, quality_threshold)
        else:
            faces_saved, augmented_faces, faces_rejected, pose_counts = process_single_image(str(file_path), person_name, quality_threshold)
        total_faces_saved += faces_saved
        total_augmented += augmented_faces
        total_rejected += faces_rejected
        for pose, count in pose_counts.items():
            all_pose_counts[pose] += count

    logger.info(f"Folder processing complete: {total_faces_saved} faces saved, {total_augmented} augmented, {total_rejected} rejected")
    return total_faces_saved, total_augmented, total_rejected, all_pose_counts

def generate_embeddings(dataset_dir, embeddings_dir, person_name=None):
    dataset_dir = Path(dataset_dir)
    embeddings_dir = Path(embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Creating embeddings directory: {embeddings_dir}")

    if not dataset_dir.exists():
        logger.error(f"Dataset directory {dataset_dir} does not exist")
        return

    if person_name:
        person_names = [person_name]
    else:
        person_names = [d.name for d in dataset_dir.iterdir() if d.is_dir()]

    total_embeddings = 0
    for person in person_names:
        person_dataset_dir = dataset_dir / person
        if not person_dataset_dir.is_dir():
            continue

        person_embeddings_file = embeddings_dir / f"{person}_embeddings.pt"
        logger.info(f"Processing embeddings for {person}: {person_embeddings_file}")
        existing_embeddings = []
        existing_image_paths = []
        if person_embeddings_file.exists():
            try:
                data = torch.load(person_embeddings_file)
                existing_embeddings = data["embeddings"].tolist()
                existing_image_paths = data["image_paths"]
                logger.info(f"Loaded {len(existing_embeddings)} existing embeddings for {person}")
            except Exception as e:
                logger.warning(f"Failed to load existing embeddings for {person}: {e}")

        new_embeddings = []
        new_image_paths = []
        for img_file in person_dataset_dir.glob("*.jpg"):
            if str(img_file) in existing_image_paths:
                continue
            try:
                embedding_obj = DeepFace.represent(
                    str(img_file), model_name="Facenet", enforce_detection=False
                )
                embedding = np.array(embedding_obj[0]["embedding"])
                new_embeddings.append(embedding)
                new_image_paths.append(str(img_file))
                logger.info(f"Generated embedding for {img_file.name}")
            except Exception as e:
                logger.debug(f"Could not generate embedding for {img_file}: {e}")

        if new_embeddings:
            embeddings = existing_embeddings + new_embeddings
            image_paths = existing_image_paths + new_image_paths
            logger.info(f"Saving embeddings to: {person_embeddings_file}")
            torch.save({"embeddings": torch.tensor(embeddings), "image_paths": image_paths}, person_embeddings_file)
            total_embeddings += len(new_embeddings)
            logger.info(f"Updated embeddings for {person}: {len(embeddings)} total embeddings")

    logger.info(f"Embedding generation complete: {total_embeddings} new embeddings added")

# ================ MAIN EXECUTION ================
def main():
    parser = argparse.ArgumentParser(description="Unified Face Processing Pipeline")
    parser.add_argument("--input", required=True, help="Path to video, image, or folder")
    parser.add_argument("--name", required=True, help="Person's name")
    parser.add_argument("--quality", type=int, default=35, help="Minimum face quality (default: 35)")
    args = parser.parse_args()

    logger.info("Face Processing Pipeline")
    logger.info(f"Processing input: {args.input} for {args.name}")

    ext = os.path.splitext(args.input)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
        process_single_image(args.input, args.name, args.quality)
    elif ext in (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"):
        process_video(args.input, args.name, args.quality)
    else:
        process_person_folder(args.input, args.name, args.quality)

    generate_embeddings("face_images", "face_embeddings", person_name=args.name)
    logger.info(f"Processing complete for {args.name}")

if __name__ == "__main__":
    main()