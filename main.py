import base64
import os
import time

import cv2
import numpy as np
from deepface import DeepFace
from flask import Flask, jsonify, render_template, request
from scipy.spatial import distance as dist

app = Flask(__name__)

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Eye cascade for drowsiness detection
EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

EMOTION_LABELS = {
    "angry": "Marah",
    "disgust": "Jijik",
    "fear": "Takut",
    "happy": "Senang",
    "sad": "Sedih",
    "surprise": "Terkejut",
    "neutral": "Netral",
}

# Drowsiness detection constants
EYE_AR_THRESH = 0.25  # Eye Aspect Ratio threshold
EYE_AR_CONSEC_FRAMES = 2  # Number of consecutive frames for drowsiness detection


def eye_aspect_ratio(eye_points):
    """
    Calculate the Eye Aspect Ratio (EAR) for drowsiness detection.
    """
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    # Compute the euclidean distance between the horizontal eye landmark
    C = dist.euclidean(eye_points[0], eye_points[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear


def detect_drowsiness(frame_bgr, faces):
    """
    Detect drowsiness by analyzing eye aspect ratio and other facial features.
    Returns: drowsy_status (bool), eye_state (str)
    """
    if len(faces) == 0:
        return False, "No face detected"

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    drowsy_count = 0
    total_eyes = 0

    for (x, y, w, h) in faces:
        # Region of Interest for eyes
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame_bgr[y:y + h, x:x + w]

        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )

        # If we found at least one eye, check for drowsiness
        if len(eyes) >= 1:
            # Look for both eyes (should be 2)
            if len(eyes) >= 2:
                # Sort eyes by x-coordinate (left to right)
                eyes = sorted(eyes, key=lambda e: e[0])
                
                # Extract eye regions
                eye1 = eyes[0]
                eye2 = eyes[1]
                
                # Check if eyes are properly positioned (roughly horizontal)
                if abs(eye1[1] - eye2[1]) < h * 0.5:
                    # Simple EAR calculation - if eyes are very small, might be closed
                    eye1_area = eye1[2] * eye1[3]
                    eye2_area = eye2[2] * eye2[3]
                    
                    # If eyes are small relative to face, likely closed
                    face_area = w * h
                    eye_area_ratio = (eye1_area + eye2_area) / face_area
                    
                    # Lower ratio means eyes are smaller (possibly closed)
                    if eye_area_ratio < 0.01:  # Threshold for closed eyes
                        drowsy_count += 1
                    total_eyes += 1
                    
            # Alternative: check if eyes are visible by area
            for (ex, ey, ew, eh) in eyes:
                eye_area = ew * eh
                face_area = w * h
                if eye_area / face_area < 0.005:  # Very small eyes (closed)
                    drowsy_count += 1
                total_eyes += 1

    # Determine drowsiness status
    if total_eyes > 0:
        drowsy_ratio = drowsy_count / total_eyes
        if drowsy_ratio > 0.5:  # More than 50% of eyes are closed/small
            return True, "Mengantuk"
        elif drowsy_ratio > 0.2:
            return False, "Mata Terbuka Sebagian"
        else:
            return False, "Mata Terbuka"
    else:
        return False, "Tidak Terdeteksi"


def analyze_faces(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    faces = sorted(faces, key=lambda rect: (rect[0], rect[1]))

    annotated = frame_bgr.copy()
    if len(faces) == 0:
        cv2.putText(
            annotated,
            "Tidak ada wajah terdeteksi",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        return annotated, {
            "face_count": 0,
            "faces": [],
            "drowsy_status": False,
            "drowsy_label": "Tidak Terdeteksi"
        }

    # Check drowsiness for all faces
    is_drowsy, drowsy_label = detect_drowsiness(frame_bgr, faces)
    
    # Override emotion detection for drowsy state
    emotion_label_override = "Mengantuk" if is_drowsy else None

    analyzed_faces = []
    sad_faces = 0

    for index, (x, y, w, h) in enumerate(faces, start=1):
        pad = int(max(w, h) * 0.18)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame_bgr.shape[1], x + w + pad)
        y2 = min(frame_bgr.shape[0], y + h + pad)
        face_crop = frame_bgr[y1:y2, x1:x2]

        emotion_key = "unknown"
        confidence = 0.0
        emotion_label = EMOTION_LABELS.get("neutral", "Netral")

        try:
            analysis = DeepFace.analyze(
                face_crop,
                actions=["emotion"],
                detector_backend="opencv",
                align=False,
                enforce_detection=False,
            )
            if isinstance(analysis, list):
                analysis = analysis[0] if analysis else {}

            emotion_key = (analysis.get("dominant_emotion") or "unknown").lower()
            emotion_scores = analysis.get("emotion") or {}
            confidence = float(emotion_scores.get(emotion_key, 0.0))
            
            # Get the base emotion label from the detected emotion
            base_emotion_label = EMOTION_LABELS.get(emotion_key, emotion_key.title())
            emotion_label = base_emotion_label
            
            # Count sad faces
            if emotion_key == "sad":
                sad_faces += 1

        except Exception as e:
            print(f"Emotion detection error: {e}")
            pass

        # Override with drowsiness if detected
        if is_drowsy and emotion_label_override:
            emotion_label = emotion_label_override
            confidence = 85.0  # High confidence for drowsiness detection

        # Additional check for sadness detection based on facial features
        # (This is a simple heuristic to enhance sadness detection)
        if emotion_key == "neutral" and not is_drowsy:
            # Check for possible sadness indicators
            gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            # Simple brightness check - sad faces often have lower contrast
            brightness = np.mean(gray_face)
            contrast = np.std(gray_face)
            if brightness < 100 and contrast < 50:  # Thresholds for possible sadness
                emotion_label = "Sedih"
                confidence = max(confidence, 65.0)

        # Draw rectangle and labels
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Different color for drowsy faces
        if is_drowsy:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 3)
        
        cv2.putText(
            annotated,
            f"Wajah {index}: {emotion_label}",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if not is_drowsy else (0, 0, 255),
            2,
        )
        cv2.putText(
            annotated,
            f"{confidence:.1f}%",
            (x, min(y + h + 24, frame_bgr.shape[0] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if not is_drowsy else (0, 0, 255),
            2,
        )

        analyzed_faces.append(
            {
                "id": index,
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "emotion_key": "drowsy" if is_drowsy else emotion_key,
                "emotion_label": emotion_label,
                "confidence": round(confidence, 1),
                "is_drowsy": is_drowsy,
            }
        )

    # Summary text
    summary_text = f"Total wajah: {len(analyzed_faces)}"
    if is_drowsy:
        summary_text += " ⚠️ MENGANTUK TERDETEKSI!"
    elif sad_faces > 0:
        summary_text += f" | {sad_faces} orang sedih"
    
    cv2.putText(
        annotated,
        summary_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255) if is_drowsy else (255, 255, 255),
        2,
    )

    return annotated, {
        "face_count": len(analyzed_faces),
        "faces": analyzed_faces,
        "drowsy_status": is_drowsy,
        "drowsy_label": drowsy_label,
        "sad_count": sad_faces,
    }


def decode_image_from_request():
    if "image" in request.files:
        return request.files["image"].read()

    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    if not image_data:
        return None

    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    return base64.b64decode(image_data)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/detect", methods=["POST"])
def api_detect():
    raw_image = decode_image_from_request()
    if not raw_image:
        return jsonify(error="Gambar tidak ditemukan"), 400

    image_array = np.frombuffer(raw_image, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(error="Gambar tidak bisa dibaca"), 400

    annotated, analysis = analyze_faces(frame)
    success, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        return jsonify(error="Gagal memproses gambar"), 500

    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return jsonify(image=f"data:image/jpeg;base64,{encoded}", **analysis)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
