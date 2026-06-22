import base64

import cv2
import numpy as np
from deepface import DeepFace
from flask import Flask, jsonify, render_template, request


app = Flask(__name__)

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

EMOTION_LABELS = {
    "angry": "Marah",
    "disgust": "Jijik",
    "fear": "Takut",
    "happy": "Senang",
    "sad": "Sedih",
    "surprise": "Terkejut",
    "neutral": "Netral",
}


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
        }

    analyzed_faces = []

    for index, (x, y, w, h) in enumerate(faces, start=1):
        pad = int(max(w, h) * 0.18)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame_bgr.shape[1], x + w + pad)
        y2 = min(frame_bgr.shape[0], y + h + pad)
        face_crop = frame_bgr[y1:y2, x1:x2]

        emotion_key = "unknown"
        confidence = 0.0

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
        except Exception:
            pass

        emotion_label = EMOTION_LABELS.get(emotion_key, emotion_key.title())

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"Wajah {index}: {emotion_label}",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            annotated,
            f"{confidence:.1f}%",
            (x, min(y + h + 24, frame_bgr.shape[0] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        analyzed_faces.append(
            {
                "id": index,
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "emotion_key": emotion_key,
                "emotion_label": emotion_label,
                "confidence": round(confidence, 1),
            }
        )

    summary_text = f"Total wajah: {len(analyzed_faces)}"
    cv2.putText(
        annotated,
        summary_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    return annotated, {
        "face_count": len(analyzed_faces),
        "faces": analyzed_faces,
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
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
