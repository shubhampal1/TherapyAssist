import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from threading import Thread

# Setup MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Setup TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Feedback text to be read aloud
live_feedback_text = ""

# Landmarks to skip
excluded_landmarks = {
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.RIGHT_EYE,
    mp_pose.PoseLandmark.LEFT_EYE_INNER, mp_pose.PoseLandmark.RIGHT_EYE_INNER,
    mp_pose.PoseLandmark.LEFT_EYE_OUTER, mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
    mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR,
    mp_pose.PoseLandmark.MOUTH_LEFT, mp_pose.PoseLandmark.MOUTH_RIGHT,
}

# Helper to calculate angle
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# Draw landmarks (excluding face)
def draw_filtered_pose(image, landmarks, connections):
    h, w, _ = image.shape
    for idx, lm in enumerate(landmarks.landmark):
        if mp_pose.PoseLandmark(idx) not in excluded_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 4, (0, 255, 255), -1)
    for connection in connections:
        start_idx, end_idx = connection
        if (mp_pose.PoseLandmark(start_idx) not in excluded_landmarks and
            mp_pose.PoseLandmark(end_idx) not in excluded_landmarks):
            x1 = int(landmarks.landmark[start_idx].x * w)
            y1 = int(landmarks.landmark[start_idx].y * h)
            x2 = int(landmarks.landmark[end_idx].x * w)
            y2 = int(landmarks.landmark[end_idx].y * h)
            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

# Flask App
app = Flask(__name__)
CORS(app)
live_analysis_route = APIRouter()

@app.route("/live-analysis")
def live_analysis():
    cap = cv2.VideoCapture(0)

    def generate():
        global live_feedback_text
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (500, 700))
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    draw_filtered_pose(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    landmarks = results.pose_landmarks.landmark

                    # Example: Give real-time feedback for squats
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    angle = calculate_angle(hip, knee, ankle)

                    if angle < 90:
                        new_feedback = "Great squat form!"
                    else:
                        new_feedback = "Lower your hips for a deeper squat."

                    if new_feedback != live_feedback_text:
                        live_feedback_text = new_feedback
                        Thread(target=speak_feedback, args=(live_feedback_text,), daemon=True).start()

                _, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        finally:
            cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def speak_feedback(text):
    engine.say(text)
    engine.runAndWait()

@app.route("/get-feedback", methods=["GET"])
def get_feedback():
    return jsonify({"text": live_feedback_text})

if __name__ == '__main__':
    app.run(debug=True)
