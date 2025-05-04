import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import csv
import time
import random
import os
from datetime import datetime
from flask import Flask, Response, request
import logging
from flask_cors import CORS
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Try to initialize TTS engine with fallback for macOS
try:
    engine = pyttsx3.init(driverName='nsss')  # for macOS
except:
    engine = pyttsx3.init()

engine.setProperty('rate', 150)

feedback_phrases = {
    'squat': {
        'correct': [
            "Great squats! Keep your knees behind your toes and chest up.",
            "Good job maintaining form on those squats, keep it up!"
        ],
        'incorrect': [
            "Try to squat lower so your thighs are parallel to the floor.",
            "Keep your back straight and don't let your knees go past your toes."
        ]
    },
    'push-up': {
        'correct': [
            "Awesome push-ups! Keep that body straight.",
            "Good form on those push-ups, you're doing great!"
        ],
        'incorrect': [
            "Engage your core and keep your body in a straight line for each push-up.",
            "Go a bit lower on your push-ups to get a full range of motion."
        ]
    },
    'lunge': {
        'correct': [
            "Nice lunges! Your form looks solid, keep it up.",
            "Great job on the lunges, you're keeping your balance well."
        ],
        'incorrect': [
            "Don't let your front knee extend past your toes on the lunge.",
            "Keep your upper body straight during the lunge for better balance."
        ]
    }
}

last_voice_msg = None
last_feedback_time = time.time()

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.before_request
def log_request_info():
    logger.debug(f"Request: {request.method} {request.url}")

@app.after_request
def log_response_info(response):
    logger.debug(f"Response: {response.status_code} {response.get_data(as_text=True)}")
    return response

live_analysis_route = APIRouter()

@live_analysis_route.get("")
async def live_analysis():
    cap = cv2.VideoCapture(0)

    def generate():
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from the camera.")
                    break

                frame = cv2.resize(frame, (700, 500))
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                    )

                _, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        finally:
            cap.release()

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

# Enhanced logging to debug voice message execution
@live_analysis_route.get("/exercise-analysis")
async def exercise_analysis(exercise: str):
    cap = cv2.VideoCapture(0)

    def log_reps(exercise, count):
        with open("exercise_log.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([exercise, count, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    def generate():
        global last_feedback_time, last_voice_msg, pose_detected
        rep_count = 0
        stage = None
        form_issue = False
        start_time = time.time()  # Track the start time
        initial_check_time = time.time()  # Track the time for the initial 2-second check

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from the camera.")
                    break

                frame = cv2.resize(frame, (700, 500))
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    pose_detected = True  # Pose detected
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                    )

                # Check if no pose is detected within 2 seconds of selecting the exercise
                if not pose_detected and time.time() - initial_check_time >= 2:
                    message = "No pose detected. Please position yourself in front of the camera."
                    logger.info(f"Voice message triggered: {message}")  # Log the message
                    try:
                        engine.say(message)
                        engine.runAndWait()
                        logger.info("Voice message executed successfully.")
                    except Exception as e:
                        logger.error(f"Error during voice message execution: {e}")
                    initial_check_time = time.time()  # Reset initial check time to avoid repeated messages

                # Check if no pose is detected within 10 seconds of start
                if not pose_detected and time.time() - start_time >= 10:
                    message = "No pose detected. Please adjust your position in front of the camera."
                    logger.info(f"Voice message triggered: {message}")  # Log the message
                    try:
                        engine.say(message)
                        engine.runAndWait()
                        logger.info("Voice message executed successfully.")
                    except Exception as e:
                        logger.error(f"Error during voice message execution: {e}")
                    start_time = time.time()  # Reset start time to avoid repeated messages

                cv2.putText(image, f"Reps: {rep_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                _, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        finally:
            cap.release()

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.route("/live-analysis/exercise-analysis")
def exercise_analysis():
    exercise = request.args.get("exercise", "squat").lower()
    cap = cv2.VideoCapture(0)

    def log_reps(exercise, count):
        with open("exercise_log.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([exercise, count, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    def generate():
        global last_feedback_time, last_voice_msg
        rep_count = 0
        stage = None
        form_issue = False
        results = None
        no_pose_start = time.time()
        no_pose_announced = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (700, 500))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            pose_detected = results and results.pose_landmarks

            if pose_detected:
                no_pose_start = time.time()
                no_pose_announced = False

                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                )

                landmarks = results.pose_landmarks.landmark

                if exercise == "squat":
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    angle = calculate_angle(hip, knee, ankle)
                    if angle > 160:
                        stage = "up"
                        form_issue = False
                    elif angle < 90 and stage == "up":
                        stage = "down"
                        rep_count += 1
                        engine.say(f"{rep_count}")
                        engine.runAndWait()
                        log_reps("squat", rep_count)
                    else:
                        form_issue = True

                elif exercise == "push-up":
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    angle = calculate_angle(shoulder, elbow, wrist)
                    if angle > 160:
                        stage = "up"
                        form_issue = False
                    elif angle < 90 and stage == "up":
                        stage = "down"
                        rep_count += 1
                        engine.say(f"{rep_count}")
                        engine.runAndWait()
                        log_reps("push-up", rep_count)
                    else:
                        form_issue = True

                elif exercise == "lunge":
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    angle = calculate_angle(hip, knee, ankle)
                    if angle > 160:
                        stage = "up"
                        form_issue = False
                    elif angle < 90 and stage == "up":
                        stage = "down"
                        rep_count += 1
                        engine.say(f"{rep_count}")
                        engine.runAndWait()
                        log_reps("lunge", rep_count)
                    else:
                        form_issue = True

            elif not no_pose_announced and time.time() - no_pose_start >= 4:
                feedback_msg = random.choice(feedback_phrases.get('no_pose', []))
                print("Speaking:", feedback_msg)
                engine.say(feedback_msg)
                engine.runAndWait()
                no_pose_announced = True

            if results is not None and time.time() - last_feedback_time >= 5:
                if pose_detected:
                    category = 'incorrect' if form_issue else 'correct'
                    messages = feedback_phrases.get(exercise, {}).get(category, [])
                else:
                    messages = feedback_phrases.get('no_pose', [])

                if messages:
                    if last_voice_msg in messages and len(messages) > 1:
                        messages = [m for m in messages if m != last_voice_msg]
                    feedback_msg = random.choice(messages)
                    print("Speaking:", feedback_msg)
                    last_voice_msg = feedback_msg
                    engine.say(feedback_msg)
                    engine.runAndWait()
                last_feedback_time = time.time()

            cv2.putText(image, f"Reps: {rep_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            _, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle
if __name__ == '__main__':
    app.run(debug=True)
