from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Flask app setup
app = Flask(__name__)

# Load the trained model
model = load_model('model/asl_model.h5')

# Class names based on folder structure
class_names = [
    'A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I',
    'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R',
    'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

# Start webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Prediction pipeline with MediaPipe cropping
def preprocess(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)

    if results.multi_hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            x1, x2 = int(min(x_coords)) - 20, int(max(x_coords)) + 20
            y1, y2 = int(min(y_coords)) - 20, int(max(y_coords)) + 20

            # Clamp to image size
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            hand_crop = frame[y1:y2, x1:x2]
            try:
                resized = cv2.resize(hand_crop, (64, 64))
            except:
                resized = cv2.resize(frame, (64, 64))  # fallback
            normalized = resized / 255.0
            return np.expand_dims(normalized, axis=0)

    # fallback: use whole frame
    fallback = cv2.resize(frame, (64, 64))
    return np.expand_dims(fallback / 255.0, axis=0)

# Frame generator for video feed
def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(img_rgb)

        label = "No hand"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks and connections
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

            # Prediction only on the first hand
            hand_bbox_img = preprocess(frame)  # your custom crop
            preds = model.predict(hand_bbox_img)
            label = class_names[np.argmax(preds)]

        # Overlay label
        cv2.putText(frame, f"Prediction: {label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Stream it
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)