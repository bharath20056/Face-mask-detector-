from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)
camera = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Indices for nose tip, upper lip, lower lip landmarks (from Mediapipe Face Mesh)
NOSE_TIP_IDX = 1
UPPER_LIP_IDX = 13
LOWER_LIP_IDX = 14

def detect_mask(frame, landmarks, image_shape):
    h, w, _ = image_shape
    # Get landmark positions in pixel coords
    nose_tip = landmarks[NOSE_TIP_IDX]
    upper_lip = landmarks[UPPER_LIP_IDX]
    lower_lip = landmarks[LOWER_LIP_IDX]

    nose_x, nose_y = int(nose_tip.x * w), int(nose_tip.y * h)
    upper_lip_x, upper_lip_y = int(upper_lip.x * w), int(upper_lip.y * h)
    lower_lip_x, lower_lip_y = int(lower_lip.x * w), int(lower_lip.y * h)

    # Extract nose and mouth region rectangle
    x1 = max(nose_x - 20, 0)
    y1 = max(nose_y - 20, 0)
    x2 = min(lower_lip_x + 20, w)
    y2 = min(lower_lip_y + 20, h)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "No Mask", (0, 0, 255)

    # Convert ROI to HSV for skin detection
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Skin color HSV range (approximate)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    skin_ratio = cv2.countNonZero(skin_mask) / (roi.shape[0] * roi.shape[1])

    # If skin visible in nose/mouth region, assume NO MASK, else MASK
    if skin_ratio > 0.3:
        return "No Mask", (0, 0, 255)
    else:
        return "Mask", (0, 255, 0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                label, color = detect_mask(frame, face_landmarks.landmark, frame.shape)

                # Draw bounding box around face (rough rectangle around landmarks)
                x_coords = [int(lm.x * frame.shape[1]) for lm in face_landmarks.landmark]
                y_coords = [int(lm.y * frame.shape[0]) for lm in face_landmarks.landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(frame, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
