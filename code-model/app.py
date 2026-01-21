from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image

app = Flask(__name__)

# โหลดโมเดลและฟอนต์
model = load_model('sign_language_model.keras')
fontpath = "angsana.ttc"  # ฟอนต์ภาษาไทย
font = ImageFont.truetype(fontpath, 48)

# รายการท่าทาง
actions = ['กลับ', 'ขอบคุณ', 'คุณสบายดีไหม', 'ช่วย', 'เชื่อ', 'แนะนำ', 'พา', 'รอ', 'สวัสดี', 'อะไร']

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
sequence, prediction = [], []
threshold = 0.8

# ตัวแปรสำหรับเก็บสถานะการทำงาน
is_detecting = False
current_prediction = ""
confidence_score = 0.0

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 4))
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468, 3))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    return np.concatenate([pose.flatten(), face.flatten(), lh.flatten(), rh.flatten()])

def gen():
    cap = cv2.VideoCapture(1)  # เปลี่ยนเป็น 0 สำหรับกล้องหลัก
    global sequence, prediction, is_detecting, current_prediction, confidence_score

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while is_detecting:
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            action_text = ""
            confidence = 0.0

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                prediction.append(np.argmax(res))

                if np.unique(prediction[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        action_text = actions[np.argmax(res)]
                        confidence = float(res[np.argmax(res)])
                        current_prediction = action_text
                        confidence_score = confidence

            # แปลง OpenCV -> PIL เพื่อวาดข้อความไทย
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            if action_text:
                draw.text((30, 30), f"{action_text} ({confidence:.2f})", font=font, fill=(0, 255, 0))

            # กลับเป็น OpenCV
            image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html', actions=actions)

@app.route('/video')
def video():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection')
def start_detection():
    global is_detecting
    is_detecting = True
    return jsonify({'status': 'started'})

@app.route('/stop_detection')
def stop_detection():
    global is_detecting
    is_detecting = False
    return jsonify({'status': 'stopped'})

@app.route('/get_prediction')
def get_prediction():
    return jsonify({
        'prediction': current_prediction,
        'confidence': confidence_score
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
