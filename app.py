
from flask import Flask, render_template, request, Response, jsonify
import os
import time
import uuid
import base64
import pickle
import re
import math
import cv2
import numpy as np

app = Flask(__name__)

# ================== CONFIG ==================
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise IOError("Haarcascade not loaded")

camera = None
recognizer = None
labels = {}

STOP_STREAM = False

# Tunables
FACE_SIZE = (200, 200)
TRAIN_MIN_IMAGES = 2


# ================== LOAD MODEL ==================
def load_model():
    global recognizer, labels
    labels.clear()

    if os.path.exists("recognizer.yml") and os.path.exists("labels.pickle"):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("recognizer.yml")
        with open("labels.pickle", "rb") as f:
            raw = pickle.load(f)
        labels.update({v: k for k, v in raw.items()})
        print("[INFO] Model loaded:", labels)
    else:
        recognizer = None
        print("[INFO] No trained model found")


load_model()


# ================== CAMERA ==================
def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        time.sleep(0.3)
    return camera


# ================== FACE DETECTION ==================
def detect_faces(gray):
    try:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(40, 40)
        )
        if isinstance(faces, np.ndarray):
            return faces.tolist()
        return []
    except cv2.error:
        return []


# ================== ROUTES ==================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    global STOP_STREAM
    STOP_STREAM = False
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def gen_frames():
    global STOP_STREAM, recognizer

    while True:
        if STOP_STREAM:
            break

        cam = get_camera()
        ret, frame = cam.read()
        if not ret:
            continue

        # 🔥 FORCE REAL-WORLD VIEW (NO MIRROR)
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            roi = cv2.resize(roi, FACE_SIZE)
            roi = cv2.equalizeHist(roi)

            name = "Unknown"
            label = "Unknown"
            color = (0, 0, 255)

            if recognizer is not None:
                try:
                    id_, conf = recognizer.predict(roi)

                    if conf < 70:
                        name = labels.get(id_, "Unknown")
                        label = "High Match"
                        color = (0, 255, 0)
                    elif conf < 100:
                        name = labels.get(id_, "Unknown")
                        label = "Medium Match"
                        color = (0, 255, 255)

                except cv2.error:
                    pass

            text = name if name == "Unknown" else f"{name} - {label}"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')


# ================== SANITIZE ==================
def sanitize_name(name):
    name = str(name).strip()
    name = re.sub(r'\s+', ' ', name)
    name = re.sub(r'[^\w\s\-]', '', name)
    return name


# ================== CAPTURE ==================
@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    data = request.json
    name = sanitize_name(data.get('name'))
    img_data = data.get('image')

    if not name or not img_data:
        return jsonify({'status': 'fail'}), 400

    person_path = os.path.join(UPLOAD_FOLDER, name)
    originals = os.path.join(person_path, "originals")
    os.makedirs(originals, exist_ok=True)

    img_bytes = base64.b64decode(img_data.split(',')[1])
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    # 🔥 SAME FLIP AS LIVE STREAM
    img = cv2.flip(img, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(originals, f"{uuid.uuid4().hex}.jpg"), img)

    faces = detect_faces(gray)
    saved = 0

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, FACE_SIZE)
        roi = cv2.equalizeHist(roi)
        cv2.imwrite(os.path.join(person_path, f"{uuid.uuid4().hex}.jpg"), roi)
        saved += 1

    return jsonify({'status': 'success', 'saved': saved})


# ================== TRAIN ==================
@app.route('/train', methods=['POST'])
def train():
    global recognizer

    x_train = []
    y_train = []
    label_ids = {}
    current_id = 0

    for person in sorted(os.listdir(UPLOAD_FOLDER)):
        person_path = os.path.join(UPLOAD_FOLDER, person)
        if not os.path.isdir(person_path):
            continue

        images = []
        for f in os.listdir(person_path):
            full = os.path.join(person_path, f)
            if os.path.isfile(full) and f.lower().endswith('.jpg'):
                images.append(f)

        if len(images) < TRAIN_MIN_IMAGES:
            continue

        label_ids[person] = current_id

        for img_name in images:
            img = cv2.imread(os.path.join(person_path, img_name),
                             cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, FACE_SIZE)
            img = cv2.equalizeHist(img)
            x_train.append(img)
            y_train.append(current_id)

        current_id += 1

    if not x_train:
        return jsonify({'status': 'fail', 'msg': 'No training data'}), 400

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(x_train, np.array(y_train))
    recognizer.save("recognizer.yml")

    with open("labels.pickle", "wb") as f:
        pickle.dump(label_ids, f)

    load_model()
    return jsonify({'status': 'success'})


# ================== GALLERY ==================
@app.route('/gallery')
def gallery():
    people = []
    for person in os.listdir(UPLOAD_FOLDER):
        originals = os.path.join(UPLOAD_FOLDER, person, "originals")
        if not os.path.isdir(originals):
            continue
        images = [f"/static/uploads/{person}/originals/{img}"
                  for img in os.listdir(originals)]
        people.append({'name': person, 'images': images})
    return render_template('gallery.html', people=people)


# ================== SHUTDOWN ==================
@app.route('/shutdown', methods=['POST'])
def shutdown():
    global STOP_STREAM, camera
    STOP_STREAM = True
    if camera:
        camera.release()
        camera = None
    cv2.destroyAllWindows()
    return jsonify({'status': 'success'})


# ================== MAIN ==================
if __name__ == "__main__":
    try:
        app.run(debug=False)
    finally:
        if camera:
            camera.release()
        cv2.destroyAllWindows()
