from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import os
import cv2
from PIL import Image
import numpy as np
import pickle
import uuid
import base64

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

recognizer = None
labels = {}

def load_model():
    global recognizer, labels
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if os.path.exists("recognizer.yml"):
        recognizer.read("recognizer.yml")
    if os.path.exists("labels.pickle"):
        with open("labels.pickle", 'rb') as f:
            raw = pickle.load(f)
            labels.clear()
            labels.update({v: k for k, v in raw.items()})

# Initial load
load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    camera = cv2.VideoCapture(0)  # Open camera for each stream
    while True:
        ret, frame = camera.read()
        if not ret or frame is None:
            break
        # (Baaki code same...)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            name = "Unknown"
            conf = 0
            if recognizer:
                try:
                    id_, conf = recognizer.predict(roi)
                    if conf < 100:
                        name = labels.get(id_, "Unknown")
                except Exception:
                    pass
            text = f"{name} ({100-int(conf)}%)" if name != "Unknown" else "Unknown"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0) if name!="Unknown" else (0,0,255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    camera.release()  # Release when done
@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    data = request.json
    name = data.get('name')
    img_data = data.get('image')
    if not name or not img_data:
        return jsonify({'status': 'fail'}), 400
    save_path = os.path.join(UPLOAD_FOLDER, name)
    originals_path = os.path.join(save_path, "originals")
    os.makedirs(originals_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    img_str = img_data.split(',')[1]
    img_bytes = base64.b64decode(img_str)
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'status': 'fail'}), 400

    # Save original image
    orig_path = os.path.join(originals_path, f"{uuid.uuid4().hex}.jpg")
    cv2.imwrite(orig_path, img)

    # Detect face(s) and save cropped versions for training
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_found = False
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(save_path, f"{uuid.uuid4().hex}.jpg"), roi)
        face_found = True
    if not face_found:
        # Still return success so originals are retained for gallery
        return jsonify({'status': 'success', 'msg': 'No face detected, but original saved.'})
    return jsonify({'status': 'success'})

@app.route('/train', methods=['POST'])
def train():
    global recognizer  # <-- Yeh line add kar
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    label_ids = {}
    x_train = []
    y_labels = []
    current_id = 0

    # Walk through each person's folder in uploads
    for person in os.listdir(UPLOAD_FOLDER):
        person_path = os.path.join(UPLOAD_FOLDER, person)
        if not os.path.isdir(person_path):
            continue
        # Skip the "originals" subfolder
        for file in os.listdir(person_path):
            if file == "originals":
                continue  # Yeh line add kar, originals folder skip kare
            file_path = os.path.join(person_path, file)
            if os.path.isdir(file_path):  # skip directories
                continue
            if file.endswith(("jpg", "jpeg", "png")):
                label = person
                if label not in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                # Since these are already cropped faces, just add them!
                x_train.append(img)
                y_labels.append(id_)

    if not x_train:
        return jsonify({'status': 'fail', 'msg': 'No faces found!'}), 400
    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("recognizer.yml")
    with open("labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)
    load_model()
    return jsonify({'status': 'success'})

@app.route('/gallery')
def gallery():
    people = []
    for person in os.listdir(UPLOAD_FOLDER):
        originals_path = os.path.join(UPLOAD_FOLDER, person, "originals")
        if os.path.isdir(originals_path):
            images = []
            for img in os.listdir(originals_path):
                images.append(f"/static/uploads/{person}/originals/{img}")
            people.append({'name': person, 'images': images})
    return render_template("gallery.html", people=people)

@app.route('/shutdown', methods=['POST'])
def shutdown():
    camera.release()
    cv2.destroyAllWindows()
    return jsonify({'status': 'Camera stopped'})

if __name__ == "__main__":
    try:
        app.run(debug=True)
    finally:
        camera.release()
        cv2.destroyAllWindows()