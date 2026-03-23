
from flask import Flask, render_template, request, Response, jsonify
import os
import time
import uuid
import base64
import pickle
import re
import threading
import collections
import shutil
import cv2
import numpy as np

app = Flask(__name__)

# ================== CONFIG ==================
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EYE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
eye_cascade  = cv2.CascadeClassifier(EYE_CASCADE_PATH)

if face_cascade.empty():
    raise IOError("Haarcascade not loaded")

camera      = None
camera_lock = threading.Lock()

recognizer  = None
model_lock  = threading.Lock()
labels: dict = {}

STOP_STREAM = False

# Tunables
FACE_SIZE      = (200, 200)
TRAIN_MIN_IMAGES = 2
NAME_MAX_LEN   = 50

# Confidence thresholds (LBPH: lower value = better match)
CONF_HIGH   = 50   # ≤ 50  → recognised with high confidence
CONF_MEDIUM = 75   # ≤ 75  → recognised with medium confidence
# > 75 → Unknown

# Motion-liveness parameters
MOTION_FRAMES    = 6     # frames kept in rolling buffer
MOTION_THRESHOLD = 2.5   # minimum mean pixel-diff to pass as "live"

# CLAHE for improved contrast normalisation
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Per-face motion buffers  { face_key -> deque of small gray frames }
_face_motion_buf: dict = {}
_face_buf_lock = threading.Lock()


# ================== LOAD MODEL ==================
def load_model() -> None:
    global recognizer, labels
    with model_lock:
        labels.clear()
        if os.path.exists("recognizer.yml") and os.path.exists("labels.pickle"):
            new_rec = cv2.face.LBPHFaceRecognizer_create()
            new_rec.read("recognizer.yml")
            with open("labels.pickle", "rb") as f:
                raw = pickle.load(f)
            labels.update({v: k for k, v in raw.items()})
            recognizer = new_rec
            print("[INFO] Model loaded:", labels)
        else:
            recognizer = None
            print("[INFO] No trained model found")


load_model()


# ================== CAMERA ==================
def get_camera():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            time.sleep(0.3)
        return camera


# ================== PREPROCESSING ==================
def preprocess_face(roi):
    """Resize and apply CLAHE to a grayscale face ROI."""
    roi_resized = cv2.resize(roi, FACE_SIZE)
    return clahe.apply(roi_resized)


# ================== AUGMENTATION ==================
def augment_images(img: np.ndarray) -> list:
    """
    Return a list of augmented variants of a preprocessed face image.
    Augmentations: original, horizontal flip, +30 brightness, -30 brightness.
    Quadrupling effective training data without extra captures.
    """
    variants = [img]
    variants.append(cv2.flip(img, 1))
    variants.append(np.clip(img.astype(np.int16) + 30, 0, 255).astype(np.uint8))
    variants.append(np.clip(img.astype(np.int16) - 30, 0, 255).astype(np.uint8))
    return variants


# ================== LIVENESS: EYE DETECTION ==================
def check_eye_liveness(face_roi_gray: np.ndarray) -> bool:
    """
    Require at least 2 eyes detected inside the face ROI.
    Printed photos rarely expose both eyes at the expected scale.
    """
    eyes = eye_cascade.detectMultiScale(
        face_roi_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20),
    )
    return len(eyes) >= 2


# ================== LIVENESS: MOTION DETECTION ==================
def check_motion_liveness(face_roi_gray: np.ndarray, face_key: tuple) -> bool:
    """
    Frame-difference anti-spoofing.
    Keeps a rolling buffer of small face ROIs per tracked face position.
    A static photo/screen replay shows near-zero inter-frame difference.
    Returns True once enough motion is observed (or while buffer is filling).
    """
    small = cv2.resize(face_roi_gray, (48, 48))
    with _face_buf_lock:
        if face_key not in _face_motion_buf:
            _face_motion_buf[face_key] = collections.deque(maxlen=MOTION_FRAMES)
        buf = _face_motion_buf[face_key]
        buf.append(small)
        if len(buf) < 2:
            return True  # not enough data yet — give benefit of the doubt
        diffs = [
            float(np.mean(cv2.absdiff(buf[i - 1], buf[i])))
            for i in range(1, len(buf))
        ]
    return float(np.mean(diffs)) > MOTION_THRESHOLD


def cleanup_face_buffers(active_keys: set) -> None:
    """Remove motion buffers for faces that are no longer visible."""
    with _face_buf_lock:
        stale = [k for k in _face_motion_buf if k not in active_keys]
        for k in stale:
            del _face_motion_buf[k]


# ================== FACE DETECTION ==================
def detect_faces(gray):
    try:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        if isinstance(faces, np.ndarray):
            return faces.tolist()
        return []
    except cv2.error:
        return []


# ================== SANITIZE ==================
def sanitize_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r'\s+', ' ', name)
    name = re.sub(r'[^\w\s\-]', '', name)
    return name[:NAME_MAX_LEN]


# ================== ROUTES ==================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/live')
def live():
    return render_template('live.html')


@app.route('/video_feed')
def video_feed():
    global STOP_STREAM
    STOP_STREAM = False
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def gen_frames():
    global STOP_STREAM

    while True:
        if STOP_STREAM:
            break

        cam = get_camera()
        try:
            ret, frame = cam.read()
        except Exception:
            continue
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)

        active_keys: set = set()

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            if roi.size == 0:
                continue

            # Approximate location key for motion buffer (50px grid)
            face_key = (x // 50, y // 50)
            active_keys.add(face_key)

            # --- Dual liveness check ---
            eye_live    = check_eye_liveness(roi)
            motion_live = check_motion_liveness(roi, face_key)
            is_live     = eye_live and motion_live

            roi_proc = preprocess_face(roi)

            label_text = "Unknown"
            box_color  = (60, 60, 255)
            text_color = (60, 60, 255)

            if not is_live:
                if not eye_live:
                    label_text = "SPOOF? (no eyes)"
                else:
                    label_text = "SPOOF? (static)"
                box_color = (0, 0, 200)
                text_color = (0, 0, 200)
            else:
                with model_lock:
                    cur_rec = recognizer
                if cur_rec is not None:
                    try:
                        id_, conf = cur_rec.predict(roi_proc)
                        if conf <= CONF_HIGH:
                            name = labels.get(id_, "Unknown")
                            label_text = f"{name} ({conf:.0f})"
                            box_color  = (0, 200, 80)
                            text_color = (0, 220, 80)
                        elif conf <= CONF_MEDIUM:
                            name = labels.get(id_, "Unknown")
                            label_text = f"{name}? ({conf:.0f})"
                            box_color  = (0, 200, 200)
                            text_color = (0, 220, 220)
                        else:
                            label_text = f"Unknown ({conf:.0f})"
                    except cv2.error:
                        pass

            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2)

        cleanup_face_buffers(active_keys)

        try:
            ret, buffer = cv2.imencode('.jpg', frame)
        except Exception:
            continue
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')


# ================== CAPTURE ==================
@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    data      = request.json or {}
    name      = sanitize_name(data.get('name', ''))
    img_data  = data.get('image', '')

    if not name or not img_data:
        return jsonify({'status': 'fail', 'msg': 'Name or image missing'}), 400

    # Prevent path traversal
    safe_name = os.path.basename(name)
    if not safe_name or safe_name in ('.', '..'):
        return jsonify({'status': 'fail', 'msg': 'Invalid name'}), 400

    person_path = os.path.join(UPLOAD_FOLDER, safe_name)
    originals   = os.path.join(person_path, "originals")
    os.makedirs(originals, exist_ok=True)

    try:
        img_bytes = base64.b64decode(img_data.split(',')[1])
    except Exception:
        return jsonify({'status': 'fail', 'msg': 'Invalid image data'}), 400

    img_np = np.frombuffer(img_bytes, np.uint8)
    img    = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'status': 'fail', 'msg': 'Could not decode image'}), 400

    img  = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(originals, f"{uuid.uuid4().hex}.jpg"), img)

    faces = detect_faces(gray)
    saved = 0

    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        roi_proc = preprocess_face(roi)
        cv2.imwrite(os.path.join(person_path, f"{uuid.uuid4().hex}.jpg"), roi_proc)
        saved += 1

    if saved == 0:
        return jsonify({
            'status': 'success',
            'saved': 0,
            'msg': 'No face detected — position your face clearly in frame.',
        })
    return jsonify({
        'status': 'success',
        'saved': saved,
        'msg': f'{saved} face(s) captured for "{safe_name}".',
    })


# ================== TRAIN ==================
@app.route('/train', methods=['POST'])
def train():
    x_train    = []
    y_train    = []
    label_ids  = {}
    current_id = 0
    skipped    = []
    trained    = []

    for person in sorted(os.listdir(UPLOAD_FOLDER)):
        person_path = os.path.join(UPLOAD_FOLDER, person)
        if not os.path.isdir(person_path):
            continue

        images = [
            f for f in os.listdir(person_path)
            if os.path.isfile(os.path.join(person_path, f))
            and f.lower().endswith('.jpg')
        ]

        if len(images) < TRAIN_MIN_IMAGES:
            skipped.append({'person': person, 'count': len(images)})
            continue

        label_ids[person] = current_id
        loaded = 0

        for img_name in images:
            img = cv2.imread(os.path.join(person_path, img_name),
                             cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, FACE_SIZE)
            img = clahe.apply(img)
            # Data augmentation — multiply training samples per capture
            for variant in augment_images(img):
                x_train.append(variant)
                y_train.append(current_id)
            loaded += 1

        if loaded < TRAIN_MIN_IMAGES:
            del label_ids[person]
            skipped.append({'person': person, 'count': loaded})
            continue

        trained.append({'person': person, 'count': loaded})
        current_id += 1

    if not x_train:
        return jsonify({
            'status': 'fail',
            'msg': (f'No training data found. '
                    f'Each person needs at least {TRAIN_MIN_IMAGES} captured images.'),
            'skipped': skipped,
        }), 400

    new_rec = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )
    new_rec.train(x_train, np.array(y_train))
    new_rec.save("recognizer.yml")

    with open("labels.pickle", "wb") as f:
        pickle.dump(label_ids, f)

    load_model()
    return jsonify({
        'status': 'success',
        'msg': (f'Model trained on {len(trained)} person(s) '
                f'using {len(x_train)} sample(s) '
                f'(with augmentation).'),
        'trained': trained,
        'skipped': skipped,
    })


# ================== GALLERY ==================
@app.route('/gallery')
def gallery():
    people = []
    for person in sorted(os.listdir(UPLOAD_FOLDER)):
        originals = os.path.join(UPLOAD_FOLDER, person, "originals")
        if not os.path.isdir(originals):
            continue
        images = sorted([
            f"/static/uploads/{person}/originals/{img}"
            for img in os.listdir(originals)
            if img.lower().endswith('.jpg')
        ])
        people.append({'name': person, 'images': images})
    return render_template('gallery.html', people=people)


# ================== DELETE PERSON ==================
@app.route('/delete_person/<name>', methods=['POST'])
def delete_person(name):
    safe_name   = os.path.basename(sanitize_name(name))
    person_path = os.path.join(UPLOAD_FOLDER, safe_name)
    if not os.path.isdir(person_path):
        return jsonify({'status': 'fail', 'msg': 'Person not found.'}), 404
    try:
        shutil.rmtree(person_path)
    except OSError as exc:
        return jsonify({'status': 'fail', 'msg': str(exc)}), 500
    return jsonify({'status': 'success', 'msg': f'"{safe_name}" deleted.'})


# ================== SHUTDOWN ==================
@app.route('/shutdown', methods=['POST'])
def shutdown():
    global STOP_STREAM, camera
    STOP_STREAM = True
    with camera_lock:
        if camera:
            camera.release()
            camera = None
    return jsonify({'status': 'success'})


# ================== MAIN ==================
if __name__ == "__main__":
    try:
        app.run(debug=False)
    finally:
        with camera_lock:
            if camera:
                camera.release()
