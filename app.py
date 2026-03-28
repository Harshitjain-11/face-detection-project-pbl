
try:
    import face_recognition
except ImportError as _exc:
    raise ImportError(
        "face_recognition is required. Install it with:\n"
        "  pip install dlib face_recognition"
    ) from _exc

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

CASCADE_PATH     = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EYE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
eye_cascade  = cv2.CascadeClassifier(EYE_CASCADE_PATH)

if face_cascade.empty():
    raise IOError("Haarcascade not loaded")

camera      = None
camera_lock = threading.Lock()

# Deep-learning embedding store {name: [128-d numpy array, …]}
known_embeddings: dict = {}
embed_lock = threading.Lock()

STOP_STREAM = False

# Tunables
TRAIN_MIN_IMAGES = 2
NAME_MAX_LEN     = 50

# Deep-learning recognition thresholds
# face_recognition uses L2 / Euclidean distance (0 = identical, 1 = very different)
DL_THRESHOLD_HIGH   = 0.45   # ≤ 0.45 → high-confidence match
DL_THRESHOLD_MEDIUM = 0.55   # ≤ 0.55 → possible match (shown with ?)
# > 0.55 → Unknown

# Motion-liveness parameters
MOTION_FRAMES    = 6     # frames kept in rolling buffer
MOTION_THRESHOLD = 2.5   # minimum mean pixel-diff to pass as "live"

# CLAHE — kept for liveness preprocessing
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Per-face motion buffers  { face_key -> deque of small gray frames }
_face_motion_buf: dict = {}
_face_buf_lock = threading.Lock()

# Embedding store file
DL_EMBED_FILE = "embeddings.pickle"


# ================== LOAD EMBEDDINGS ==================
def load_embeddings() -> None:
    """Load persisted deep-learning face embeddings from disk."""
    global known_embeddings
    with embed_lock:
        known_embeddings.clear()
        if os.path.exists(DL_EMBED_FILE):
            with open(DL_EMBED_FILE, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                known_embeddings.update(data)
                total = sum(len(v) for v in data.values())
                print(f"[INFO] DL embeddings loaded: {list(data.keys())} "
                      f"({total} vectors)")
            else:
                print("[WARN] embeddings.pickle has unexpected format — ignoring")
        else:
            print("[INFO] No DL embeddings found — model not trained yet")


load_embeddings()


# ================== CAMERA ==================
def get_camera():
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            time.sleep(0.3)
        return camera


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
def detect_faces(gray: np.ndarray) -> list:
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


# ================== DL RECOGNITION ==================
def dl_recognize(
    rgb_frame: np.ndarray,
    face_locations: list,
    current_embeddings: dict,
) -> list:
    """
    Run batched deep-learning face recognition for one video frame.

    face_locations: list of (top, right, bottom, left) — face_recognition format.
    current_embeddings: shallow-copied snapshot of known_embeddings.

    Returns a list of (name, distance) tuples, one per location.
    distance is in [0, 1]; lower = more similar.
    """
    if not face_locations:
        return []

    if not current_embeddings:
        return [("Unknown", 1.0)] * len(face_locations)

    try:
        encodings = face_recognition.face_encodings(
            rgb_frame, face_locations, num_jitters=1
        )
    except Exception:
        return [("Unknown", 1.0)] * len(face_locations)

    results = []
    for enc in encodings:
        best_dist = float("inf")
        best_name = "Unknown"
        for person_name, person_encs in current_embeddings.items():
            if not person_encs:
                continue
            dists = face_recognition.face_distance(person_encs, enc)
            min_d = float(np.min(dists))
            if min_d < best_dist:
                best_dist = min_d
                best_name = person_name
        results.append((best_name, best_dist))
    return results


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

        frame     = cv2.flip(frame, 1)
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces     = detect_faces(gray)

        active_keys: set = set()

        # Separate live faces (pass liveness) for batched DL recognition
        live_faces:   list = []  # (x, y, w, h)
        fr_locations: list = []  # (top, right, bottom, left)

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            if roi.size == 0:
                continue

            face_key = (x // 50, y // 50)
            active_keys.add(face_key)

            eye_live    = check_eye_liveness(roi)
            motion_live = check_motion_liveness(roi, face_key)
            is_live     = eye_live and motion_live

            if not is_live:
                label_text = "SPOOF? (no eyes)" if not eye_live else "SPOOF? (static)"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 200), 2)
                cv2.putText(frame, label_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 200), 2)
            else:
                live_faces.append((x, y, w, h))
                # face_recognition format: (top, right, bottom, left)
                fr_locations.append((y, x + w, y + h, x))

        # Snapshot embeddings (avoid holding lock during recognition)
        with embed_lock:
            cur_embeddings = {k: list(v) for k, v in known_embeddings.items()}

        recognition_results = dl_recognize(rgb_frame, fr_locations, cur_embeddings)

        for (x, y, w, h), (name, dist) in zip(live_faces, recognition_results):
            if dist <= DL_THRESHOLD_HIGH:
                label_text = f"{name} ({dist:.2f})"
                box_color  = (0, 200, 80)
                text_color = (0, 220, 80)
            elif dist <= DL_THRESHOLD_MEDIUM:
                label_text = f"{name}? ({dist:.2f})"
                box_color  = (0, 200, 200)
                text_color = (0, 220, 220)
            else:
                label_text = f"Unknown ({dist:.2f})"
                box_color  = (60, 60, 255)
                text_color = (60, 60, 255)

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
    data     = request.json or {}
    name     = sanitize_name(data.get('name', ''))
    img_data = data.get('image', '')

    if not name or not img_data:
        return jsonify({'status': 'fail', 'msg': 'Name or image missing'}), 400

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

    # Save the full-color original — used by the DL training pipeline
    cv2.imwrite(os.path.join(originals, f"{uuid.uuid4().hex}.jpg"), img)

    faces = detect_faces(gray)
    saved = sum(
        1 for (x, y, w, h) in faces
        if gray[y:y + h, x:x + w].size > 0
    )

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
    """
    Extract 128-dimensional deep-learning face embeddings
    (dlib ResNet — same model used in face_recognition library) from
    each person's captured original images and persist to embeddings.pickle.
    """
    new_embeddings: dict = {}
    skipped: list = []
    trained: list = []

    for person in sorted(os.listdir(UPLOAD_FOLDER)):
        person_path    = os.path.join(UPLOAD_FOLDER, person)
        originals_path = os.path.join(person_path, "originals")
        if not os.path.isdir(originals_path):
            continue

        orig_files = [
            f for f in os.listdir(originals_path)
            if f.lower().endswith('.jpg')
        ]

        person_encs: list = []
        for img_name in orig_files:
            try:
                img = face_recognition.load_image_file(
                    os.path.join(originals_path, img_name)
                )
            except Exception:
                continue
            # HOG detector is fast and reliable for frontal enrollment shots
            locs = face_recognition.face_locations(img, model="hog")
            if locs:
                encs = face_recognition.face_encodings(img, locs)
                person_encs.extend(encs)

        if len(person_encs) < TRAIN_MIN_IMAGES:
            skipped.append({'person': person, 'count': len(person_encs)})
            continue

        new_embeddings[person] = person_encs
        trained.append({'person': person, 'count': len(person_encs)})

    if not new_embeddings:
        return jsonify({
            'status': 'fail',
            'msg': (f'No faces found in captured images. '
                    f'Each person needs at least {TRAIN_MIN_IMAGES} frames '
                    f'with a clearly visible face.'),
            'skipped': skipped,
        }), 400

    with open(DL_EMBED_FILE, "wb") as f:
        pickle.dump(new_embeddings, f)

    load_embeddings()

    total_vecs = sum(len(v) for v in new_embeddings.values())
    return jsonify({
        'status': 'success',
        'msg': (f'Deep learning model enrolled {len(trained)} person(s) — '
                f'{total_vecs} face embedding(s) stored.'),
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
