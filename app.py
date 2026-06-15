# app.py — FaceVault (LBPH, Fixed & Final)
# Back to LBPH — reliable, fast, no heavy dependencies
# Fixes vs all previous versions:
#   • No deadlock — camera lock only held during read, never nested
#   • Server-side capture (/capture_snapshot) — no canvas issues
#   • CONF thresholds tuned: 60 confirmed, 80 tentative
#   • Training re-detects from originals if crops are missing
#   • Full crash protection — terminal never closes unexpectedly
#   • Background thread for smooth stream

from __future__ import annotations

import base64
import json
import logging
import os
import pickle
import queue
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("facevault")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

BASE_DIR       = Path(__file__).parent.resolve()
UPLOAD_FOLDER  = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
MODEL_YML      = BASE_DIR / "recognizer.yml"
LABELS_PKL     = BASE_DIR / "labels.pickle"

# ── Tunables ──────────────────────────────────────────────────────────────────
FACE_SIZE      = (200, 200)
TRAIN_MIN_IMGS = 5
CONF_HIGH      = 50    # below this → confirmed match (green)
CONF_MEDIUM    = 70    # below this → tentative match (yellow)
                       # above 80   → Unknown (red)
ALLOWED_EXT    = {".jpg", ".jpeg", ".png", ".webp"}
JPEG_QUALITY   = 80
PROCESS_EVERY  = 3     # run recognition every 3rd frame

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ── Cascade classifiers ───────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    raise IOError("Face cascade not loaded")

# ── Model state ───────────────────────────────────────────────────────────────
_recognizer = None
_labels: dict[int, str] = {}
_model_lock = threading.Lock()

# ── Camera ────────────────────────────────────────────────────────────────────
_camera        = None
_cam_lock      = threading.Lock()   # only held during actual read — never nested
_latest_frame  = None               # most recent frame from stream (for capture)
_frame_lock    = threading.Lock()   # protects _latest_frame
_active_streams: dict[str, threading.Event] = {}


# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess(roi_gray: np.ndarray) -> np.ndarray:
    return clahe.apply(cv2.resize(roi_gray, FACE_SIZE))


def detect_faces(gray: np.ndarray) -> list:
    try:
        rects = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        return rects.tolist() if isinstance(rects, np.ndarray) else []
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def load_model() -> None:
    global _recognizer, _labels
    with _model_lock:
        _labels.clear()
        _recognizer = None
        if MODEL_YML.exists() and LABELS_PKL.exists():
            try:
                rec = cv2.face.LBPHFaceRecognizer_create()
                rec.read(str(MODEL_YML))
                with open(LABELS_PKL, "rb") as f:
                    raw: dict[str, int] = pickle.load(f)
                _labels = {int(v): k for k, v in raw.items()}
                _recognizer = rec
                log.info("Model loaded — persons: %s", list(_labels.values()))
            except Exception as e:
                log.error("Model load failed: %s", e)
        else:
            log.info("No model found — train first.")


load_model()


def _predict(roi: np.ndarray) -> tuple[str, float]:
    with _model_lock:
        rec = _recognizer
        lbl = dict(_labels)
    if rec is None:
        return "Train model first", 999.0
    try:
        label_id, conf = rec.predict(roi)
        if conf < CONF_HIGH:
            return lbl.get(label_id, "Unknown"), conf
        if conf < CONF_MEDIUM:
            return lbl.get(label_id, "Unknown") + "?", conf
        return "Unknown", conf
    except Exception as e:
        log.warning("Predict error: %s", e)
        return "Unknown", 999.0


# ═══════════════════════════════════════════════════════════════════════════════
# CAMERA  — lock only during read, never nested
# ═══════════════════════════════════════════════════════════════════════════════

def _open_camera() -> bool:
    """Open camera — call OUTSIDE _cam_lock to avoid blocking readers."""
    global _camera
    try:
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cam.set(cv2.CAP_PROP_FPS, 30)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        time.sleep(0.3)          # warm-up — safely outside any lock
        if not cam.isOpened():
            log.warning("Camera did not open")
            return False
        with _cam_lock:
            _camera = cam
        return True
    except Exception as e:
        log.error("Camera open failed: %s", e)
        return False


def _release_camera() -> None:
    global _camera, _latest_frame
    with _cam_lock:
        if _camera:
            try:
                _camera.release()
            except Exception:
                pass
            _camera = None
    with _frame_lock:
        _latest_frame = None


def read_frame() -> np.ndarray | None:
    """Read one frame. Lock held only during the actual read — never nested."""
    with _cam_lock:
        if _camera is None or not _camera.isOpened():
            return None            # caller must open camera separately
        try:
            ret, frame = _camera.read()
            if not ret:
                return None
            return cv2.flip(frame, 1)
        except Exception as e:
            log.warning("Frame read error: %s", e)
            return None


def get_latest_frame() -> np.ndarray | None:
    """Return the most recent frame captured by the stream (for snapshot)."""
    with _frame_lock:
        if _latest_frame is not None:
            return _latest_frame.copy()
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND RECOGNITION — smooth stream, recognition in separate thread
# ═══════════════════════════════════════════════════════════════════════════════

class RecognitionWorker:
    def __init__(self):
        self._q       = queue.Queue(maxsize=1)
        self._results : list[tuple] = []
        self._rlock   = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, frame: np.ndarray) -> None:
        if self._q.full():
            try: self._q.get_nowait()
            except queue.Empty: pass
        try: self._q.put_nowait(frame.copy())
        except queue.Full: pass

    def results(self) -> list[tuple]:
        with self._rlock:
            return list(self._results)

    def _run(self) -> None:
        while True:
            try:
                frame = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detect_faces(gray)
                out   = []
                for (x, y, w, h) in faces:
                    roi = gray[y:y+h, x:x+w]
                    if roi.size == 0:
                        continue
                    roi_proc    = preprocess(roi)
                    name, conf  = _predict(roi_proc)
                    out.append((x, y, w, h, name, conf))
                with self._rlock:
                    self._results = out
            except Exception as e:
                log.warning("Worker error: %s", e)


_worker = RecognitionWorker()


# ═══════════════════════════════════════════════════════════════════════════════
# MJPEG STREAM
# ═══════════════════════════════════════════════════════════════════════════════

def _draw(frame: np.ndarray, results: list[tuple]) -> np.ndarray:
    for (x, y, w, h, name, conf) in results:
        if "Unknown" in name or "Train" in name:
            color = (60, 60, 255)
        elif name.endswith("?"):
            color = (0, 200, 200)
        else:
            color = (0, 200, 80)

        label = f"{name} ({conf:.0f})" if conf < 999 else name
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(frame, (x, y-30), (x+w, y), color, cv2.FILLED)
        cv2.putText(frame, label, (x+4, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def _gen_frames(stop: threading.Event):
    global _latest_frame
    fc = 0
    cam_fail = 0
    target_dt = 1.0 / 20              # cap stream at ~20 FPS

    while not stop.is_set():
        t0 = time.monotonic()
        frame = read_frame()

        if frame is None:
            # Camera not ready — attempt (re)open with back-off
            cam_fail += 1
            if cam_fail <= 5:
                _open_camera()         # safe: called outside _cam_lock
            else:
                time.sleep(min(cam_fail * 0.5, 3.0))
            continue

        cam_fail = 0

        # Store latest frame for capture_snapshot()
        with _frame_lock:
            _latest_frame = frame

        fc += 1
        if fc % PROCESS_EVERY == 0:
            _worker.submit(frame)

        try:
            frame = _draw(frame, _worker.results())
        except Exception as e:
            log.warning("Draw error: %s", e)

        try:
            ret, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            if not ret:
                continue
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + buf.tobytes() + b"\r\n"
            )
        except Exception as e:
            log.warning("Encode error: %s", e)

        # FPS cap — avoid burning CPU needlessly
        elapsed = time.monotonic() - t0
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)


# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY
# ═══════════════════════════════════════════════════════════════════════════════

def _sanitize(name: str) -> str:
    name = re.sub(r"\s+", " ", str(name).strip())
    return re.sub(r"[^\w\s\-]", "", name)[:50]


def _safe_path(name: str) -> Path | None:
    safe = os.path.basename(_sanitize(name))
    if not safe or safe in (".", ".."):
        return None
    candidate = (UPLOAD_FOLDER / safe).resolve()
    if not str(candidate).startswith(str(UPLOAD_FOLDER.resolve())):
        return None
    return candidate


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/live")
def live_redirect():
    from flask import redirect
    return redirect("/")


@app.route("/gallery")
def gallery():
    people = []
    try:
        for pd in sorted(UPLOAD_FOLDER.iterdir()):
            if not pd.is_dir():
                continue
            orig = pd / "originals"
            if not orig.is_dir():
                continue
            imgs = sorted(
                f"/static/uploads/{pd.name}/originals/{f.name}"
                for f in orig.iterdir() if f.suffix.lower() in ALLOWED_EXT
            )
            crops = sum(
                1 for f in pd.iterdir()
                if f.is_file() and f.suffix.lower() in ALLOWED_EXT
            )
            people.append({
                "name": pd.name, "images": imgs,
                "crops": crops, "ready": crops >= TRAIN_MIN_IMGS,
            })
    except Exception as e:
        log.error("Gallery error: %s", e)
    return render_template("gallery.html", people=people)


@app.route("/video_feed")
def video_feed():
    stop   = threading.Event()
    req_id = str(uuid.uuid4())
    _active_streams[req_id] = stop

    def gen():
        try:
            yield from _gen_frames(stop)
        finally:
            _active_streams.pop(req_id, None)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/capture_snapshot", methods=["POST"])
def capture_snapshot():
    """Server grabs frame directly — browser sends only the name."""
    data = request.get_json(silent=True) or {}
    name = data.get("name", "")
    if not name:
        return jsonify({"status": "fail", "msg": "Name missing."}), 400

    person_path = _safe_path(name)
    if person_path is None:
        return jsonify({"status": "fail", "msg": "Invalid name."}), 400

    # Use latest frame from stream (avoids lock contention with _gen_frames)
    frame = get_latest_frame()
    if frame is None:
        # Fallback: try direct camera read when stream is not running
        if _camera is None:
            _open_camera()
        frame = read_frame()
    if frame is None:
        return jsonify({"status": "fail",
                        "msg": "Camera not ready. Start camera first."}), 503

    originals = person_path / "originals"
    originals.mkdir(parents=True, exist_ok=True)

    # Save original photo
    orig_file = originals / f"{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(orig_file), frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    # Detect and save face ROI crop for training
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)

    # Keep only the largest face to avoid mislabelling bystanders
    if len(faces) > 1:
        faces = [max(faces, key=lambda f: f[2] * f[3])]

    saved = 0
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        crop = preprocess(roi)
        cv2.imwrite(str(person_path / f"{uuid.uuid4().hex}.jpg"), crop)
        saved += 1

    msg = (
        f"✓ {saved} face(s) captured for \"{person_path.name}\"."
        if saved else
        "⚠ Photo saved but no face detected — face camera directly in good light."
    )
    return jsonify({"status": "success", "saved": saved, "msg": msg})


# Keep old route name working too
@app.route("/capture_frame", methods=["POST"])
def capture_frame():
    return capture_snapshot()


@app.route("/train", methods=["POST"])
def train():
    x_train:   list[np.ndarray] = []
    y_train:   list[int]        = []
    label_ids: dict[str, int]   = {}
    cur_id  = 0
    trained : list[dict] = []
    skipped : list[dict] = []

    try:
        persons = sorted(UPLOAD_FOLDER.iterdir())
    except Exception as e:
        return jsonify({"status": "fail", "msg": f"Cannot read uploads: {e}"}), 500

    for pd in persons:
        if not pd.is_dir():
            continue

        # Collect saved ROI crops
        crops = [
            f for f in pd.iterdir()
            if f.is_file() and f.suffix.lower() in ALLOWED_EXT
        ]

        rois: list[np.ndarray] = []
        for f in crops:
            try:
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Crops already have CLAHE from capture — just resize
                    rois.append(cv2.resize(img, FACE_SIZE))
            except Exception:
                continue

        # If not enough crops, re-detect from originals
        if len(rois) < TRAIN_MIN_IMGS:
            orig_dir = pd / "originals"
            if orig_dir.is_dir():
                for f in orig_dir.iterdir():
                    if f.suffix.lower() not in ALLOWED_EXT:
                        continue
                    try:
                        img  = cv2.imread(str(f))
                        if img is None:
                            continue
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        for (x, y, w, h) in detect_faces(gray):
                            roi = gray[y:y+h, x:x+w]
                            if roi.size > 0:
                                rois.append(preprocess(roi))
                    except Exception:
                        continue

        if len(rois) < TRAIN_MIN_IMGS:
            skipped.append({
                "person": pd.name,
                "count":  len(rois),
                "reason": f"only {len(rois)} face(s) found — need {TRAIN_MIN_IMGS}",
            })
            log.info("Skip %s — %d faces", pd.name, len(rois))
            continue

        label_ids[pd.name] = cur_id
        for roi in rois:
            x_train.append(roi)
            y_train.append(cur_id)
        trained.append({"person": pd.name, "count": len(rois)})
        log.info("Training %s — %d ROIs", pd.name, len(rois))
        cur_id += 1

    if not x_train:
        detail = " | ".join(
            f"{s['person']}: {s['reason']}" for s in skipped
        )
        return jsonify({
            "status":  "fail",
            "msg":     f"No training data. {detail}",
            "skipped": skipped,
        }), 400

    try:
        rec = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8
        )
        rec.train(x_train, np.array(y_train, dtype=np.int32))
        rec.save(str(MODEL_YML))
        with open(LABELS_PKL, "wb") as f:
            pickle.dump(label_ids, f)
        load_model()
    except Exception as e:
        log.error("Training failed: %s", e)
        return jsonify({"status": "fail", "msg": f"Training error: {e}"}), 500

    log.info("Done — %d person(s), %d images", len(trained), len(x_train))
    return jsonify({
        "status":  "success",
        "msg":     f"Trained {len(trained)} person(s) using {len(x_train)} face image(s).",
        "trained": trained,
        "skipped": skipped,
    })


@app.route("/delete_person", methods=["POST"])
def delete_person():
    data = request.get_json(silent=True) or {}
    path = _safe_path(data.get("name", ""))
    if path is None or not path.exists():
        return jsonify({"status": "fail", "msg": "Person not found."}), 404
    try:
        shutil.rmtree(path, ignore_errors=True)
        return jsonify({"status": "success", "msg": f'"{path.name}" removed.'})
    except Exception as e:
        return jsonify({"status": "fail", "msg": str(e)}), 500


@app.route("/shutdown", methods=["POST"])
def shutdown():
    for ev in list(_active_streams.values()):
        ev.set()
    _active_streams.clear()
    time.sleep(0.15)
    _release_camera()
    return jsonify({"status": "success"})


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def server_error(e):
    log.exception("Server error")
    return render_template("500.html"), 500


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        _release_camera()