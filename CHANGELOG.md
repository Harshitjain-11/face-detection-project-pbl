# Changelog — Face Detection Project

> **Yahan aap dekh sakte hain ki code mein kya aur kahan badla gaya hai.**
> (Here you can see what was changed in the code and where.)

---

## v3 — Deep Learning Face Recognition

### Why deep learning?

LBPH (Local Binary Pattern Histograms) works on raw pixel patterns.  
It is easily confused by:
- Changes in lighting
- Slight pose variation
- Multiple people in training data

The **dlib ResNet face embedding model** (used by `face_recognition`) was trained on 3 million faces.  
It produces a **128-dimensional vector** that captures high-level facial features — not pixels.  
Result: same person from different angles/lighting → small distance; different people → large distance.

### What changed

#### `requirements.txt`

| Change | Why |
|--------|-----|
| `dlib` added | Underlying C++ library for the ResNet face model |
| `face_recognition` added | Python API for 128-d embeddings |

#### `app.py`

| Where | Old (LBPH) | New (Deep Learning) |
|-------|-----------|---------------------|
| Recognition model | `cv2.face.LBPHFaceRecognizer_create()` | `face_recognition.face_encodings()` — 128-d ResNet |
| Model storage | `recognizer.yml` + `labels.pickle` | `embeddings.pickle` → `{name: [array128d, …]}` |
| Globals | `recognizer`, `labels`, `model_lock`, `CONF_HIGH`, `CONF_MEDIUM` | `known_embeddings`, `embed_lock`, `DL_THRESHOLD_HIGH=0.45`, `DL_THRESHOLD_MEDIUM=0.55` |
| Training | Pixel-histogram training on grayscale ROIs with CLAHE + augmentation | `face_recognition.face_locations()` + `face_recognition.face_encodings()` on color originals |
| Recognition score | LBPH confidence 0–100 (lower=better) | L2 distance 0–1 (lower=better); same semantics, different scale |
| `gen_frames()` | Per-face LBPH predict inside lock | `dl_recognize()` batches **all live faces in one call** — more efficient |
| `capture_frame()` | Saved color original + grayscale CLAHE ROI | Saves color original only (grayscale ROIs no longer needed) |
| `/train` route | LBPH training on grayscale ROIs | DL embedding extraction from color originals via HOG face detector |
| `load_model()` | Loaded `recognizer.yml` + `labels.pickle` | Removed; replaced by `load_embeddings()` |
| `augment_images()` | Manual flip/brightness augmentation for LBPH | Removed — ResNet embeddings are inherently robust to lighting/pose |

#### New function: `dl_recognize()`

```python
def dl_recognize(rgb_frame, face_locations, current_embeddings) -> list:
    """
    Batched deep-learning recognition.
    Returns (name, distance) per face_location.
    distance in [0, 1]: 0 = identical, 1 = very different.
    """
```

- Calls `face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1)` **once** for all faces in the frame
- For each encoding, computes L2 distance to all stored embeddings with `face_recognition.face_distance()`
- Returns the nearest neighbour; caller applies `DL_THRESHOLD_HIGH / DL_THRESHOLD_MEDIUM`

#### `templates/index.html`

- Updated button label: "Enrol with Deep Learning"
- Updated hint text to mention 128-d ResNet vectors

---

## v2 — Recognition Fix & Anti-Spoofing Upgrade

### File: `requirements.txt`

| Kya badla | Kyun |
|-----------|------|
| `opencv-python` → `opencv-contrib-python` | **CRITICAL fix.** `cv2.face.LBPHFaceRecognizer_create()` is in the `contrib` package. Without this change the app crashes on import. |
| `numpy` explicitly added | Ensures correct numpy is installed alongside OpenCV. |

---

### File: `app.py`

| Where | What changed | Why |
|-------|-------------|-----|
| Line 12 | `import collections` added | Needed for `collections.deque` used in motion buffer. |
| Line 13 | `import shutil` added | Needed for `shutil.rmtree()` in delete endpoint. |
| Lines 43–44 | `CONF_HIGH 55→50`, `CONF_MEDIUM 80→75` | Stricter thresholds → fewer false positives. LBPH lower = more confident, so 50 means "very close match only". |
| Lines 47–50 | `MOTION_FRAMES = 6`, `MOTION_THRESHOLD = 2.5` | Config for new motion-based liveness. |
| Lines 53–55 | `_face_motion_buf`, `_face_buf_lock` | Thread-safe rolling frame buffer per tracked face position. |
| Lines 88–96 | `augment_images()` function added | Generates 4 variants per captured face (original + flip + bright + dark). Quadruples training data without extra captures. This is the main fix for "fails when multiple people are trained". |
| Lines 99–108 | `check_eye_liveness()` — **now requires 2 eyes** | `len(eyes) >= 1` → `len(eyes) >= 2`. Real faces have two eyes; a flat photo of a face rarely shows both at detector scale. |
| Lines 111–131 | `check_motion_liveness()` added | Keeps a rolling `deque(maxlen=6)` of 48×48 face crops. Calculates mean abs-diff between consecutive frames. If avg diff ≤ 2.5 px → static image/replay attack → `SPOOF? (static)`. |
| Lines 133–136 | `cleanup_face_buffers()` added | Removes buffers for faces that are no longer in the frame, preventing unbounded memory growth during a long session. |
| Lines 179–192 | `gen_frames()` — dual liveness | Now calls both `check_eye_liveness` AND `check_motion_liveness`. Shows `SPOOF? (no eyes)` or `SPOOF? (static)` with distinct reason. Also cleans up stale face buffers each frame. |
| Lines 191–192 | `face_key = (x // 50, y // 50)` | Approximate grid position used as the motion-buffer dictionary key to track individual faces in multi-face scenes. |
| Lines 268–277 | `train()` — data augmentation | `augment_images(img)` is called for every loaded face image, quadrupling the training set. This makes the model more robust for multiple users and better at recognising faces under varying lighting. |
| Lines 284–295 | `/delete_person/<name>` endpoint added | POST request deletes a person's entire folder from `static/uploads/`. Uses `sanitize_name` + `os.path.basename` to prevent path traversal. |

---

### File: `static/css/style.css`

| What changed | Why |
|-------------|-----|
| `.person-block-header` flex row added | Lays out the person name and delete button side-by-side. |
| `.person-name` — `margin-bottom` moved to `.person-block-header` | Needed because the name is now inside the header div. |
| `.btn-sm` utility class added | Smaller variant for the compact delete button in the gallery. |

---

### File: `templates/gallery.html`

| What changed | Why |
|-------------|-----|
| `<div class="person-block-header">` wraps name + delete button | New flex row layout for the card header. |
| `<button class="btn btn-danger btn-sm delete-btn" data-name="...">✕</button>` | Delete button per enrolled person. |
| `document.querySelectorAll('.delete-btn')` JS block | Wires each delete button: calls `POST /delete_person/<name>`, removes the card on success. |

---

## v1 — Initial UI Redesign & Security Fixes

### File: `app.py`

| Line(s) | What changed | Why |
|---------|-------------|-----|
| 29–32 | `camera_lock` + `model_lock` added | Thread-safety for camera and model access. |
| 46–47 | CLAHE added | Better contrast normalisation than `equalizeHist`. |
| 65–68 | Eye cascade loaded | Used for liveness detection. |
| 82–86 | `preprocess_face()` | Centralised resize + CLAHE. |
| 89–101 | `check_liveness()` | Basic eye-based liveness check. |
| 121–128 | `sanitize_name()` + `NAME_MAX_LEN` | Prevents special characters in person names. |
| 134–136 | `/live` route registered | Route was linked in HTML but not defined in Flask. |
| 222–232 | `os.path.basename()` in capture | **Security fix**: blocked `../` path traversal. |
| 278–350 | Training: `if img is None: continue` | Prevents crash on corrupt images. |
| 354–368 | Gallery: `sorted()` | Deterministic ordering. |
| 371–378 | Shutdown: `with camera_lock:` | Thread-safe shutdown. |

### File: `static/css/style.css`
Complete redesign: CSS variables, sticky nav, two-column layout, responsive grid, accessibility.

### File: `static/js/main.js`
- `cameraActive` flag replacing unreliable src-string check.
- `showMsg(text, type)` for colour-coded status messages.
- Progress bar for auto-capture.

### Files: `templates/index.html`, `live.html`, `gallery.html`
Modern two-column layout, sticky header, status dots, empty-state messages.

---

## Summary of All Changes (Quick Reference)

```
requirements.txt       ← CRITICAL: opencv-python → opencv-contrib-python + numpy
app.py                 ← Motion liveness, 2-eye check, augmentation, delete endpoint, stricter thresholds
static/css/style.css   ← person-block-header layout, btn-sm
templates/gallery.html ← Delete button per person + JS delete handler
CHANGELOG.md           ← This file
```

---

## How to run the project

```bash
pip install -r requirements.txt
python app.py
# Open in browser: http://127.0.0.1:5000
```

