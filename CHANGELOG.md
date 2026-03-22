# Changelog — Face Detection Project

> **Yahan aap dekh sakte hain ki code mein kya aur kahan badla gaya hai.**
> (Here you can see what was changed in the code and where.)

---

## Latest Changes

### File: `app.py` (Main Backend)

| Line(s) | Kya badla / What changed | Kyun / Why |
|---------|--------------------------|------------|
| 1–11 | `import math` hataya; `threading` add kiya | `math` kabhi use nahi hota tha. Threading locks ke liye zaroori hai. |
| 29–32 | `camera_lock` aur `model_lock` add kiye | Camera aur model ek saath multiple requests mein use hote the — crash hone ka risk tha. |
| 46–47 | `clahe = cv2.createCLAHE(...)` add kiya | Purana `equalizeHist` low-light mein kaam nahi karta tha. CLAHE better hai. |
| 65–68 | `EYE_CASCADE_PATH` + `eye_cascade` add kiya | Aankhon ki detection se pata chalega ki face real hai ya printed photo. |
| 82–86 | `preprocess_face()` function add kiya | Face ROI ko resize + CLAHE apply karna ek jagah karna behtar hai. |
| 89–100 | `check_liveness()` function add kiya | Aankhon ki ginatee karta hai — agar 0 hain to `SPOOF?` dikhayega. |
| 107–109 | `scaleFactor` 1.3 → 1.1 kiya | Chhote faces bhi detect hone lagenge. |
| 121–128 | `sanitize_name()` mein `NAME_MAX_LEN = 50` add kiya | Bahut lamba naam server par problem bana sakta tha. |
| 134–136 | `/live` route add kiya | Pehle yeh route tha hi nahi — HTML isko link karta tha magar Flask mein register nahi tha. |
| 171–196 | `gen_frames()` mein liveness check + CONF_HIGH/CONF_MEDIUM thresholds | Pehle thresholds 70/100 the (magic numbers). Ab named constants hain aur confidence number bhi dikhata hai. |
| 222–275 | `/capture_frame` mein `os.path.basename()` + `try/except` for base64 | **Security fix**: user `../` deke kisi bhi folder mein file likh sakta tha. Ab blocked hai. |
| 278–350 | `/train` mein `if img is None: continue` add kiya | Agar ek bhi image corrupt thi to poori training crash ho jaati thi. Ab skip hoti hai. |
| 354–368 | `/gallery` mein `sorted()` add kiya | Pehle random order mein dikhta tha. |
| 371–378 | `/shutdown` mein `with camera_lock:` add kiya | Thread-safe shutdown. |

---

### File: `static/css/style.css` (Styling)

| Kya badla / What changed | Kyun / Why |
|--------------------------|------------|
| Poora file rewrite — CSS variables (`:root`) add kiye | Ab ek jagah se colors change kar sakte hain (`--accent-blue`, `--surface`, etc.) |
| `.app-header` + `.app-nav` add kiya | Sticky navbar jo teeno pages par dikhega |
| `.main-content` grid layout (2 columns) | Pehle sab ek chhoti si box mein tha. Ab proper layout hai. |
| `.btn-primary`, `.btn-secondary`, `.btn-danger`, `.btn-accent`, `.btn-outline` | Different buttons ke liye alag styling |
| `.status-msg.info/.success/.error/.warn` | Green/Red/Yellow/Blue color-coded messages |
| `.progress-bar-track` + `.progress-bar-fill` | Auto-capture ke liye progress bar |
| `@media (prefers-reduced-motion: reduce)` | Accessibility — jo log animations off rakhte hain unke liye |
| Responsive: `@media (max-width: 900px)` + `(max-width: 600px)` | Mobile friendly |

---

### File: `static/js/main.js` (Frontend Logic)

| Line(s) | Kya badla / What changed | Kyun / Why |
|---------|--------------------------|------------|
| 1–91 | ~90 lines ka commented-out purana code hataya | Dead code tha — confusing tha |
| 17 | `let cameraActive = false;` add kiya | Pehle camera status check karne ka tarika galat tha (URL string se check karna unreliable hai) |
| 40–42 | Purana `isCameraOn()` function hataya | `cameraActive` flag se replace kiya — reliable hai |
| `startBtn.onclick` | `cameraActive = true` set karta hai | Camera on hone par flag update hota hai |
| `stopBtn.onclick` | `cameraActive = false` set karta hai | Camera off hone par flag update hota hai |
| 229–239 (purani file) | `flipBtn` wala code hataya | **Bug fix**: `flipBtn` element HTML mein kabhi tha hi nahi — page load par `ReferenceError` aata tha |
| `showMsg(text, type)` function | Typed messages (info/success/error/warn) | Ab messages color-coded hain |
| `showProgress` / `hideProgress` | Progress bar functions add kiye | Auto-capture progress dikhane ke liye |
| `autoCaptureBtn.onclick` | `data.saved > 0` check add kiya | Pehle success tab bhi count karta tha jab face detect na ho |

---

### File: `templates/index.html` (Home Page)

| Kya badla / What changed | Kyun / Why |
|--------------------------|------------|
| ~65 lines ka commented-out code hataya (file ke top par tha) | Dead code |
| Sticky header + navigation add kiya | Easy navigation teeno pages ke beech |
| `.panel.video-panel` + `.panel.control-panel` layout | Two-column layout — professional dikhta hai |
| Progress bar HTML add kiya (`#progressWrap`) | Auto-capture ke dauran progress dikhega |
| Status message div (`#msg`) ko typed styling mili | Color-coded feedback |
| Font change: Google Fonts se `Inter` + `Orbitron` load hota hai | Clean, modern font |

---

### File: `templates/gallery.html` (Gallery Page)

| Kya badla / What changed | Kyun / Why |
|--------------------------|------------|
| Sticky header + navigation add kiya | Consistent navigation |
| `gallery-tree` grid layout | Responsive grid — mobile par bhi theek dikhta hai |
| Empty state message add kiya | Agar koi face enrolled nahi hai to "No faces enrolled yet" dikhega |
| Image count per person | Har person ke neeche kitni images hain woh dikhega |

---

### File: `templates/live.html` (Live Detection Page)

| Kya badla / What changed | Kyun / Why |
|--------------------------|------------|
| Sticky header + navigation add kiya | Consistent navigation |
| Status dot (animated green dot) add kiya | Stream on/off visually dikhta hai |
| Status message typing fix | Start/stop par proper messages |
| `particles-js` background hataya | Live detection page par unnecessary tha, performance better |

---

## Summary of All Changes (Quick Reference)

```
app.py               ← Backend logic: security, liveness, threading, training fixes
static/css/style.css ← Complete UI redesign with CSS variables and components
static/js/main.js    ← Bug fixes, dead code removal, better state management
templates/index.html ← New layout, progress bar, clean structure
templates/gallery.html ← Grid layout, empty state, image count
templates/live.html  ← Navigation, status dot, clean layout
CHANGELOG.md         ← Yeh file (changes ki list)
```

---

## How to run the project

```bash
pip install flask opencv-contrib-python pillow
python app.py
# Browser mein kholein: http://127.0.0.1:5000
```
