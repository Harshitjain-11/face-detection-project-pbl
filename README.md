FaceVault – Real-Time Face Recognition System
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Flask-3.0+-black.svg?style=for-the-badge&logo=flask&logoColor=white" alt="Flask" />
  <img src="https://img.shields.io/badge/OpenCV-4.9+-green.svg?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/Computer_Vision-LBPH-orange.svg?style=for-the-badge" alt="LBPH" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="MIT License" />
</p>
<p align="center">
  <strong>A lightweight Flask + OpenCV face recognition system with real-time capture, training, and live identification.</strong>
</p>
---
Project Banner
```text
┌──────────────────────────────────────────────────────────────────────────┐
│                                FaceVault                                │
│                 Real-Time Face Recognition System                       │
│                    Flask • OpenCV • LBPH • Web UI                       │
└──────────────────────────────────────────────────────────────────────────┘
```
---
Table of Contents
Project Overview
Features
System Architecture
Complete Workflow Flowchart
Folder Structure
Installation Guide
Usage Guide
Technical Implementation
Why LBPH Instead of Deep Learning?
Challenges Faced & Solutions
Performance Optimizations
Why This Project Is Not Directly Deployable to Typical Cloud Hosts
Future Improvements
Screenshots
Results & Observations
Learning Outcomes
Author
License
---
Project Overview
FaceVault is a web-based face recognition application built with Flask and OpenCV. It allows a user to:
register a person name,
capture face images from a local webcam,
generate a training dataset,
train an LBPH recognition model,
and recognize faces in real time from a live video feed.
The project is designed as a practical computer vision portfolio project that demonstrates:
webcam handling,
image preprocessing,
face detection,
model training,
confidence-based recognition,
and a responsive browser UI.
This project is intentionally lightweight and CPU-friendly so it can run smoothly on a normal laptop without a dedicated GPU.
---
Features
Real-time face detection using OpenCV Haar Cascade.
Face enrollment with a simple browser-based form.
Automatic dataset creation with original images and cropped face samples.
LBPH training pipeline for local, fast, and reliable recognition.
Unknown face handling using confidence thresholds.
Multiple face detection support during live recognition.
Gallery view to inspect enrolled people and stored images.
Delete profile support for removing a person’s dataset.
Live MJPEG stream in the browser.
Improved camera synchronization to reduce freezes and capture conflicts.
Production-oriented logging and error pages for safer debugging.
---
System Architecture
```text
┌──────────────────────┐
│    User / Browser    │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│   Flask Application  │
│  (routes, templates) │
└─────────┬────────────┘
          │
   ┌──────┼───────────────────────────────────┐
   │      │                                   │
   ▼      ▼                                   ▼
┌───────────────┐                  ┌────────────────────┐
│ Camera Module  │                  │ Dataset Manager    │
│ OpenCV webcam  │                  │ uploads + crops    │
└───────┬───────┘                  └─────────┬──────────┘
        │                                    │
        ▼                                    ▼
┌──────────────────────┐         ┌──────────────────────┐
│ Face Detection       │         │ Training Pipeline    │
│ Haar Cascade         │         │ LBPH model creation  │
└─────────┬────────────┘         └─────────┬────────────┘
          │                                 │
          ▼                                 ▼
┌──────────────────────┐         ┌──────────────────────┐
│ Preprocessing        │         │ recognizer.yml       │
│ resize + CLAHE       │         │ labels.pickle        │
└─────────┬────────────┘         └─────────┬────────────┘
          │                                 │
          ▼                                 ▼
┌──────────────────────┐         ┌──────────────────────┐
│ Recognition Worker   │────────▶│ Confidence Scoring    │
│ background queue     │         │ confirm / tentative   │
└──────────────────────┘         └──────────────────────┘
```
---
Complete Workflow Flowchart
```text
START
  │
  ▼
Open application in browser
  │
  ▼
Start camera
  │
  ▼
Capture live frame from webcam
  │
  ├──────────────► No face detected ───────────────► Show warning / retry
  │
  ▼
Face detected
  │
  ▼
Capture snapshot for enrolled person
  │
  ▼
Save:
  ├─ original image
  └─ cropped face sample
  │
  ▼
Repeat capture until enough samples are collected
  │
  ▼
Train model
  │
  ├──────────────► Not enough data ───────────────► Ask user to capture more images
  │
  ▼
Save recognizer.yml + labels.pickle
  │
  ▼
Load trained model
  │
  ▼
Live recognition starts
  │
  ├──────────────► High confidence  ───────────────► Show green box + name
  ├──────────────► Medium confidence ───────────────► Show yellow box + name?
  └──────────────► Low confidence   ───────────────► Show red box + Unknown
  │
  ▼
END
```
---
Folder Structure
```text
face-detection-project-pbl/
├── app.py
├── requirements.txt
├── README.md
├── CHANGELOG.md
├── .gitignore
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── uploads/
│       └── <person_name>/
│           ├── originals/
│           │   └── *.jpg
│           └── *.jpg
└── templates/
    ├── index.html
    ├── gallery.html
    ├── live.html
    ├── 404.html
    └── 500.html
```
Generated files
These are created at runtime and should not be committed:
`recognizer.yml`
`labels.pickle`
`static/uploads/`
temporary logs and cache files
---
Installation Guide
Prerequisites
Python 3.11 recommended
Webcam or laptop camera
Git
Setup
```bash
git clone https://github.com/Harshitjain-11/face-detection-project-pbl.git
cd face-detection-project-pbl
```
Create and activate a virtual environment:
```powershell
python -m venv venv312
venv312\Scripts\activate
```
Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
Run the project:
```bash
python app.py
```
Open the application in your browser:
```text
http://127.0.0.1:5000
```
---
Usage Guide
Step	Action	Result
1	Open the app	Dashboard loads in the browser
2	Enter a name	A new person profile is created
3	Start camera	Live webcam feed begins
4	Capture faces	Original images and crops are saved
5	Train model	`recognizer.yml` and `labels.pickle` are generated
6	View live feed	Face labels appear in real time
7	Open gallery	Enrolled people and saved images are shown
Recommended enrollment workflow
Capture at least 20–30 images per person with small variations in:
angle,
lighting,
expression,
and distance from the camera.
---
Technical Implementation
Face Detection
The project uses OpenCV Haar Cascade to detect faces in grayscale frames. This method is fast and suitable for local CPU-based applications.
Face Preprocessing
Each detected face is:
cropped,
resized to a fixed size,
and enhanced using CLAHE to improve local contrast.
This helps make the dataset more consistent across different lighting conditions.
Training Pipeline
During training:
the app reads cropped face images from each person folder,
falls back to the original images if crops are missing,
and trains an LBPH classifier with integer labels.
Recognition Pipeline
During live recognition:
the camera frame is processed,
faces are detected,
each face is normalized,
and the LBPH model predicts identity and confidence.
Unknown faces are handled by confidence thresholds.
Confidence Logic
Lower confidence values mean a better match.
Score Range	Meaning
`< 50`	Confirmed match
`50–69`	Tentative match
`>= 70`	Unknown
These values were chosen for the current preprocessing pipeline and may be tuned after retraining.
---
Why LBPH Instead of Deep Learning?
Several deep learning approaches were considered, including:
FaceNet
DeepFace
Dlib embeddings
ArcFace
CNN-based classifiers
The final implementation uses LBPH because it fits the project goals better:
CPU friendly: runs smoothly without a GPU.
Fast training: new users can be added quickly.
Fast inference: works well for real-time browser streaming.
Lower setup complexity: no heavyweight model/runtime dependency.
Better for this scope: ideal for an academic or college-level project where reliability and responsiveness matter.
This is a deliberate engineering choice based on performance, simplicity, and the hardware available for the project.
---
Challenges Faced & Solutions
1) Camera lock contention
Problem: Stream handling and snapshot capture were competing for the webcam.  
Solution: The camera pipeline was reworked so that frame access is synchronized and snapshot capture uses the latest available frame safely.
2) Wrong faces being captured
Problem: A bystander in the frame could contaminate training data.  
Solution: The capture flow keeps the largest detected face only.
3) Inconsistent recognition
Problem: Recognition accuracy dropped when images had different lighting conditions.  
Solution: CLAHE preprocessing was added to normalize contrast before training and recognition.
4) Missing or incomplete training data
Problem: Sometimes only originals existed and no cropped samples were available.  
Solution: The training pipeline can re-detect faces from original images if the crop set is insufficient.
---
Performance Optimizations
Background recognition worker for smoother live streaming.
Frame throttling to reduce CPU usage.
Lock reduction to avoid camera freezes.
Fallback training from originals to avoid manual re-capture.
Confidence thresholds to reduce false positives.
---
Why This Project Is Not Directly Deployable to Typical Cloud Hosts
This version uses:
```python
cv2.VideoCapture(0)
```
That means the webcam is accessed from the machine running the Flask server.
On local development machines, that works well.  
On cloud platforms such as Railway, Render, or Vercel, the server usually does not have access to a physical webcam. Because of that, the current architecture is best suited for local execution and demos.
Future cloud-ready architecture
A browser-based webcam flow would solve this:
```text
Browser Webcam
   ↓
getUserMedia()
   ↓
Send image/frame to Flask API
   ↓
Recognition engine
   ↓
Return result to browser
```
That redesign would make cloud deployment possible.
---
Future Improvements
Browser-based webcam capture using `getUserMedia()`
Optional deep learning recognition mode
SQLite / PostgreSQL storage
Login system and role-based access
Attendance export
Mobile-friendly capture flow
Cloud deployment support
---
Screenshots
Add your screenshots here before publishing the README.
Home Page
![Home Page](https://via.placeholder.com/900x500.png?text=FaceVault+Home+Page)
Live Recognition
![Live Recognition](https://via.placeholder.com/900x500.png?text=Live+Recognition+View)
Training Screen
![Training Screen](https://via.placeholder.com/900x500.png?text=Training+Pipeline+Output)
Gallery
![Gallery](https://via.placeholder.com/900x500.png?text=Gallery+View)
---
Results & Observations
Works well for local webcam-based demo scenarios.
Suitable for multiple enrolled people.
Recognition quality improves with clean lighting and enough training samples.
Best performance is achieved on CPU-friendly environments.
---
Learning Outcomes
This project helped me learn:
Flask backend development
OpenCV integration
Camera handling and stream management
Dataset creation and preprocessing
LBPH face recognition
Debugging real-time computer vision pipelines
Practical trade-offs between classic ML and deep learning
---
Author
Harshit Jain  
B.Tech AIML Student
GitHub: Harshitjain-11
LinkedIn: Add your LinkedIn link here
---
License
This project is licensed under the MIT License.
