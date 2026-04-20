// static/js/main.js

const videoFeed        = document.getElementById('videoFeed');
const videoPlaceholder = document.getElementById('videoPlaceholder');
const startBtn         = document.getElementById('startBtn');
const stopBtn          = document.getElementById('stopBtn');
const captureForm      = document.getElementById('captureForm');
const trainBtn         = document.getElementById('trainBtn');
const msgBox           = document.getElementById('msg');
const autoCaptureBtn   = document.getElementById('autoCaptureBtn');
const camStatus        = document.getElementById('camStatus');
const progressWrap     = document.getElementById('progressWrap');
const progressBarFill  = document.getElementById('progressBarFill');
const progressText     = document.getElementById('progressText');

// Explicit camera state flag — avoids unreliable src-string comparisons
let cameraActive = false;

// ── Helpers ─────────────────────────────────────────────────────────────────

function showMsg(text, type = 'info') {
  if (!msgBox) return;
  msgBox.textContent = text;
  msgBox.className = 'status-msg show ' + type;
}

function showProgress(current, total) {
  if (!progressWrap) return;
  progressWrap.style.display = 'block';
  const pct = Math.round((current / total) * 100);
  if (progressBarFill) progressBarFill.style.width = pct + '%';
  if (progressText)    progressText.textContent = current + ' / ' + total;
}

function hideProgress() {
  if (progressWrap) progressWrap.style.display = 'none';
  if (progressBarFill) progressBarFill.style.width = '0%';
}

// ── Camera controls ──────────────────────────────────────────────────────────

startBtn.onclick = function () {
  videoFeed.src = '/video_feed';
  videoFeed.classList.add('active');
  if (videoPlaceholder) videoPlaceholder.style.display = 'none';
  startBtn.style.display = 'none';
  stopBtn.style.display  = 'inline-flex';
  if (camStatus) camStatus.classList.add('active');
  cameraActive = true;
  showMsg('Camera starting…', 'info');
};

stopBtn.onclick = async function () {
  startBtn.style.display = 'inline-flex';
  stopBtn.style.display  = 'none';
  showMsg('Stopping camera…', 'info');
  try {
    await fetch('/shutdown', { method: 'POST' });
    await new Promise(r => setTimeout(r, 300));
    videoFeed.src = '';
    videoFeed.classList.remove('active');
    if (videoPlaceholder) videoPlaceholder.style.display = 'flex';
    if (camStatus) camStatus.classList.remove('active');
    cameraActive = false;
    showMsg('Camera stopped.', 'info');
  } catch (e) {
    console.error(e);
    showMsg('Error stopping camera.', 'error');
  }
};

// ── Capture helpers ──────────────────────────────────────────────────────────

function getDisplayedSize(imgEl) {
  if (!imgEl) return { w: 320, h: 240 };
  const rect = imgEl.getBoundingClientRect();
  return {
    w: Math.max(160, Math.floor(rect.width)),
    h: Math.max(120, Math.floor(rect.height))
  };
}

async function captureOnce(name) {
  const { w, h } = getDisplayedSize(videoFeed);
  const canvas = document.createElement('canvas');
  canvas.width  = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  try {
    ctx.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
  } catch (e) {
    console.warn('drawImage failed:', e);
    return { status: 'fail', msg: 'Video not ready — try again.' };
  }
  const dataURL = canvas.toDataURL('image/jpeg', 0.9);
  try {
    const res = await fetch('/capture_frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, image: dataURL })
    });
    return await res.json();
  } catch (e) {
    console.error('Capture error:', e);
    return { status: 'fail', msg: 'Network error.' };
  }
}

// ── Manual capture ───────────────────────────────────────────────────────────

captureForm.onsubmit = async function (e) {
  e.preventDefault();
  if (!cameraActive) {
    showMsg('Start the camera first!', 'warn');
    return;
  }
  const name = document.getElementById('name').value.trim();
  if (!name) {
    showMsg('Please enter a name.', 'warn');
    return;
  }
  showMsg('Capturing frame…', 'info');
  const data = await captureOnce(name);
  if (data.status === 'success') {
    showMsg(data.msg || 'Frame captured!', data.saved === 0 ? 'warn' : 'success');
  } else {
    showMsg('Capture failed: ' + (data.msg || 'Unknown error'), 'error');
  }
};

// ── Train model ──────────────────────────────────────────────────────────────

trainBtn.onclick = async function () {
  showMsg('Training model — please wait…', 'info');
  trainBtn.disabled = true;
  try {
    const res  = await fetch('/train', { method: 'POST' });
    const data = await res.json();
    if (data.status === 'success') {
      showMsg(data.msg || 'Model trained successfully!', 'success');
    } else {
      showMsg('Training failed: ' + (data.msg || 'Unknown error'), 'error');
    }
  } catch (e) {
    console.error(e);
    showMsg('Training error — check server logs.', 'error');
  } finally {
    trainBtn.disabled = false;
  }
};

// ── Auto capture ─────────────────────────────────────────────────────────────

autoCaptureBtn.onclick = async function () {
  if (!cameraActive) {
    showMsg('Start the camera first!', 'warn');
    return;
  }
  const name = document.getElementById('name').value.trim();
  if (!name) {
    showMsg('Please enter a name before capturing.', 'warn');
    return;
  }

  autoCaptureBtn.disabled = true;
  const total = 10;
  let successCount = 0;
  showProgress(0, total);

  for (let i = 0; i < total; i++) {
    await new Promise(r => setTimeout(r, 700));
    const data = await captureOnce(name);
    if (data && data.status === 'success' && data.saved > 0) successCount++;
    showProgress(i + 1, total);
    showMsg('Auto-capturing: ' + (i + 1) + '/' + total + ' (' + successCount + ' saved)', 'info');
  }

  hideProgress();
  showMsg(
    'Auto-capture complete: ' + successCount + '/' + total + ' frames saved.',
    successCount > 0 ? 'success' : 'warn'
  );
  autoCaptureBtn.disabled = false;
};
