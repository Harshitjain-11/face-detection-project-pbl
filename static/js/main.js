// main.js — FaceVault
// KEY CHANGE: captureOnce() now calls /capture_snapshot (server grabs frame
// directly from camera) instead of doing canvas.drawImage(videoFeed).
// This fixes the black feed / 0/20 frames problem completely.

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

let cameraActive  = false;
let autoCapturing = false;

// ── Helpers ───────────────────────────────────────────────────────────────────

function showMsg(text, type = 'info') {
  if (!msgBox) return;
  msgBox.textContent = text;
  msgBox.className   = 'status-msg show ' + type;
}

function showProgress(current, total) {
  if (!progressWrap) return;
  progressWrap.style.display = 'block';
  const pct = Math.round((current / total) * 100);
  if (progressBarFill) progressBarFill.style.width = pct + '%';
  if (progressText)    progressText.textContent     = `${current} / ${total}`;
}

function hideProgress() {
  if (progressWrap)    progressWrap.style.display  = 'none';
  if (progressBarFill) progressBarFill.style.width = '0%';
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Camera ────────────────────────────────────────────────────────────────────

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
  videoFeed.src = '';
  videoFeed.classList.remove('active');
  if (videoPlaceholder) videoPlaceholder.style.display = 'flex';
  startBtn.style.display = 'inline-flex';
  stopBtn.style.display  = 'none';
  if (camStatus) camStatus.classList.remove('active');
  cameraActive = false;
  try { await fetch('/shutdown', { method: 'POST' }); } catch (_) {}
  showMsg('Camera stopped.', 'info');
};

window.addEventListener('beforeunload', () => {
  if (cameraActive) {
    videoFeed.src = '';
    navigator.sendBeacon('/shutdown');
  }
});

// ── Capture — SERVER SIDE (no canvas needed) ──────────────────────────────────
// Server grabs the frame directly from OpenCV camera.
// Browser just sends the person name — nothing else.

async function captureOnce(name) {
  try {
    const res  = await fetch('/capture_snapshot', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ name }),
    });
    return await res.json();
  } catch (e) {
    return { status: 'fail', msg: 'Network error — is server running?' };
  }
}

// ── Manual capture ────────────────────────────────────────────────────────────

captureForm.onsubmit = async function (e) {
  e.preventDefault();
  if (!cameraActive) { showMsg('Start the camera first!', 'warn'); return; }
  const name = document.getElementById('name').value.trim();
  if (!name)  { showMsg('Please enter a name.', 'warn'); return; }
  showMsg('Capturing…', 'info');
  const data = await captureOnce(name);
  if (data.status === 'success') {
    showMsg(data.msg, data.saved === 0 ? 'warn' : 'success');
  } else {
    showMsg('Failed: ' + (data.msg || 'Unknown error'), 'error');
  }
};

// ── Auto Capture ──────────────────────────────────────────────────────────────

autoCaptureBtn.onclick = async function () {
  if (!cameraActive)  { showMsg('Start the camera first!', 'warn'); return; }
  if (autoCapturing)  { showMsg('Already running…', 'warn'); return; }
  const name = document.getElementById('name').value.trim();
  if (!name)          { showMsg('Enter a name first.', 'warn'); return; }

  autoCapturing           = true;
  autoCaptureBtn.disabled = true;

  const total   = 20;
  const delayMs = 800;
  let   saved   = 0;
  let   missed  = 0;

  showMsg(`Auto-capturing for "${name}"… look at camera slowly turning head`, 'info');
  showProgress(0, total);

  for (let i = 0; i < total; i++) {
    await sleep(delayMs);
    const data = await captureOnce(name);
    if (data?.status === 'success') {
      if (data.saved > 0) saved++;
      else                missed++;
    }
    showProgress(i + 1, total);
    showMsg(`Capturing ${i + 1}/${total} — ${saved} with face, ${missed} without`, 'info');
  }

  hideProgress();

  let summary = `Done! ${saved}/${total} frames with face saved.`;
  if (saved === 0) {
    summary = 'No faces captured — make sure camera is on and you are in frame!';
  } else if (saved < 5) {
    summary += ' Need 5+ — run Auto again.';
  } else {
    summary += ' Now click Train Recognition Model.';
  }
  showMsg(summary, saved >= 5 ? 'success' : 'warn');

  autoCapturing           = false;
  autoCaptureBtn.disabled = false;
};

// ── Train ─────────────────────────────────────────────────────────────────────

trainBtn.onclick = async function () {
  showMsg('Training… this takes 30–60 seconds, please wait.', 'info');
  trainBtn.disabled = true;
  try {
    const res  = await fetch('/train', { method: 'POST' });
    const data = await res.json();
    if (data.status === 'success') {
      let msg = data.msg;
      if (data.trained?.length) {
        msg += '\n✓ ' + data.trained.map(t => `${t.person} (${t.count})`).join(', ');
      }
      if (data.skipped?.length) {
        msg += '\n✗ Skipped: ' + data.skipped.map(s => `${s.person} — ${s.reason}`).join(', ');
      }
      showMsg(msg, 'success');
    } else {
      showMsg('Training failed:\n' + (data.msg || 'Unknown error'), 'error');
    }
  } catch (e) {
    showMsg('Training error — check terminal for details.', 'error');
  } finally {
    trainBtn.disabled = false;
  }
};