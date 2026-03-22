// let videoFeed = document.getElementById('videoFeed');
// let videoBox = document.getElementById('videoBox');
// let startBtn = document.getElementById('startBtn');
// let stopBtn = document.getElementById('stopBtn');
// let captureForm = document.getElementById('captureForm');
// let trainBtn = document.getElementById('trainBtn');
// let msgBox = document.getElementById('msg');
// let autoCaptureBtn = document.getElementById('autoCaptureBtn');

// startBtn.onclick = function() {
//   videoFeed.src = "/video_feed";
//   videoBox.style.display = 'block';
//   startBtn.style.display = 'none';
//   stopBtn.style.display = 'inline-block';
// };
// stopBtn.onclick = function() {
//   videoFeed.src = "";
//   videoBox.style.display = 'none';
//   startBtn.style.display = 'inline-block';
//   stopBtn.style.display = 'none';
//   fetch('/shutdown', {method: 'POST'});
// };

// captureForm.onsubmit = async function(e) {
//   e.preventDefault();
//   if (videoFeed.src === "") {
//     msgBox.innerText = "Start webcam before capturing!";
//     return;
//   }
//   let name = document.getElementById('name').value.trim();
//   if (!name) {
//     msgBox.innerText = "Name required!";
//     return;
//   }
//   let canvas = document.createElement('canvas');
//   canvas.width = videoFeed.width || 320;
//   canvas.height = videoFeed.height || 240;
//   let ctx = canvas.getContext('2d');
//   ctx.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
//   let dataURL = canvas.toDataURL('image/jpeg');
//   let res = await fetch('/capture_frame', {
//     method: 'POST',
//     headers: {'Content-Type': 'application/json'},
//     body: JSON.stringify({name: name, image: dataURL})
//   });
//   let data = await res.json();
//   msgBox.innerText = data.status === "success" ? "Captured!" : "Failed to capture";
// };

// trainBtn.onclick = async function() {
//   let res = await fetch('/train', {method: 'POST'});
//   let data = await res.json();
//   msgBox.innerText = data.status === "success" ? "Model Trained!" : "Train failed: "+data.msg;
// };

// autoCaptureBtn.onclick = function() {
//   if (videoFeed.src === "") {
//     msgBox.innerText = "Start webcam before capturing!";
//     return;
//   }
//   let name = document.getElementById('name').value.trim();
//   if (!name) {
//     msgBox.innerText = "Name required!";
//     return;
//   }
//   msgBox.innerText = "Auto capturing 10 images...";
//   autoCaptureBtn.disabled = true;
//   let count = 0, i = 0;
//   let interval = setInterval(async () => {
//     let canvas = document.createElement('canvas');
//     canvas.width = videoFeed.width || 320;
//     canvas.height = videoFeed.height || 240;
//     let ctx = canvas.getContext('2d');
//     ctx.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
//     let dataURL = canvas.toDataURL('image/jpeg');
//     let res = await fetch('/capture_frame', {
//       method: 'POST',
//       headers: {'Content-Type': 'application/json'},
//       body: JSON.stringify({name: name, image: dataURL})
//     });
//     let data = await res.json();
//     if (data.status === "success") count++;
//     msgBox.innerText = `Captured ${i+1}/10...`;
//     i++;
//     if (i >= 10) {
//       clearInterval(interval);
//       msgBox.innerText = `Auto-capture done! ${count}/10 images saved.`;
//       autoCaptureBtn.disabled = false;
//     }
//   }, 1000); // 1 sec delay
// };
// static/js/main.js
let videoFeed = document.getElementById('videoFeed');
let videoBox = document.getElementById('videoBox');
let startBtn = document.getElementById('startBtn');
let stopBtn = document.getElementById('stopBtn');
let captureForm = document.getElementById('captureForm');
let trainBtn = document.getElementById('trainBtn');
let msgBox = document.getElementById('msg');
let autoCaptureBtn = document.getElementById('autoCaptureBtn');
// let flipBtn = document.getElementById('flipBtn');

function showMsg(m) {
  if (msgBox) msgBox.innerText = m;
}

// Safe start
startBtn.onclick = function() {
  videoFeed.src = "/video_feed";
  videoBox.style.display = 'block';
  startBtn.style.display = 'none';
  stopBtn.style.display = 'inline-block';
  showMsg("Webcam started...");
};

// Safe stop: wait for server to release camera, then clear src
stopBtn.onclick = async function() {
  startBtn.style.display = 'inline-block';
  stopBtn.style.display = 'none';
  try {
    showMsg("Stopping camera...");
    await fetch('/shutdown', {method: 'POST'});
    // small wait to let server close camera
    await new Promise(res => setTimeout(res, 300));
    videoFeed.src = "";
    videoBox.style.display = 'none';
    showMsg("Camera stopped.");
  } catch (e) {
    console.error(e);
    showMsg("Error stopping camera.");
  }
};

// helper: get displayed size of the img element
function getDisplayedSize(imgEl) {
  if (!imgEl) return {w: 320, h: 240};
  const rect = imgEl.getBoundingClientRect();
  return {w: Math.max(160, Math.floor(rect.width)), h: Math.max(120, Math.floor(rect.height))};
}

// Capture single frame (used by manual and auto)
async function captureOnce(name) {
  const {w, h} = getDisplayedSize(videoFeed);
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  // drawImage on <img> works for MJPEG stream as long as it has content
  try {
    ctx.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
  } catch (e) {
    console.warn("drawImage failed:", e);
    return {status: 'fail', msg: 'Capture failed (drawImage).'};
  }
  const dataURL = canvas.toDataURL('image/jpeg', 0.9);
  try {
    const res = await fetch('/capture_frame', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: name, image: dataURL})
    });
    return await res.json();
  } catch (e) {
    console.error("capture fetch error:", e);
    return {status: 'fail', msg: 'Network or server error.'};
  }
}

captureForm.onsubmit = async function(e) {
  e.preventDefault();
  if (videoFeed.src === "") {
    showMsg("Start webcam before capturing!");
    return;
  }
  let name = document.getElementById('name').value.trim();
  if (!name) {
    showMsg("Name required!");
    return;
  }
  showMsg("Capturing...");
  const data = await captureOnce(name);
  if (data.status === "success") showMsg(data.msg || "Captured!");
  else showMsg("Failed to capture: " + (data.msg || ""));
};

trainBtn.onclick = async function() {
  showMsg("Training... please wait");
  try {
    const res = await fetch('/train', {method: 'POST'});
    const data = await res.json();
    if (data.status === 'success') {
      showMsg(data.msg || "Model trained successfully!");
    } else {
      showMsg("Train failed: " + (data.msg || ""));
      if (data.skipped) showMsg("Skipped: " + JSON.stringify(data.skipped));
    }
  } catch (e) {
    console.error(e);
    showMsg("Training error: check server logs");
  }
};

// Sequential auto-capture: avoids overlapping requests
autoCaptureBtn.onclick = async function() {
  if (videoFeed.src === "") {
    showMsg("Start webcam before capturing!");
    return;
  }
  let name = document.getElementById('name').value.trim();
  if (!name) {
    showMsg("Name required!");
    return;
  }
  autoCaptureBtn.disabled = true;
  showMsg("Auto-capturing 10 images...");
  let successCount = 0;
  for (let i = 0; i < 10; i++) {
    // small wait so face detection has slightly different frames (and to avoid flooding)
    await new Promise(res => setTimeout(res, 700));
    const data = await captureOnce(name);
    if (data && data.status === 'success') successCount++;
    showMsg(`Auto capture: ${i+1}/10 — saved ${successCount}`);
  }
  showMsg(`Auto capture finished: ${successCount}/10 saved.`);
  autoCaptureBtn.disabled = false;
};

// Flip toggle
if (flipBtn) {
  flipBtn.onclick = async function() {
    try {
      const res = await fetch('/toggle_flip', {method: 'POST'});
      const data = await res.json();
      showMsg("Flip set to: " + data.flip);
    } catch (e) {
      showMsg("Toggle flip failed");
    }
  };
}