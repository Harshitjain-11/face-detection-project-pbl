let videoFeed = document.getElementById('videoFeed');
let videoBox = document.getElementById('videoBox');
let startBtn = document.getElementById('startBtn');
let stopBtn = document.getElementById('stopBtn');
let captureForm = document.getElementById('captureForm');
let trainBtn = document.getElementById('trainBtn');
let msgBox = document.getElementById('msg');

startBtn.onclick = function() {
  videoFeed.src = "/video_feed";
  videoBox.style.display = 'block';
  startBtn.style.display = 'none';
  stopBtn.style.display = 'inline-block';
};
stopBtn.onclick = function() {
  videoFeed.src = "";
  videoBox.style.display = 'none';
  startBtn.style.display = 'inline-block';
  stopBtn.style.display = 'none';
  fetch('/shutdown', {method: 'POST'});
};

captureForm.onsubmit = async function(e) {
  e.preventDefault();
  if (videoFeed.src === "") {
    msgBox.innerText = "Start webcam before capturing!";
    return;
  }
  let name = document.getElementById('name').value.trim();
  if (!name) {
    msgBox.innerText = "Name required!";
    return;
  }
  let canvas = document.createElement('canvas');
  canvas.width = videoFeed.width || 320;
  canvas.height = videoFeed.height || 240;
  let ctx = canvas.getContext('2d');
  ctx.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
  let dataURL = canvas.toDataURL('image/jpeg');
  let res = await fetch('/capture_frame', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name: name, image: dataURL})
  });
  let data = await res.json();
  msgBox.innerText = data.status === "success" ? "Captured!" : "Failed to capture";
};

trainBtn.onclick = async function() {
  let res = await fetch('/train', {method: 'POST'});
  let data = await res.json();
  msgBox.innerText = data.status === "success" ? "Model Trained!" : "Train failed: "+data.msg;
};