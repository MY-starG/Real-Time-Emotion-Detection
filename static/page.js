let stream = null;

navigator.mediaDevices.enumerateDevices().then(devices => {
  const videoSelect = document.createElement('select');
  videoSelect.id = 'cameraSelect';
  document.body.insertBefore(videoSelect, document.getElementById('video-container'));

  devices.forEach((device, index) => {
    if (device.kind === 'videoinput') {
      const option = document.createElement('option');
      option.value = device.deviceId;
      option.text = device.label || `Camera ${index + 1}`;
      videoSelect.appendChild(option);
    }
  });

  videoSelect.addEventListener('change', () => {
    startCamera(videoSelect.value);
  });

  if (videoSelect.options.length > 0) {
    startCamera(videoSelect.options[0].value);
  }
});

function startCamera(deviceId) {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
  }

  navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: deviceId } } })
    .then(s => {
      stream = s;
      const video = document.getElementById('video');
      video.srcObject = stream;
      sendFrames(video);
    })
    .catch(err => console.error("Camera access error:", err));
}

function sendFrames(video) {
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');

  setInterval(() => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(blob => {
      const formData = new FormData();
      formData.append('frame', blob);
      fetch('/process_frame', {
        method: 'POST',
        body: formData
      })
      .then(response => response.json())
      .then(data => {
        const box = document.getElementById('emotion-box');
        if (box && data.emotion && data.confidence !== undefined) {
          box.textContent = `${data.emotion} (${data.confidence.toFixed(2)}%)`;
          box.className = `emotion-${data.emotion}`;
        }
      });
    }, 'image/jpeg');
  }, 1000);
}
