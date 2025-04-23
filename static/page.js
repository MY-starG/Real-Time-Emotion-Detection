let stream = null;

// Get camera list and populate dropdown
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
        startCamera(videoSelect.options[0].value); // Start with first camera
    }
});

// Start camera using selected deviceId
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

// Capture and send frames to Flask backend
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
                console.log('Detected Emotion:', data.emotion);
                document.getElementById('emotion-display').innerText = "Detected: " + data.emotion;
            });
        }, 'image/jpeg');
    }, 1000); // Send 1 frame/sec
}
