// JavaScript to handle camera change 
function changeCamera(cameraIndex) {
    fetch('/change_camera/' + cameraIndex)
        .then(response => response.text())
        .then(data => {
            console.log(data);  // Log the response from the server

            // Update the video feed source to reflect the new camera
            var videoFeed = document.getElementById('live-video');
            // Update the video source URL with the new camera index
            videoFeed.src = "/video_feed_fer?camera=" + cameraIndex + "&time=" + new Date().getTime();
        });
}
