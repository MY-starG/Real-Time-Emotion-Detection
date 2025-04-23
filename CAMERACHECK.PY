import cv2

# Open the camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera is working!")
    ret, frame = camera.read()
    if ret:
        print("Frame captured successfully!")
    else:
        print("Error: Could not capture frame.")

# Release the camera
camera.release()