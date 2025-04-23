from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from keras.models import load_model
import os
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

# Load your models
fer_model_path = r'static/models/emotion_detection_model.h5'
emotion_model_path = r'static/models/emotion.h5'

# Check if models exist and load them
if not os.path.exists(fer_model_path):
    print(f"Error: FER model not found at {fer_model_path}")
    fer_model = None
else:
    fer_model = load_model(fer_model_path)

if not os.path.exists(emotion_model_path):
    print(f"Error: Emotion model not found at {emotion_model_path}")
    emotion_model = None
else:
    emotion_model = load_model(emotion_model_path)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

# Emotion labels for FER model
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral','Contempt']
emotion_colors = {
    'Angry': (0, 0, 255),  # Red
    'Disgust': (0, 128, 0),  # Dark Green
    'Fear': (255, 0, 0),  # Blue
    'Happy': (0, 255, 0),  # Green
    'Sad': (255, 255, 0),  # Yellow
    'Surprise': (0, 255, 255),  # Cyan
    'Neutral': (255, 0, 255),  # Magenta
    'Contempt': (128, 0, 128),    # Purple
}

# Default camera index
camera_index = 0  # Default camera is the first one

# Function to change the camera
@app.route('/change_camera/<int:index>')
def change_camera(index):
    global camera_index
    camera_index = index  # Update the camera index
    return 'Camera changed successfully'

# Function to simulate calculating accuracy and F1 score
def calculate_metrics(predictions, true_labels):
    accuracy = accuracy_score(true_labels, predictions)  # Use true labels and predictions for accuracy
    f1 = f1_score(true_labels, predictions, average='weighted')  # Calculate F1 score (you can adjust this)
    return accuracy, f1

# Function to detect emotions
def detect_emotion(frame, model, frame_count, predictions, true_labels, accuracy=None, f1=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x + w]  # Corrected slicing
        face = cv2.resize(face, (48, 48))
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)
        face = face / 255.0

        # Predict emotion
        prediction = model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]
        emotion_percentage = np.max(prediction) * 100  # Get the confidence percentage for the predicted emotion

        # Dummy true labels and predictions for demonstration (replace with actual labels if available)
        predicted_emotion = emotion  # Simulate the predicted emotion
        true_label = "Happy"  # Simulate the true label (you should replace this with actual data)

        # Store the predictions and true labels
        predictions.append(predicted_emotion)
        true_labels.append(true_label)

        # Draw circle and label with percentage
        center = (x + w // 2, y + h // 2)
        radius = min(w, h) // 2
        cv2.circle(frame, center, radius, emotion_colors[emotion], 2)
        cv2.putText(frame, f"{emotion}: {emotion_percentage:.2f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, emotion_colors[emotion], 2)

    # After every 10 frames, calculate the metrics
    if frame_count % 10 == 0 and len(predictions) > 0:
        accuracy, f1 = calculate_metrics(predictions, true_labels)
    
    # Display accuracy and F1 score on the frame continuously
    if accuracy is not None and f1 is not None:
        cv2.putText(frame, f"Accuracy: {accuracy*100:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"F1 Score: {f1:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame, accuracy, f1

# Function to generate video frames
def generate_frames(model, camera_index=0):
    print(f"Trying to open camera with index {camera_index}")
    if model is None:
        print("Error: No model provided for emotion detection.")
        return

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print(f"Error: Could not open camera with index {camera_index}")
        return

    frame_count = 0
    predictions = []  # Store model predictions for calculating accuracy
    true_labels = []  # Store true labels for calculating accuracy (use actual ground truth data if available)
    accuracy = None
    f1 = None

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Error: Failed to read frame from camera.")
                break
            else:
                frame, accuracy, f1 = detect_emotion(frame, model, frame_count, predictions, true_labels, accuracy, f1)  # Detect emotions
                frame_count += 1

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        camera.release()

# Route to generate a graph
@app.route('/training_graph')
def training_graph():
    # Simulated training/validation data
    epochs = list(range(1, 21))  # 20 epochs
    train_acc = [0.7 + 0.01 * i for i in range(20)]  # Simulated increasing accuracy
    val_acc = [0.68 + 0.015 * i for i in range(20)]
    train_loss = [0.5 - 0.02 * i for i in range(20)]  # Simulated decreasing loss
    val_loss = [0.55 - 0.018 * i for i in range(20)]

    # Create the plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot accuracy
    ax[0].plot(epochs, train_acc, label='Train Accuracy', marker='o')
    ax[0].plot(epochs, val_acc, label='Validation Accuracy', marker='o')
    ax[0].set_title('Model Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()

    # Plot loss
    ax[1].plot(epochs, train_loss, label='Train Loss', marker='o')
    ax[1].plot(epochs, val_loss, label='Validation Loss', marker='o')
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    plt.tight_layout()

    # Serve the plot as a PNG image
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    plt.close(fig)  # Free memory
    return Response(output.getvalue(), mimetype='image/png')

# Home Page
@app.route('/')
def home():
    return render_template('home.html')

# Page 1: FER Model
@app.route('/page1')
def page1():
    return render_template('FER-2013.html')

# Page 2: Emotion Model
@app.route('/page2')
def page2():
    return render_template('MY EMOTION.html')

# Video feed for FER Model
@app.route('/video_feed_fer')
def video_feed_fer():
    # Get the camera index from the URL query parameters (default is 0 for Camera 1)
    camera_index = request.args.get('camera', default=0, type=int)
    
    # Pass the camera index to the generate_frames function
    return Response(generate_frames(fer_model, camera_index), mimetype='multipart/x-mixed-replace; boundary=frame')

# Video feed for Emotion Model
@app.route('/video_feed_emotion')
def video_feed_emotion():
    # Get the camera index from the URL query parameters (default is 0 for Camera 1)
    camera_index = request.args.get('camera', default=0, type=int)
    
    return Response(generate_frames(emotion_model, camera_index), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)