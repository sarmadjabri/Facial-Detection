import cv2
import numpy as np
import os
import json
import base64
import time
from google.colab.patches import cv2_imshow
from IPython.display import display, HTML
from google.colab.output import eval_js

# Path for storing face data
FACE_DATA_PATH = "faces_data.json"
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Load or initialize known face data (encodings and names)
def load_known_faces():
    if os.path.exists(FACE_DATA_PATH):
        with open(FACE_DATA_PATH, 'r') as f:
            return json.load(f)
    else:
        return {"names": [], "encodings": []}

# Save face data to file
def save_known_faces(data):
    with open(FACE_DATA_PATH, 'w') as f:
        json.dump(data, f)

# Initialize face detector
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Load previously registered face data
known_faces_data = load_known_faces()

# Confidence threshold for face recognition (we'll use histogram comparison)
CONFIDENCE_THRESHOLD = 0.6
is_registering = False  # Flag to enable face registration mode

# Function to compare histograms (simple face recognition method)
def compare_histograms(hist1, hist2):
    # Correlation method (range between 0 and 1)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# Function to register a new face (capture and store the face)
def register_face(face, name):
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_face], [0], None, [256], [0, 256])  # Create a histogram
    hist = cv2.normalize(hist, hist).flatten()  # Normalize the histogram
    
    # Save the new face encoding and name
    known_faces_data['encodings'].append(hist)
    known_faces_data['names'].append(name)

    # Save the updated data
    save_known_faces(known_faces_data)

# Set up webcam input using JavaScript in Colab
def start_webcam():
    display(HTML('''
        <script>
        const video = document.createElement('video');
        video.width = 640;
        video.height = 480;
        const canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 480;
        const context = canvas.getContext('2d');
        document.body.appendChild(video);

        const constraints = {
            video: {
                facingMode: "user"
            }
        };

        const streamPromise = navigator.mediaDevices.getUserMedia(constraints);
        streamPromise.then(function(stream) {
            video.srcObject = stream;
            video.play();
            const capture = () => {
                context.drawImage(video, 0, 0, video.width, video.height);
                const frame = canvas.toDataURL('image/jpeg');
                window.frame = frame;
                requestAnimationFrame(capture);
            };
            capture();
        }).catch(err => {
            alert("Error accessing webcam: " + err);
        });
        </script>
    '''))

# Function to get the current webcam frame (called every time we need it)
def get_frame_from_webcam():
    return eval_js('window.frame')

# Initialize the webcam
start_webcam()

# Main loop for capturing frames and detecting faces
while True:
    # Get the frame from webcam
    img_data = get_frame_from_webcam()
    
    # Convert the base64 string into an image
    img_array = np.frombuffer(base64.b64decode(img_data.split(',')[1]), dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if frame is None:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y+h, x:x+w]
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Convert the face to grayscale and calculate the histogram
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_face], [0], None, [256], [0, 256])  # Create a histogram
        hist = cv2.normalize(hist, hist).flatten()  # Normalize the histogram
        
        # Compare the detected face with known faces
        best_match_name = "Unknown"
        best_match_confidence = 0.0

        for i, known_hist in enumerate(known_faces_data['encodings']):
            correlation = compare_histograms(hist, known_hist)
            if correlation > best_match_confidence:
                best_match_confidence = correlation
                best_match_name = known_faces_data['names'][i]

        # Display the recognized name and confidence
        if best_match_confidence >= CONFIDENCE_THRESHOLD:
            cv2.putText(frame, f"{best_match_name} ({best_match_confidence*100:.1f}%)", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # If registering, capture the new face and name
        if is_registering:
            print("Registering new face...")
            name = input("Enter the name of the person: ").strip()
            if name:
                register_face(face, name)
                print(f"New face registered as {name}!")
                is_registering = False

    # Show the live video feed with faces and names in Colab
    cv2_imshow(frame)

    # Wait for a moment to simulate a continuous loop
    time.sleep(0.1)  # Adjust time as needed
