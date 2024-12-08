import cv2
import numpy as np
import base64
from google.colab.patches import cv2_imshow
from IPython.display import display, HTML
from google.colab.output import eval_js
import time
import requests

# Path for Haar cascade classifier
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

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

# Function for reverse image search using TinEye
def search_image_online(image_path):
    url = "https://api.tineye.com/rest/search/"
    api_key = "your_tineye_api_key"  # Replace with your actual TinEye API key
    api_secret = "your_tineye_api_secret"  # Replace with your actual TinEye API secret

    with open(image_path, 'rb') as image_file:
        files = {'image': image_file}
        data = {'api_key': api_key, 'api_secret': api_secret}
        
        response = requests.post(url, files=files, data=data)
        return response.json()

# Start webcam capture
start_webcam()

# Retry mechanism to ensure we get a valid frame
MAX_RETRIES = 10
frame_attempts = 0

# Main loop for capturing frames and detecting faces
while True:
    img_data = None
    while img_data is None and frame_attempts < MAX_RETRIES:
        img_data = get_frame_from_webcam()
        frame_attempts += 1
        time.sleep(0.1)  # Give it a moment to capture a frame

    # If we couldn't get a frame after several attempts, exit the loop
    if img_data is None:
        print("Unable to capture frame from webcam.")
        break

    # Convert the base64 string into an image
    try:
        img_array = np.frombuffer(base64.b64decode(img_data.split(',')[1]), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decoding image: {e}")
        break

    if frame is None:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1,     # Increase to improve accuracy
        minNeighbors=5,      # Increase to avoid detecting smaller objects (e.g., eyes)
        minSize=(80, 80)     # Make sure the face is large enough to be detected
    )

    # Debugging: Log the number of faces detected
    print(f"Detected faces: {len(faces)}")

    # Loop through detected faces and draw bounding boxes
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face (RED)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Crop the face from the image
            face_img = frame[y:y + h, x:x + w]

            # Save the face as a separate image (for use in a web search)
            cv2.imwrite('detected_face.jpg', face_img)

            # Perform web search
            search_results = search_image_online('detected_face.jpg')
            if search_results.get("results"):
                for result in search_results["results"]:
                    print(f"Match found at: {result['domain']}")
                    print(f"URL: {result['url']}")
            else:
                print("No matches found.")
    else:
        print("No faces detected.")

    # Show the live video feed with faces in Colab
    cv2_imshow(frame)

    # Wait for a moment to simulate a continuous loop
    time.sleep(0.1)  # Adjust time as needed
