import cv2
import numpy as np
import base64
from google.colab.patches import cv2_imshow
from IPython.display import display, HTML
from google.colab.output import eval_js
import time
import requests
from googleapiclient.discovery import build

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

# Function to upload image to Imgur and get the URL
def upload_to_imgur(image_path, client_id):
    headers = {'Authorization': f'Client-ID {client_id}'}
    with open(image_path, 'rb') as image_file:
        data = {'image': image_file.read()}
        response = requests.post("https://api.imgur.com/3/upload", headers=headers, files=data)
        response_json = response.json()
        if response.status_code == 200:
            return response_json['data']['link']  # Return image URL
        else:
            print("Imgur upload failed:", response_json)
            return None

# Function to search for the image using Google Custom Search API
def search_image_online(image_url, api_key, cse_id):
    service = build("customsearch", "v1", developerKey=api_key)
    
    # Perform the search query using the image URL
    try:
        res = service.cse().list(
            q="face",   # This could be anything related to the image context
            cx=cse_id,
            searchType='image',
            imgUrl=image_url  # Search by image URL
        ).execute()
        
        if 'items' in res:
            return res['items']
        else:
            return None

    except Exception as e:
        print(f"Error during search: {e}")
        return None

# Start webcam capture
start_webcam()

# Retry mechanism to ensure we get a valid frame
MAX_RETRIES = 10
frame_attempts = 0

# Replace with your own Google Custom Search API key and CSE ID
API_KEY = 'YOUR_GOOGLE_API_KEY'
CSE_ID = 'YOUR_CUSTOM_SEARCH_ENGINE_ID'
IMGUR_CLIENT_ID = 'YOUR_IMGUR_CLIENT_ID'  # Get your Imgur Client ID from Imgur API

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

            # Upload the image to Imgur
            image_url = upload_to_imgur('detected_face.jpg', IMGUR_CLIENT_ID)

            if image_url:
                print(f"Uploaded image to: {image_url}")

                # Perform web search using Google Custom Search API
                search_results = search_image_online(image_url, API_KEY, CSE_ID)
                
                if search_results:
                    for result in search_results:
                        print(f"Match found at: {result['displayLink']}")
                        print(f"Title: {result['title']}")
                        print(f"URL: {result['link']}")
                else:
                    print("No matches found.")
    else:
        print("No faces detected.")

    # Show the live video feed with faces in Colab
    cv2_imshow(frame)

    # Wait for a moment to simulate a continuous loop
    time.sleep(0.1)  # Adjust time as needed
