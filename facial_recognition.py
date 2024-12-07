import cv2
import face_recognition
import json
import os
import threading
import time

# Load or initialize known face encodings and names
def load_known_faces():
    if os.path.exists('faces_data.json'):
        with open('faces_data.json', 'r') as f:
            data = json.load(f)
            return data['encodings'], data['names']
    else:
        return [], []
aimport cv2
import numpy as np
import os
import json

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

# Initialize webcam
cap = cv2.VideoCapture(0)

# Confidence threshold for face recognition (we'll use histogram comparison)
CONFIDENCE_THRESHOLD = 0.6
is_registering = False  # Flag to enable face registration mode

# Function to compare histograms (simple face recognition method)
def compare_histograms(hist1, hist2):
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

# Main loop for capturing frames and detecting faces
while True:
    ret, frame = cap.read()
    if not ret:
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
        cv2.putText(frame, f"{best_match_name} ({best_match_confidence*100:.1f}%)", 
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # If the confidence is above the threshold, recognize the face
        if best_match_confidence < CONFIDENCE_THRESHOLD:
            best_match_name = "Unknown"
        
        # If registering, capture the new face and name
        if is_registering:
            print("Registering new face...")
            name = input("Enter the name of the person: ").strip()
            if name:
                register_face(face, name)
                print(f"New face registered as {name}!")
                is_registering = False

    # Show the live video feed with faces and names
    cv2.imshow("Face Recognition", frame)

    # Key actions
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit on 'q'
        break
    elif key == ord('r'):  # Register a new face on 'r'
        is_registering = True
        print("Please look at the camera to register a new face.")

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Save face encodings and names to a file
def save_known_faces(encodings, names):
    data = {'encodings': encodings, 'names': names}
    with open('faces_data.json', 'w') as f:
        json.dump(data, f)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load known faces from file (if any)
known_face_encodings, known_face_names = load_known_faces()

# Performance parameters
frame_skip = 5  # Process every 5th frame to save processing time
frame_count = 0

# Confidence threshold for recognition
CONFIDENCE_THRESHOLD = 0.6

# Set a flag to allow face registration when the user presses 'r'
is_registering = False

# Lock for thread safety
frame_lock = threading.Lock()
frame = None

# Function for frame capture in a separate thread
def capture_frame():
    global frame
    while True:
        ret, current_frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue
        with frame_lock:
            frame = current_frame
        time.sleep(0.01)  # Sleep to avoid high CPU usage in the capture thread

# Start capture thread
capture_thread = threading.Thread(target=capture_frame)
capture_thread.daemon = True
capture_thread.start()

while True:
    # Ensure we have a valid frame before proceeding
    if frame is not None and frame_count % frame_skip == 0:
        with frame_lock:
            current_frame = frame.copy()  # Work with a copy of the frame to avoid race conditions

        # Resize frame for faster processing
        small_frame = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR (OpenCV) to RGB (face_recognition)
        rgb_frame = small_frame[:, :, ::-1]

        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            name = "Unknown"
            confidence = 1.0

            if True in matches:
                first_match_index = matches.index(True)
                confidence = 1.0 - distances[first_match_index]
                if confidence >= CONFIDENCE_THRESHOLD:
                    name = known_face_names[first_match_index]
                else:
                    name = "Unknown"

            # Draw rectangle around the face and display the name
            cv2.rectangle(current_frame, (left * 4, top * 4), (right * 4, bottom * 4), (0, 255, 0), 2)
            cv2.putText(current_frame, f"{name} ({confidence*100:.1f}%)", (left * 4, top * 4 - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

        # Handle face registration
        if is_registering:
            # Only register faces when one is detected
            if face_locations:
                print("Face detected! Now registering...")
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # Register the new face
                    known_face_encodings.append(face_encoding)
                    name = input("Enter the name of the new person: ").strip()
                    if name == "":  # Ensure the name is not empty
                        print("Name cannot be empty. Please try again.")
                        continue
                    known_face_names.append(name)

                    # Provide feedback on successful registration
                    print(f"New face registered as {name}!")

                    # Save to the file so that new faces are persistent
                    save_known_faces(known_face_encodings, known_face_names)
                    is_registering = False
                    print("Registration complete. Press 'r' to register another face or 'q' to quit.")
                # Visual feedback on screen
                cv2.putText(current_frame, "Face successfully registered!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the video feed
        cv2.imshow('Face Recognition', current_frame)

    frame_count += 1

    # Key actions
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') and not is_registering:  # Register a new face when 'r' is pressed
        print("Registering a new face. Please look at the camera.")
        is_registering = True

# Release the webcam and close OpenCV window
cap.release()
cv2.destroyAllWindows()
