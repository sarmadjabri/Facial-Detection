import cv2
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
