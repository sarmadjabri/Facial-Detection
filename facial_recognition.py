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
