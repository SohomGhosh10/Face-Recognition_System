import cv2
import numpy as np

def preprocess_image(image_path):
    # Load and preprocess the reference image for face recognition
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from '{image_path}'")
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        print(f"No faces detected in '{image_path}'")
        return None
    
    (x, y, w, h) = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    face_roi_resized = cv2.resize(face_roi, (100, 100))  # Resize to a fixed size
    
    return face_roi_resized  # Return the resized face ROI

def recognize_faces(video_capture, reference_image_path):
    # Load the reference image for comparison
    reference_face = preprocess_image(reference_image_path)
    if reference_face is None:
        print("Error: Unable to process reference image")
        return
    
    # Resize the reference face for consistency
    reference_face_resized = cv2.resize(reference_face, (100, 100))
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Unable to capture frame")
            break
        
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Process each detected face
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi_resized = cv2.resize(face_roi, (100, 100))  # Resize to match reference image
            
            # Compare the detected face with the reference face
            if np.array_equal(face_roi_resized, reference_face_resized):
                match_status = "Matching"
            else:
                match_status = "Not Matching"
            
            # Draw rectangle around the detected face and display match status
            color = (0, 255, 0) if match_status == "Matching" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, match_status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display the frame
        cv2.imshow('Face Recognition', frame)
        
        # Check for exit key (press 'q' to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release video capture and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

def main(reference_image_path):
    # Open the webcam
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Unable to access the webcam")
        return
    
    # Perform real-time face recognition using webcam
    recognize_faces(video_capture, reference_image_path)

if __name__ == '__main__':
    # Provide the path to the new reference image for face matching
    reference_image_path = r'C:\Users\sohom\OneDrive\Pictures\Sohom1.jpg'
    main(reference_image_path)
