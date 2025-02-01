import cv2
from deepface import DeepFace

def detect_and_describe_faces(image_path):
    # Load the Haar cascade file
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    descriptions = []
    
    # Process each detected face
    for i, (x, y, w, h) in enumerate(faces):
        face = img[y:y+h, x:x+w]  # Crop the face

        # Analyze face using DeepFace
        try:
            analysis = DeepFace.analyze(face, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)[0]
            
            # Construct a "Guess Who" style description
            description = f"Person {i+1}: Appears to be around {analysis['age']} years old, {analysis['dominant_gender']}. "
            description += f"They have {analysis['dominant_race']} skin tone and seem to be feeling {analysis['dominant_emotion']}."
            descriptions.append(description)
            print(descriptions)

            # Draw rectangle and label
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, f"Person {i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        except Exception as e:
            print(f"Error processing face {i+1}: {e}")

    # Display the output
    cv2.imshow('Face Detection & Description', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print descriptions
    for desc in descriptions:
        print(desc)

# Example usage
detect_and_describe_faces('everyone2.jpg')