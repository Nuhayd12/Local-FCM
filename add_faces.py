import os
import cv2
import pickle
import numpy as np
from deepface import DeepFace

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Create dataset folder if not exists
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Get user input
name = input("Enter Your Name: ")
person_dir = f"dataset/{name}"
if not os.path.exists(person_dir):
    os.makedirs(person_dir)

i = 0 # Counter for frame capture
face_embeddings = []
face_labels = []

def adjust_brightness(img):
    """
    Check and normalize brightness using CLAHE.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def extract_embedding(image_path):
    """
    Extract embedding using DeepFace with multiple models.
    """
    try:
        models = ["Facenet", "VGG-Face", "OpenFace"]
        embeddings = []
        for model in models:
            embedding = DeepFace.represent(img_path=image_path, model_name=model)[0]["embedding"]
            embeddings.append(np.array(embedding))
        return np.mean(embeddings, axis=0)  # Averaging across models
    except Exception as e:
        print(f"Embedding Error: {e}")
        return None

while i < 100:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_img = adjust_brightness(face_img)
        face_img = cv2.resize(face_img, (160, 160))  # Standard DeepFace input size

        if i % 10 == 0:
            face_path = os.path.join(person_dir, f"face_{i//10}.jpg")
            cv2.imwrite(face_path, face_img)
            print(f"Saved {face_path}")

            embedding = extract_embedding(face_path)
            if embedding is not None:
                face_embeddings.append(embedding)
                face_labels.append(name)

        i += 1
        cv2.putText(frame, f"Collecting: {i}/100", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Data Collection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Data collection completed!")
video.release()
cv2.destroyAllWindows()

# Save embeddings
print("Saving DeepFace embeddings...")
if not os.path.exists("data"):
    os.makedirs("data")

with open("data/face_embeddings.pkl", "wb") as f:
    pickle.dump(face_embeddings, f)

with open("data/face_labels.pkl", "wb") as f:
    pickle.dump(face_labels, f)

print("Embeddings saved!")
