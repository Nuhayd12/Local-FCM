import cv2
import os
import numpy as np
import pickle
from deepface import DeepFace

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Load embeddings
with open("data/face_embeddings.pkl", "rb") as f:
    face_embeddings = pickle.load(f)

with open("data/face_labels.pkl", "rb") as f:
    face_labels = pickle.load(f)

db_path = "dataset"
print("Initializing Face Recognition...")

def adjust_brightness(img):
    """
    Normalize brightness using CLAHE.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_img = adjust_brightness(face_img)
        temp_path = "temp_face.jpg"
        cv2.imwrite(temp_path, face_img)

        try:
            models = ["Facenet", "VGG-Face", "OpenFace"]
            scores = {}

            for model in models:
                result = DeepFace.find(img_path=temp_path, db_path=db_path, model_name=model, enforce_detection=False)
                if len(result) > 0:
                    for idx, match in enumerate(result[0]["identity"]):
                        recognized_name = match.split(os.sep)[-2]
                        confidence = round(1 - result[0]["distance"][idx], 2)

                        if recognized_name in scores:
                            scores[recognized_name].append(confidence)
                        else:
                            scores[recognized_name] = [confidence]

            # Compute average confidence per identity
            final_recognition = None
            max_confidence = 0.0
            for name, confs in scores.items():
                avg_conf = np.mean(confs)
                if avg_conf > max_confidence and avg_conf > 0.55:  # Confidence threshold
                    max_confidence = avg_conf
                    final_recognition = name

            display_text = f"{final_recognition} ({max_confidence*100:.1f}%)" if final_recognition else "Unknown"

        except Exception as e:
            print(f"Error: {e}")
            display_text = "Error"

        # Display results
        cv2.putText(frame, display_text, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Face recognition stopped.")
video.release()
cv2.destroyAllWindows()
