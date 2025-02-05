import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deepface import DeepFace
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Paths
db_path = "dataset"  # Directory containing known faces
test_path = "test_data"  # Directory containing test images

# Initialize variables
true_labels = []
predicted_labels_deepface = []

def predict_face_deepface(img_path):
    """
    Predicts a face using DeepFace with optimized settings.
    Returns the recognized name or 'Unknown'.
    """
    try:
        result = DeepFace.find(img_path=img_path, db_path=db_path, model_name="Facenet", enforce_detection=False, distance_metric="cosine")

        if len(result) > 0 and result[0]["distance"][0] < 0.4:  # Adjusted threshold for better accuracy
            recognized_name = result[0]["identity"][0].split(os.sep)[-2]
            return recognized_name
        else:
            return "Unknown"
    except Exception as e:
        print(f"DeepFace Prediction Error: {e}")
        return "Unknown"

# Load Test Data and Evaluate
print("Starting Evaluation...")

for person_name in os.listdir(test_path):  # Iterate through test dataset
    person_folder = os.path.join(test_path, person_name)
    
    if os.path.isdir(person_folder):
        for img_name in os.listdir(person_folder):  # Iterate through images
            img_path = os.path.join(person_folder, img_name)

            # Predict using DeepFace
            predicted_name_deepface = predict_face_deepface(img_path)

            # Store predictions and true labels
            true_labels.append(person_name)
            predicted_labels_deepface.append(predicted_name_deepface)

            print(f"Image: {img_name}, True: {person_name}, DeepFace: {predicted_name_deepface}")

# Generate Confusion Matrix and Performance Metrics
print("\nEvaluation Results:")

# Unique labels including 'Unknown'
labels = list(set(true_labels)) + ["Unknown"]

# DeepFace Evaluation
print("\nConfusion Matrix (DeepFace):")
cm_deepface = confusion_matrix(true_labels, predicted_labels_deepface, labels=labels)
print(cm_deepface)

print("\nClassification Report (DeepFace):")
print(classification_report(true_labels, predicted_labels_deepface, labels=labels))

accuracy_deepface = accuracy_score(true_labels, predicted_labels_deepface)
print(f"Overall Accuracy (DeepFace): {accuracy_deepface:.2f}")

# Visualization
plt.figure(figsize=(10, 5))

# Confusion Matrix Heatmap
sns.heatmap(cm_deepface, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix (DeepFace)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Accuracy Comparison Bar Graph
plt.figure(figsize=(6, 4))
plt.bar(["DeepFace"], [accuracy_deepface], color=["blue"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

# Cumulative Accuracy for DeepFace Model
cumulative_accuracies = []
correct_predictions = 0

for i in range(len(true_labels)):
    if true_labels[i] == predicted_labels_deepface[i]:
        correct_predictions += 1
    cumulative_accuracies.append(correct_predictions / (i + 1))

# Plot the Line Graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(true_labels) + 1), cumulative_accuracies, label="DeepFace Cumulative Accuracy", color="blue", linewidth=2)
plt.axhline(y=accuracy_deepface, color="red", linestyle="--", label=f"Final Accuracy: {accuracy_deepface:.2f}")
plt.title("DeepFace Model Performance Over Test Set")
plt.xlabel("Number of Test Samples")
plt.ylabel("Cumulative Accuracy")
plt.legend()
plt.grid(True)
plt.show()
