import cv2
import numpy as np
import os
import glob
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# Load Pretrained CNN (Feature Extractor)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
feature_extractor = Model(inputs=base_model.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.output))

# Function to extract CNN features
def extract_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))  # Resize to match CNN input
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert color format
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    features = feature_extractor.predict(img)
    return features.flatten()

# Function to compute Euclidean Distance
def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

dataset_path = r"C:\Users\User\Desktop\Project Frame\118"
subjects = ["nm-06", "nm-01", "nm-02", "nm-03", "nm-04", "nm-05"]
sequences = ["000","018","036","054","072","090","108","126","144","162","180"]

# Compare images using CNN-based features
for sequence in sequences:
    print(f"\n🔹 Comparing Sequence: {sequence}")

    # Collect feature vectors
    subject_features = {}
    for subject in subjects:
        seq_path = os.path.join(dataset_path, subject, sequence)
        if os.path.exists(seq_path):
            images = sorted(glob.glob(os.path.join(seq_path, "*.png")))
            if images:
                subject_features[subject] = extract_features(images[0])  # Extract CNN features from first image

    # Compare features using Euclidean distance
    for subj1 in subject_features:
        for subj2 in subject_features:
            if subj1 != subj2:
                distance = euclidean_distance(subject_features[subj1], subject_features[subj2])
                print(f"Distance between {subj1} and {subj2} in {sequence}: {distance:.3f}")

                # Define a threshold (Lower distance = More similar)
                threshold = 10.0
                if distance < threshold:
                    print(f"✅ {subj1} and {subj2} likely belong to the SAME person.")
                else:
                    print(f"❌ {subj1} and {subj2} might be DIFFERENT people.")
