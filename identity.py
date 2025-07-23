import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical

# Path to dataset
DATA_PATH = r'C:\Users\User\Downloads\Data1'

# Subjects and sequences
SUBJECTS = ['113', '114', '115', '116', '117', '118', '119']
SEQUENCES_TRAIN = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05']
SEQUENCE_TEST = 'nm-06'
ANGLES = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']

# Load images and labels
def load_data(data_path, subjects, sequences):
    images, labels = [], []
    for label in subjects:
        for seq in sequences:
            for angle in ANGLES:
                folder = os.path.join(data_path, f'{label}', seq, angle)
                if os.path.exists(folder):
                    for file in sorted(os.listdir(folder)):
                        img_path = os.path.join(folder, file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (128, 128))  # Normalize size
                            img = img.astype('float32') / 255.0
                            images.append(img)
                            labels.append(label)
    images = np.expand_dims(np.array(images), axis=-1)
    return images, np.array(labels)

# Load training data
train_images, train_labels = load_data(DATA_PATH, SUBJECTS, SEQUENCES_TRAIN)

# Label encoding
le = LabelEncoder()
train_labels_encoded = le.fit_transform(train_labels)
train_labels_cat = to_categorical(train_labels_encoded)

# Build model
model = Sequential([
    Input(shape=(128, 128, 1)),
    Conv2D(32, (3, 3), activation='relu', name='conv1'),
    MaxPooling2D((2, 2), name='pool1'),
    Conv2D(64, (3, 3), activation='relu', name='conv2'),
    MaxPooling2D((2, 2), name='pool2'),
    Flatten(name='flatten'),
    Dense(128, activation='relu', name='fc1'),
    Dense(len(SUBJECTS), activation='softmax', name='output')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_images, train_labels_cat, epochs=10, batch_size=64)

# Predict on test data
test_images, test_labels = load_data(DATA_PATH, SUBJECTS, [SEQUENCE_TEST])
test_labels_encoded = le.transform(test_labels)

predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
confidences = np.max(predictions, axis=1)

# Print result like: Image 1: True = 119, Predicted = 119, Confidence = 0.98
for idx, (true, pred, conf) in enumerate(zip(test_labels_encoded, predicted_classes, confidences)):
    true_label = le.inverse_transform([true])[0]
    pred_label = le.inverse_transform([pred])[0]
    print(f"Image {idx+1}: True = {true_label}, Predicted = {pred_label}, Confidence = {conf:.2f}")
