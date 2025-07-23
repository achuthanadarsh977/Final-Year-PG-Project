
import os 
import numpy as np
import cv2
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

IMG_HEIGHT, IMG_WIDTH = 128, 128
data_dir = r"C:\Users\User\Downloads\Data1"

train_data, train_labels = [], []
test_data, test_labels = [], []

# Define training and testing folders
train_folders = ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05']
test_folder = 'nm-06'

subject_folders = sorted(os.listdir(data_dir))

for subject in subject_folders:
    subject_path = os.path.join(data_dir, subject)
    if not os.path.isdir(subject_path):
        continue

    for nm in os.listdir(subject_path):
        nm_path = os.path.join(subject_path, nm)
        if not os.path.isdir(nm_path):
            continue

        # Determine if this is training or testing
        if nm in train_folders:
            label = subject
            for angle in os.listdir(nm_path):
                angle_path = os.path.join(nm_path, angle)
                for fname in os.listdir(angle_path):
                    if fname.lower().endswith((".png", ".jpg")):
                        img_path = os.path.join(angle_path, fname)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                            img = img.astype(np.float32) / 255.0
                            img = np.expand_dims(img, axis=-1)
                            train_data.append(img)
                            train_labels.append(label)

        elif nm == test_folder:
            label = subject
            for angle in os.listdir(nm_path):
                angle_path = os.path.join(nm_path, angle)
                for fname in os.listdir(angle_path):
                    if fname.lower().endswith((".png", ".jpg")):
                        img_path = os.path.join(angle_path, fname)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                            img = img.astype(np.float32) / 255.0
                            img = np.expand_dims(img, axis=-1)
                            test_data.append(img)
                            test_labels.append(label)




# Convert to numpy arrays
X_train = np.array(train_data)
X_test = np.array(test_data)

# Label encoding
le = LabelEncoder()
y_train_encoded = le.fit_transform(train_labels)
y_test_encoded = le.transform(test_labels)

# Convert to one-hot
y_train_cat = to_categorical(y_train_encoded)
y_test_cat = to_categorical(y_test_encoded)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("Unique training labels (encoded):", np.unique(y_train_encoded))
print("Unique testing labels (encoded):", np.unique(y_test_encoded))

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), name='conv1'),
    layers.MaxPooling2D((2, 2), name='pool1'),
    layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
    layers.MaxPooling2D((2, 2), name='pool2'),
    layers.Flatten(),
    layers.Dense(128, activation='relu', name='fc1'),
    layers.Dense(y_train_cat.shape[1], activation='softmax', name='output')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
model.fit(X_train, y_train_cat, epochs=10, batch_size=8, validation_split=0.1)

# Evaluate model
loss, acc = model.evaluate(X_test, y_test_cat)
print(f"\nTest Accuracy: {acc:.4f}, Test Loss: {loss:.4f}")