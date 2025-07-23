import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

# Load data
train_images, train_labels = [], []
test_images, test_labels = [], []

base_path = r'C:\Users\User\Downloads\Data'

for folder in sorted(os.listdir(base_path)):
    folder_path = os.path.join(base_path, folder)
    if not os.path.isdir(folder_path):
        continue

    for seq in sorted(os.listdir(folder_path)):
        seq_path = os.path.join(folder_path, seq)
        if not os.path.isdir(seq_path):
            continue

        for view in sorted(os.listdir(seq_path)):
            view_path = os.path.join(seq_path, view)
            if not os.path.isdir(view_path):
                continue

            for file in sorted(os.listdir(view_path)):
                if file.endswith('.png'):
                    img_path = os.path.join(view_path, file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(img, (128, 128))
                    img = img.astype('float32') / 255.0
                    img = np.expand_dims(img, axis=-1)

                    label = folder  # e.g., "113", "114", etc.
                    if seq != "nm-06":
                        train_images.append(img)
                        train_labels.append(label)
                    else:
                        test_images.append(img)
                        test_labels.append(label)

X_train = np.array(train_images)
X_test = np.array(test_images)

le = LabelEncoder()
y_train = le.fit_transform(train_labels)
y_test = le.transform(test_labels)

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Unique training labels (encoded): {np.unique(y_train)}")
print(f"Unique testing labels (encoded): {np.unique(y_test)}")

# Build model
model = Sequential([
    Input(shape=(128, 128, 1)),
    Conv2D(32, kernel_size=3, activation='relu', name='conv1'),
    MaxPooling2D(pool_size=2, name='pool1'),
    Conv2D(64, kernel_size=3, activation='relu', name='conv2'),
    MaxPooling2D(pool_size=2, name='pool2'),
    Flatten(name='flatten'),
    Dense(128, activation='relu', name='fc1'),
    Dense(len(np.unique(y_train)), activation='softmax', name='output')
])

model.summary()

# Compile & train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=10,
    batch_size=8,
    verbose=1
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f"\nTest Accuracy: {accuracy:.4f}, Test Loss: {loss:.4f}")
