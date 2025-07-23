import numpy as np
import cv2

# Initialize lists
train_data = ["nm-01","nm-02","nm-03","nm-04","nm-05"]
test_data = ["nm-06"]
labels = ["000","018","036","054","072","090","108","126","144","162","180"]

# Define dummy subject IDs
subject_ids = ["000","018","036","054","072","090","108","126","144","162","180"]  # Example subjects

# Image dimensions (for resizing)
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Dummy function to create a fake GEI image
def generate_fake_gei():
    return np.random.rand(IMG_HEIGHT, IMG_WIDTH)  # Random GEI image

# Loop over subjects and store data
for _ in subject_ids:
    gei_train = generate_fake_gei()  # Generate a dummy GEI for training
    gei_test = generate_fake_gei()  # Generate a dummy GEI for testing

    train_data.append(gei_train)  # Add GEI to training set
    test_data.append(gei_test)  # Add GEI to testing set

# Convert lists to NumPy arrays
train_data = np.array(train_data)  # Reshape for CNN input
test_data = np.array(test_data)

# Print dataset sizes
print(f"Training samples: {train_data.shape[0]}")
print(f"Testing samples: {test_data.shape[0]}")
