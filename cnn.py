import os
import cv2
import numpy as np


# Define the path where the images are stored
image_folder = "C:\\Users\\User\\Downloads\\GaitDatasetB-silh\\GaitDatasetB-silh\\113\\113\\nm-06\\180"  # Update if needed

# Get a sorted list of images (to maintain the sequence)
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])

# Load images into a list
images = []
for file in image_files:
    img_path = os.path.join(image_folder, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
    images.append(img)

# Convert list to NumPy array
images = np.array(images)
print("Loaded images shape:", images.shape)  # Should be (num_frames, height, width)
