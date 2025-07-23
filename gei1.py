import cv2
import numpy as np
import os
import glob

# Define dataset path
dataset_path = r"C:\Users\User\Desktop\Project Frame\113"
output_gei_path = r"C:\Users\User\Desktop\Project Frame\gei"
os.makedirs(output_gei_path, exist_ok=True)

# Function to compute GEI for a sequence
def compute_gei(image_folder, output_path):
    images = sorted(glob.glob(os.path.join(image_folder, "*.png")))

    if not images:
        print(f"No images found in {image_folder}")
        return None

    # Read the first image to get dimensions
    first_image = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
    h, w = first_image.shape
    gei = np.zeros((h, w), dtype=np.float32)

    # Sum all silhouette images
    for img_path in images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0  # Normalize
        gei += img

    # Compute average (Gait Energy Image)
    gei /= len(images)

    # Save GEI as an image
    gei_output_file = os.path.join(output_path, os.path.basename(image_folder) + "_gei.png")
    cv2.imwrite(gei_output_file, (gei * 255).astype(np.uint8))

    return gei_output_file

# Iterate through all subjects and sequences
subjects = os.listdir(dataset_path)

for subject in subjects:
    subject_path = os.path.join(dataset_path, subject)
    if os.path.isdir(subject_path):
        sequences = os.listdir(subject_path)

        for sequence in sequences:
            sequence_path = os.path.join(subject_path, sequence)

            # Compute GEI for each sequence
            gei_image = compute_gei(sequence_path, output_gei_path)
            if gei_image:
                print(f"GEI created for {subject} - {sequence}: {gei_image}")
