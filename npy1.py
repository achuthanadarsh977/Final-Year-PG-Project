import os
import numpy as np
from PIL import Image

image_dir = r'C:\Users\User\Desktop\Project Frame\gei'
image_size = (128, 128)

X = []

for filename in sorted(os.listdir(image_dir)):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        img = Image.open(os.path.join(image_dir, filename)).convert('L')  # Convert to grayscale
        img = img.resize(image_size)
        img_array = np.array(img)
        img_array = img_array.reshape(128, 128, 1)  # Add channel dimension
        X.append(img_array)

X = np.array(X)
np.save('data.npy', X)
print("✅ data.npy saved:", X.shape)
