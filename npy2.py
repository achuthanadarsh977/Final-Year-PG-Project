import numpy as np
y = np.array([0, 1, 0, 2, 4, 1, ..., 6])  # total length = 76
np.save('labels.npy', y)
print("✅ labels.npy saved:", y.shape)
