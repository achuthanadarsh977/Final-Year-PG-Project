import os

# Path to extracted dataset
dataset_path = r"C:\Users\User\Desktop\Project Frame\115"

# List all subject IDs
subjects = os.listdir(dataset_path)
print(f"Subjects in the dataset: {subjects}")
