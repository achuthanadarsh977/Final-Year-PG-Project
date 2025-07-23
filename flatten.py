import tarfile
import os
import glob

# List of uploaded dataset files
dataset_files = [
    r"C:\Users\User\Downloads\GaitDatasetB-silh\GaitDatasetB-silh\113.tar.gz", 
     r"C:\Users\User\Downloads\GaitDatasetB-silh\GaitDatasetB-silh\114.tar.gz",  r"C:\Users\User\Downloads\GaitDatasetB-silh\GaitDatasetB-silh\115.tar.gz"
     ,
    r"C:\Users\User\Downloads\GaitDatasetB-silh\GaitDatasetB-silh\116.tar.gz" ,  r"C:\Users\User\Downloads\GaitDatasetB-silh\GaitDatasetB-silh\117.tar.gz",  r"C:\Users\User\Downloads\GaitDatasetB-silh\GaitDatasetB-silh\118.tar.gz",
     r"C:\Users\User\Downloads\GaitDatasetB-silh\GaitDatasetB-silh\119.tar.gz"
]

# Extract location
extract_path = r"C:\Users\User\Desktop\Project Frame"

# Extract all tar.gz files
os.makedirs(extract_path, exist_ok=True)

for file in dataset_files:
    with tarfile.open(file, "r:gz") as tar:
        tar.extractall(extract_path)

print(f"All datasets extracted to {extract_path}")

# List all subject IDs
subjects = os.listdir(extract_path)
for person in subjects:
    person_path = os.path.join(extract_path, person)
    if os.path.isdir(person_path):
        sequences = os.listdir(person_path)
        print(f"Person {person} has sequences: {sequences}")
