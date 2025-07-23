import cv2
import os

# Set the path to your image folder
image_folder = r'C:\Users\User\Downloads\GaitDatasetB-silh\113\nm-06\180\113\nm-06\018'
output_video = r'C:\Users\User\Downloads\3Silhouette.mp4'

# Get all image file names and sort them
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])

# Read the first image to get the dimensions
first_image = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = first_image.shape

# Define the video writer with the desired output file and codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
video_writer = cv2.VideoWriter(output_video, fourcc, 30, (width, height))  # 30 FPS

# Loop through all the images and add them to the video
for image in images:
    img = cv2.imread(os.path.join(image_folder, image))
    video_writer.write(img)

# Release the video writer object
video_writer.release()

print("Video created successfully!")

cap = cv2.VideoCapture(output_video)


# Check if the video was opened correctly
if not cap.isOpened():
    print("Error: Couldn't open the video.")
    exit()

# Display the video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the current frame
    cv2.imshow('Video', frame)

    # Wait for key press, exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()