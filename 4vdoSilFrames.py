import cv2
import os

input_video_path = r"C:\Users\User\Downloads\1Sample.mp4"  # Replace with the actual file name
output_silhouette_path = r"C:\Users\User\Downloads\3Silhouette.mp4"  # Replace with the path to the silhouette video
output_frames_folder = r''  # Replace with the path to the output frames folder

# Open the silhouette video file
cap_silhouette = cv2.VideoCapture(output_silhouette_path)

# Create the output frames folder if it doesn't exist
os.makedirs(output_frames_folder, exist_ok=True)

frame_count = 0

while cap_silhouette.isOpened():
    ret, silhouette_frame = cap_silhouette.read()

    if not ret:
        break

    # Save the silhouette frame as an image file
    frame_count += 1
    frame_filename = f"frame_{frame_count:04d}.png"
    frame_path = os.path.join(output_frames_folder, frame_filename)
    cv2.imwrite(frame_path, silhouette_frame)

# Release resources
cap_silhouette.release()
cv2.destroyAllWindows()
