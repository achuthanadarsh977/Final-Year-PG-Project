import cv2
import glob

# List of video files to be merged
video_files = [r"C:\Users\User\Downloads\3Silhouette.mp4", 
               r"C:\Users\User\Downloads\4Silhouette.mp4"]  # Add all your videos here

# Read the first video to get dimensions
cap = cv2.VideoCapture(video_files[0])
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
cap.release()

# Define output video
output_video = r"C:\Users\User\Downloads\CombinedVideo.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# Loop through each video and add frames to the final video
for video in video_files:
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        video_writer.write(frame)
    cap.release()

video_writer.release()
print("Videos merged successfully!")
