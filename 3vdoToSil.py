#AVI
# import cv2
# import numpy as np

# def extract_silhouette_frames(video_path, output_path, threshold=50):
#     # Open the video file
#     cap = cv2.VideoCapture(video_path)

#     # Get video properties
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Create VideoWriter object to save the output
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

#     while cap.isOpened():
#         ret, frame = cap.read()

#         if not ret:
#             break

#         # Convert the frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply a simple threshold to create a binary image
#         _, binary_frame = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

#         # Invert the binary image to get the silhouette
#         silhouette_frame = cv2.bitwise_not(binary_frame)

#         # Write the silhouette frame to the output video
#         out.write(silhouette_frame)

#     # Release resources
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

# # Example usage
# input_video_path = r'C:\Users\Kalatmika\Desktop\Gait\fyp\Video\SampleFinal\original.mp4'
# output_video_path = r'C:\Users\Kalatmika\Desktop\Gait\fyp\Video\SampleFinal\sil.mp4'
# extract_silhouette_frames(input_video_path, output_video_path)

#MP4---------------------

import cv2

def extract_silhouette_frames(video_path, output_path, threshold=50):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object to save the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply a simple threshold to create a binary image
        _, binary_frame = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Invert the binary image to get the silhouette
        silhouette_frame = cv2.bitwise_not(binary_frame)

        # Write the silhouette frame to the output video
        out.write(silhouette_frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
input_video_path = r'C:\Users\User\Desktop\Project Frame\1Sample.mp4'
output_video_path = r'C:\Users\User\Desktop\Project Frame\3Silhouette.mp4'
extract_silhouette_frames(input_video_path, output_video_path)

