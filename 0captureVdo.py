
import cv2

def capture_video(output_file, duration=10):
    # Open the camera
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (640, 480))

    # Capture video for the specified duration
    start_time = cv2.getTickCount()
    while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < duration:
        ret, frame = cap.read()  # Read a frame from the camera

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Write the frame to the output video file
        out.write(frame)

        # Display the frame (optional)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    output_file = r''  # Change this to your desired output file name
    duration = 15  # Duration of the captured video in seconds
    capture_video(output_file, duration)
