#KAMA DEUNI

import numpy as np
import cv2

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# create a background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

cv2.startWindowThread()

# open your video file
video_path = r'C:\Users\User\Desktop\Project Frame\3Silhouette.mp4'
cap = cv2.VideoCapture(video_path)

# specify the output file
output_path = r'C:\Users\User\Desktop\Project Frame\1Sample.mp4'
out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640, 480),
    isColor=False)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))
    
    # apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the red color
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                      (0, 0, 255), 2)

    # Apply the foreground mask to obtain the silhouette
    silhouette = cv2.bitwise_and(frame, frame, mask=fg_mask)

    # Write the silhouette video
    out.write(silhouette)
    # Display the resulting frame
    cv2.imshow('frame', silhouette)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
# and release the output
out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)
