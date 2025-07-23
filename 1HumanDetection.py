
# import the necessary packages
import numpy as np
import cv2

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# open webcam video stream
video_path = r'C:\Users\User\Downloads\1Sample.mp4'

cap = cv2.VideoCapture(video_path)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Couldn't open video file.")
    exit()

# the output will be written to output.avi
output_path = r'C:\Users\User\Downloads\3Silhouette.mp4'
out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.0,
    (640, 480))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame is not successfully read, break out of the loop
    if not ret:
        print("Error: Couldn't read frame from video.")
        break

    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))
    
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8))

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
    
    # Write the output video
    out.write(frame.astype('uint8'))

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and writer
cap.release()
out.release()

# Close the window
cv2.destroyAllWindows()
cv2.waitKey(1)

# import numpy as np
# import cv2

# # initialize the HOG descriptor/person detector
# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# cv2.startWindowThread()

# # open your video file
# video_path = r'C:\Users\Kalatmika\Desktop\Gait\fyp\Video\SampleGreenScreen\Original\manwalking.avi'
# cap = cv2.VideoCapture(video_path)

# # specify the output file
# output_path = r'C:\Users\Kalatmika\Desktop\Gait\fyp\Video\SampleGreenScreen\ObjDetection\od2.avi'
# out = cv2.VideoWriter(
#     output_path,
#     cv2.VideoWriter_fourcc(*'MJPG'),
#     15.,
#     (640, 480))

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     if not ret:
#         break

#     # resizing for faster detection
#     frame = cv2.resize(frame, (640, 480))
#     # using a greyscale picture, also for faster detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # detect people in the image
#     # returns the bounding boxes for the detected objects
#     boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

#     boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

#     for (xA, yA, xB, yB) in boxes:
#         # display the detected boxes in the color picture
#         cv2.rectangle(frame, (xA, yA), (xB, yB),
#                       (255, 0, 0), 2)

#     # Write the output video
#     out.write(frame.astype('uint8'))

#     # turn to greyscale:
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     # apply threshold. all pixels with a level larger than 80 are shown in white. the others are shown in black:
#     ret,frame = cv2.threshold(frame,80,255,cv2.THRESH_BINARY)

#     # Display the resulting frame
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything is done, release the capture
# cap.release()
# # and release the output
# out.release()
# # finally, close the window
# cv2.destroyAllWindows()
# cv2.waitKey(1)



# # # import the necessary packages
# # import numpy as np
# # import cv2
 
# # # initialize the HOG descriptor/person detector
# # hog = cv2.HOGDescriptor()
# # hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# # cv2.startWindowThread()

# # # open your video file
# # video_path = r'C:\Users\Kalatmika\Desktop\Gait\fyp\Video\SampleFinal\original.mp4'
# # cap = cv2.VideoCapture(video_path)

# # # the output
# # output_path = r'C:\Users\Kalatmika\Desktop\Gait\fyp\Video\SampleFinal\od2.mp4'
# # out = cv2.VideoWriter(
# #     output_path,
# #     cv2.VideoWriter_fourcc(*'MJPG'),
# #     15.,
# #     (640,480))

# # while(True):
# #     # Capture frame-by-frame
# #     ret, frame = cap.read()

# #     # resizing for faster detection
# #     frame = cv2.resize(frame, (640, 480))
# #     # using a greyscale picture, also for faster detection
# #     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

# #     # detect people in the image
# #     # returns the bounding boxes for the detected objects
# #     boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )

# #     boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

# #     for (xA, yA, xB, yB) in boxes:
# #         # display the detected boxes in the colour picture
# #         cv2.rectangle(frame, (xA, yA), (xB, yB),
# #                           (255, 0, 0), 2)
    
# #     # Write the output video 
# #     out.write(frame.astype('uint8'))
# #     # Display the resulting frame
# #     cv2.imshow('frame',frame)
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # # When everything done, release the capture
# # cap.release()
# # # and release the output
# # out.release()
# # # finally, close the window
# # cv2.destroyAllWindows()
# # cv2.waitKey(1)