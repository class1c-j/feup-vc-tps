import sys
import cv2
import numpy as np

capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Failed to open camera")
    sys.exit()

while True:

    # Capture frame by frame
    ret, frame = capture.read()

    # If frame is read correctly, ret is True
    if not ret:
        print("Failed to receive frame")
        break

    # Transformations on the frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the resulting frame
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("result", result)

    if cv2.waitKey(1) == ord("q"):
        break


capture.release()
cv2.destroyAllWindows()
