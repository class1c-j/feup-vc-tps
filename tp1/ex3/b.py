import sys
import cv2

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
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow("frame", frame)
    cv2.imshow("gray", grayscale)

    if cv2.waitKey(1) == ord("q"):
        break


capture.release()
cv2.destroyAllWindows()
