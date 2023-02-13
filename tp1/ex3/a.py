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

    # Display the resulting frame
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord("c"):
        cv2.imshow("Capture", frame)
        cv2.imwrite("../out/capture.jpg", frame)
        cv2.destroyWindow("frame")
        break


capture.release()

cv2.waitKey(0)
cv2.destroyAllWindows()
