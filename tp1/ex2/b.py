import cv2
import numpy as np

HEIGHT = 100
WIDTH = 200

image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

image[:] = (0, 255, 255)

cv2.line(image, (0, 0), (WIDTH, HEIGHT), (0, 0, 255), 5)
cv2.line(image, (WIDTH, 0), (0, HEIGHT), (255, 0, 0), 5)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
