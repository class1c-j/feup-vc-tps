import cv2
import numpy as np

HEIGHT = 100
WIDTH = 200

# Create white image
matrix = np.ones((HEIGHT, WIDTH, 1), np.uint8) 

# Intenstity at 100
matrix *= 100

# Draw diagonals with intensity 255
cv2.line(matrix, (0, 0), (WIDTH, HEIGHT), (255, 255, 255), 5)
cv2.line(matrix, (WIDTH, 0), (0, HEIGHT), (255, 255, 255), 5)


cv2.imshow("Image", matrix)

cv2.waitKey(0)
cv2.destroyAllWindows()
