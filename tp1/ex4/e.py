import cv2
import numpy as np
from scipy import ndimage

IMAGE_PATH = "../out/noisy_fruits.jpg"

image = cv2.imread(IMAGE_PATH)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gaussian_blurred = cv2.GaussianBlur(image, (3, 3), 0)

kernel_3x3 = (1 / 16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
print(kernel_3x3)
result = ndimage.convolve(image, kernel_3x3)

cv2.imshow("Original", image)
cv2.imshow("OpenCV GaussianBlur", gaussian_blurred)
cv2.imshow("My 3x3 convolution with gaussian mask", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
