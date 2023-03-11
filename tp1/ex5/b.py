import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "../../images/lowContrast_01.jpg"

image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

equalization = cv2.equalizeHist(image)
# result_equalization = np.hstack((image, equalization))

# hist = cv2.calcHist([image], [0], None, [256], [0, 256])

# cv2.imshow("Original", image)
#
# plt.plot(hist, color="black")
# plt.show()
#
hist_equalization = cv2.calcHist([equalization], [0], None, [256], [0, 256])

cv2.imshow("Equalization", equalization)

plt.plot(hist_equalization, color="black")
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
