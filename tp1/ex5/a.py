import cv2
import matplotlib.pyplot as plt

IMAGE_PATH = "../../images/lowContrast_01.jpg"

image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

hist = cv2.calcHist([image], [0], None, [256], [0, 256])

plt.plot(hist, color="black")
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
