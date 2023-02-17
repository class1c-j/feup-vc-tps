import cv2
import numpy as np

IMAGE_PATH = "../../images/fruits.jpg"

image = cv2.imread(IMAGE_PATH)

file_name = IMAGE_PATH.rsplit("/", maxsplit=1)[-1]

channels = image.shape[2]
PROBABILITY = 0.1

result = image.copy()

cv2.imshow("Image", image)

if len(image.shape) == 2:  # Grayscale image
    pepper = 0
    salt = 255
else:
    if channels == 3:  # RGB image
        pepper = np.array([0, 0, 0], dtype="uint8")
        salt = np.array([255, 255, 255], dtype="uint8")
    else:  # RGBA image
        pepper = np.array([0, 0, 0, 255], dtype="uint8")
        salt = np.array([255, 255, 255, 255], dtype="uint8")

probs = np.random.random(image.shape[:2])
result[probs < (PROBABILITY / 2)] = pepper
result[probs > 1 - (PROBABILITY / 2)] = salt

cv2.imshow("Result", result)

cv2.imwrite("../out/noisy_" + file_name, result)

cv2.waitKey(0)
cv2.destroyAllWindows()
