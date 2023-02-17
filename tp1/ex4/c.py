import cv2

IMAGE_PATH = "../out/noisy_fruits.jpg"

image = cv2.imread(IMAGE_PATH)

median = cv2.medianBlur(image, 5)

cv2.imshow("Median", median)

cv2.waitKey(0)
cv2.destroyAllWindows()
