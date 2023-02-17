import cv2

IMAGE_PATH = "../out/noisy_fruits.jpg"

image = cv2.imread(IMAGE_PATH)

gaussian = cv2.GaussianBlur(image, (5, 5), 0)

cv2.imshow("Gaussian", gaussian)

cv2.waitKey(0)
cv2.destroyAllWindows()
