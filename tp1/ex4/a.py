import cv2

IMAGE_PATH = "../out/noisy_fruits.jpg"

image = cv2.imread(IMAGE_PATH)

mean = cv2.blur(image, (4, 4))

cv2.imshow("Mean", mean)

cv2.waitKey(0)
cv2.destroyAllWindows()
