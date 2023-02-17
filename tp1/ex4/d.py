import cv2

IMAGE_PATH = "../out/noisy_fruits.jpg"

image = cv2.imread(IMAGE_PATH)

bilateral = cv2.bilateralFilter(image, 5, 200, 200)

cv2.imshow("Bilateral", bilateral)

cv2.waitKey(0)
cv2.destroyAllWindows()
