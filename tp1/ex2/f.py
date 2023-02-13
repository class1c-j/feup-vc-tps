import cv2

IMAGE_PATH = "../../images/bank_notes_1.JPG"

image = cv2.imread(IMAGE_PATH)

cv2.imshow("Original", image)

cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(image)

cv2.imshow("H", h)
cv2.imshow("S", s)
cv2.imshow("V", v)

s += 10

result = cv2.merge([h, s, v])

cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
