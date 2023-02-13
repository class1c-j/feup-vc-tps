import cv2

IMAGE_PATH = "../../images/bank_note.JPG"

file_name = IMAGE_PATH.rsplit("/", maxsplit=1)[-1]

image = cv2.imread(IMAGE_PATH)

grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite("../out/grayscale_" + file_name, grayscale)

cv2.imshow("Original", image)
cv2.imshow("Grayscale", grayscale)

cv2.waitKey(0)
cv2.destroyAllWindows()
