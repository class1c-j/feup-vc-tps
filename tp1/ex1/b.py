import cv2

IMAGE_PATH = "../../images/fruits.jpg"

file_name = IMAGE_PATH.rsplit("/", maxsplit=1)[-1]
no_extension = file_name.rsplit(".", maxsplit=1)[0]

image = cv2.imread(IMAGE_PATH)

cv2.imwrite("out/" + no_extension + ".bmp", image)
