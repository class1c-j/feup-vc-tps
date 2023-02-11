import cv2

IMAGE_PATH = "../../images/fruits.jpg"

file_name = IMAGE_PATH.rsplit("/", maxsplit=1)[-1]

print(file_name)

image = cv2.imread(IMAGE_PATH)

print("Height:", image.shape[0])
print("Width:", image.shape[1])

cv2.imshow(file_name, image)

cv2.waitKey()
