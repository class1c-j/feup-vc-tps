import cv2

IMAGE_PATH = "../../images/fruits.jpg"

file_name = IMAGE_PATH.rsplit("/", maxsplit=1)[-1]

image = cv2.imread(IMAGE_PATH)


def mouse_callback(event, x, y, flags, params):
    if event == cv2.EVENT_MOUSEMOVE:
        print(
            f"coords {x, y}, colors Blue: {image[y, x, 0]} , Green: {image[y, x, 1]}, Red: {image[y, x, 2]} "
        )
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Editing pixel on", x, y)
        red = input("New red? ")
        green = input("New green? ")
        blue = input("New blue? ")
        image[y, x] = (blue, green, red)
        cv2.imshow(file_name, image)


cv2.namedWindow(file_name)
cv2.setMouseCallback(file_name, mouse_callback)

cv2.imshow(file_name, image)

cv2.waitKey(0)
cv2.destroyAllWindows()
