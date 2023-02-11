import cv2

IMAGE_PATH = "../../images/fruits.jpg"

file_name = IMAGE_PATH.rsplit("/", maxsplit=1)[-1]

image = cv2.imread(IMAGE_PATH)

selected_points = []


def mouse_callback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(selected_points) < 2:
            selected_points.append((x, y))
        if len(selected_points) == 2:
            cropped_image = image[
                selected_points[0][1] : selected_points[1][1],
                selected_points[0][0] : selected_points[1][0],
            ]
            cv2.imwrite("../out/cropped_" + file_name, cropped_image)
            cv2.imshow(file_name, cropped_image)


cv2.namedWindow(file_name)
cv2.setMouseCallback(file_name, mouse_callback)

cv2.imshow(file_name, image)

cv2.waitKey(0)
cv2.destroyAllWindows()
