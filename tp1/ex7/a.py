import cv2

img = cv2.imread("../../images/home.jpg", 0)


min_threshold = 0
max_threshold = 255
aperture_size = 3


def trackbar_callback(x):
    edges = cv2.Canny(img, 100, 200)


cv2.waitKey(0)
cv2.destroyAllWindows()
