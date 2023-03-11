import cv2

IMAGE_PATH = "../../images/home.jpg"

image = cv2.imread(IMAGE_PATH)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gaussian_blurred = cv2.GaussianBlur(image_gray, (7, 7), 0)

sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=5)

abs_grad_x = cv2.convertScaleAbs(sobel_x)
abs_grad_y = cv2.convertScaleAbs(sobel_y)

grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted"
trackbar_value = "Value"
max_value = 255
max_type = 4
max_binary_value = 255
window_name = "Threshold Demo"


def Threshold_Demo(val):
    # 0: Binary
    # 1: Binary Inverted
    # 2: Threshold Truncated
    # 3: Threshold to Zero
    # 4: Threshold to Zero Inverted
    threshold_type = cv2.getTrackbarPos(trackbar_type, window_name)
    threshold_value = cv2.getTrackbarPos(trackbar_value, window_name)
    _, dst = cv2.threshold(grad, threshold_value, max_binary_value, threshold_type)
    cv2.imshow(window_name, dst)


cv2.namedWindow(window_name)
cv2.createTrackbar(trackbar_type, window_name, 3, max_type, Threshold_Demo)
# Create Trackbar to choose Threshold value
cv2.createTrackbar(trackbar_value, window_name, 0, max_value, Threshold_Demo)

Threshold_Demo(0)

cv2.waitKey(0)
cv2.destroyAllWindows()
