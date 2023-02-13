import cv2

IMAGE_PATH = "../../images/fruits.jpg"

image = cv2.imread(IMAGE_PATH)

b_channel, g_channel, r_channel = cv2.split(image)

cv2.imshow("Red", r_channel)
cv2.imshow("Green", g_channel)
cv2.imshow("Blue", b_channel)

b_channel += 100

merged_channels = cv2.merge([b_channel, g_channel, r_channel])

cv2.imshow("Result", merged_channels)

cv2.waitKey(0)
cv2.destroyAllWindows()
