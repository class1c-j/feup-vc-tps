"""%%%
# 10. Local Features - Blob Detectors
%%"""


import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


def display_image(img, title=""):
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


"""%%%
**a)** Use the SIFT blob detector to detect the keypoints in an image and show their position over the image, like in
figure 2
%%"""

IMAGE_PATH = "../../images/home.jpg"

image = cv2.imread(IMAGE_PATH)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
keypoints = sift.detect(gray, None)
img = cv2.drawKeypoints(
    gray, keypoints, image, (-1, -1, -1), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

display_image(image, "SIFT")

"""%%%
**b)** Use the ORB blob detector to detect the keypoints in an image and show their position over the image
%%"""

image = cv2.imread(IMAGE_PATH)

# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints with ORB
keypoints = orb.detect(image, None)
# compute the descriptors with ORB
keypoints, des = orb.compute(image, keypoints)
# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)

display_image(img2, "ORB")
