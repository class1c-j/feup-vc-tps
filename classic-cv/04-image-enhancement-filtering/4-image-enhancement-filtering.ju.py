"""%%%
# 4. Image Enhancement - Filtering
%%"""
import cv2
import matplotlib.pyplot as plt
import numpy as np


def display_image(image):
    plt.figure()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


"""%%%
Take a noisy image (see problem 2.d) and filter it (try different filter sizes), using

**a)** a mean filter;
%%"""

IMAGE_PATH = "../2-images-representation-color-spaces/out/noisy_fruits.jpg"

image = cv2.imread(IMAGE_PATH)

mean = cv2.blur(image, (4, 4))

display_image(mean)

"""%%%
**b)** a Gaussian filter;
%%"""

gaussian = cv2.GaussianBlur(image, (5, 5), 0)

display_image(gaussian)

"""%%%
**c)** a median filter;
%%"""

median = cv2.medianBlur(image, 5)

display_image(median)

"""%%%
**d)** a bilateral filter;
%%"""

bilateral = cv2.bilateralFilter(image, 5, 200, 200)

display_image(bilateral)

"""%%%
**e)** a filter defined by you, adapting the following code;
%%"""

import cv2
import numpy as np
from scipy import ndimage

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gaussian_blurred = cv2.GaussianBlur(image, (3, 3), 0)

kernel_3x3 = (1 / 16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
print(kernel_3x3)
result = ndimage.convolve(image, kernel_3x3)

print("Original image")
display_image(image)

print("OpenCV GaussianBlur")
display_image(gaussian_blurred)

print("My 3x3 convolution with gaussian mask")
display_image(result)
