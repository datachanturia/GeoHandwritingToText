
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

img = cv2.imread("./images/chars/g/ambrost0.jpg23.jpg", cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap=plt.cm.gray)
plt.show()

A = img.shape[0] / 3.0
w = img.shape[1] / 2.6

shift = lambda x: A * np.sin(2.0*np.pi*x * w)

for i in range(img.shape[0]):
    img[:,i] = np.roll(img[:,i], int(shift(i)))

plt.imshow(img, cmap=plt.cm.gray)
plt.show()
