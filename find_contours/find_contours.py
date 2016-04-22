import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('test.jpg', 0)
blur1 = cv2.GaussianBlur(img, (5,5), 0)
blur2 = cv2.medianBlur(img, 5)
blur3 = cv2.blur(img, (5,5))
blur4 = cv2.bilateralFilter(img,9,75,75)
edges = cv2.Canny(img, 150, 200)
thresh = cv2.adaptiveThreshold(blur4, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 121, 9)
#ret, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
#print len(cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE))
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[4]
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
plt.figure()
plt.imshow(im2)
plt.show()
