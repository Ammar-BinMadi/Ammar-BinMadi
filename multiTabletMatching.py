
# multi Tamblate matching

import cv2
import matplotlib.pyplot as plt
import numpy
from matplotlib import pyplot as np

# rgb_image
img_rgb = cv2.imread('sourceTissue.jpg')
tamplate_rgb = cv2.imread('templateTissue.png')

#gray image
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
tamplate_gray_big = cv2.cvtColor(tamplate_rgb, cv2.COLOR_BGR2GRAY)
tamplate_gray = cv2.resize(tamplate_gray_big, (17, 17))

# get hight and width of tmaplate
h, w = tamplate_gray.shape[::]

#match tamplate
res = cv2.matchTemplate(img_gray, tamplate_gray, cv2.TM_CCOEFF_NORMED)

threshold = 0.5
loc = numpy.where(res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

cv2.imshow("matched image",img_rgb)
cv2.waitKey()
cv2.destroyWindow()