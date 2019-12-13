#!/usr/bin/env python
'''
this is an example for draw points

'''


# Python 2/3 compatibility
from __future__ import print_function
import sys
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range


def main():
    file = './images/5295f2f8-9825-43b8-bda0-a07b279ce145.jpg'
    #file = './images/testimg_peakfilt2.bmp'
    img = cv.imread(file)

    kernel = np.ones((3,3),np.uint8)
    erosion = cv.erode(img,kernel,iterations = 1)
    cv.imshow('image', erosion)
    cv.imwrite('image_tack.jpeg', erosion)
    ch = cv.waitKey()


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
