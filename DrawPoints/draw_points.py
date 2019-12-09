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

def readPoints():
    file1 = open("points2.cvs","r+")
    pointsString = file1.read()
    print(pointsString)
    pa = pointsString.split(';')
    print(pa)
    points = []
    for p in pa:
        p1 = p.split(',')
        points.append([int(p1[0]), int(p1[1])])
    file1.close()
    return points

def main():
    file = './images/testimg_sampling.tif'
    #file = './images/testimg_peakfilt2.bmp'
    img = cv.imread(file)
    points3 = readPoints()
    for point in points3:
        img[point[1], point[0]] = [0, 0, 255]
        #img[point[1]+1, point[0]] = [0, 0, 255]
        #img[point[1]-1, point[0]] = [0, 0, 255]
        #img[point[1], point[0]+1] = [0, 0, 255]
        #img[point[1], point[0]-1] = [0, 0, 255]
    cv.imshow('image', img)
    cv.imwrite('image_tack.tif', img)
    ch = cv.waitKey()


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
