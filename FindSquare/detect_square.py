#!/usr/bin/env python
'''
this is an example for find square

'''


# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt



def is_cnt_valid(cnt):
    for c in cnt:
        if c[0] ==0 or c[1] == 0:
            return False
        else:
            return True

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )


def sort_cnt(cnt, append):
    center_x = 0
    center_y = 0
    t =[];
    for s in cnt:
        center_x = center_x + s[0]
        center_y = center_y + s[1]

    center_x = center_x / 4
    center_y = center_y / 4

    for s in cnt:
        if s[0] < center_x and s[1] < center_y:
            t.append([s[0] - append, s[1] - append])

    for s in cnt:
        if s[0] > center_x and s[1] < center_y:
            t.append([s[0] + append, s[1] - append])
    for s in cnt:
        if s[0] < center_x and s[1] > center_y:
            t.append([s[0] - append, s[1] + append])
    for s in cnt:
        if s[0] > center_x and s[1] > center_y:
            t.append([s[0] + append, s[1] + append])
    return t

def find_squares(src):
    img = cv.cvtColor(src, cv.COLOR_BGR2GRAY)


    img = cv.GaussianBlur(img, (15, 15), 0)
    cv.imshow('squares gauss blur', img)
    squares = []
    print(len(cv.split(img)))
    mean = cv.mean(img)[0]
    retval, bin = cv.threshold(img, mean, 255, cv.THRESH_BINARY)
    cv.imshow('squares bin', bin)
    contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)


    idx = 0
    append = 15

    for cnt in contours:
        cnt_len = cv.arcLength(cnt, True)
        cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
        if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
            #print(cnt)
            cnt = cnt.reshape(-1, 2)
            # check cnt in the middle
            if is_cnt_valid(cnt):
                max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in xrange(4)])
                if max_cos < 0.1:
                    squares.append(cnt)

                    mask=src.copy()
                    #cv.polylines(mask, cnt, 2, 255)
                    #cv.drawContours(mask,contours, idx, (0,0,255),2)

                    #cv.polylines(mask, np.int32([cnt]),1,255)
                    cv.imshow('mask',mask)
                    print(cnt)


                    #                            pts1 = np.float32([cnt[1], cnt[0], cnt[2], cnt[3]])

                    pts1 = np.float32(sort_cnt(cnt, append))
                    print('perspect p ', pts1)
                    pts2 = np.float32([[0, 0], [1160+append, 0], [0, 360+append], [1160+append, 360+append]])
                    print('end')
                    matrix = cv.getPerspectiveTransform(pts1, pts2)
                    result = cv.warpPerspective(mask, matrix, (1160+append*2, 360+append*2))
                    cv.imshow('perspective',result)
                    cv.imwrite('perspective.tif', result)
                    ch = cv.waitKey()

        idx = idx + 1


    return squares


def main():
    file = './images/1575422597008.jpg'

    crop_width = 1300
    crop_height = 440

    print(file)
    img = cv.imread(file)
    h,w,l = np.shape(img)
    offset_x = (int)((w-crop_width)/2)
    offset_y = (int)((h-crop_height)/2)
    crop_img = img[offset_y:(offset_y+crop_height), offset_x:(offset_x+crop_width)]
    cv.imshow('squares', crop_img)
    cv.imwrite('output_{0}.bmp'.format(121), crop_img)

    squares = find_squares(crop_img)
    #print(squares)
    #cv.drawContours( img, squares, -1, (0, 255, 0), 3 )

    i = 0
    for s in squares:
        print(s)
        border =5

        #replicate = cv.copyMakeBorder(img, s[0][1], s[3][1], s[0][0], s[3][0], cv.BORDER_CONSTANT)
        ball = crop_img[s[0][1]-border:s[2][1]+border, s[0][0]-border:s[2][0]+border]
        #cv.imshow('s', ball)
        hist = cv.calcHist(ball, [0], None, [256], [0, 256])
        plt.hist(ball.ravel(), 256, [0, 256]);
        plt.show()
        cv.imwrite('output_{0}.bmp'.format(i), ball)
        i = i + 1
        #break
    #ch = cv.waitKey()
    ch = cv.waitKey()


    print('Done')






if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()