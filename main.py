#!/usr/bin/python
# Import necessary packages
from __future__ import print_function
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import cv2
import argparse
import os

# Parse image argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

def main():
    thresh()
    x = input()
    i = line()
    cnts(x,i)

# Define function for thresholding
def thresh():
    # Blank function for trackbar
    def nothing(x):
        pass

    # Create a named window and trackbar
    cv2.namedWindow('Threshold')
    cv2.createTrackbar('Threshold', 'Threshold', 200, 255, nothing)

    # Create loop to redraw window everytime user changes value on the threshold trackbar
    while(1):
        # Read image
        img = cv2.imread(args["image"])

        # Output Image on Enter key
        if cv2.waitKey(20) & 0xFF == 13:
            cv2.imwrite("seg.png",thr)
            break

        # Get trackbar position and value from named window
        t = cv2.getTrackbarPos('Threshold','Threshold')

        # Convert image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply gaussian blur to reduce noise
        cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)

        # Create binary mask using value from trackbar
        ret,thr = cv2.threshold(img,t,255,cv2.THRESH_BINARY)

        # Invert mask so the foreground objects are white
        thr = cv2.bitwise_not(thr)

        # Define kernel and use morphological close to cover noise in foreground objects
        kernel = np.ones((3,3),np.uint8)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)

        # Overlay mask on original image and display only the objects under white regions of the mask i.e subtract
        img = cv2.bitwise_and(img, thr)

        # Find contours on the binary image mask
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort the contours in descending order of area
        contours  = sorted(contours, key=cv2.contourArea, reverse=True)

        # For each contour found, where the area is between 35px and 2000px,
        # find minimum area to draw a rectangle
        # identiy the four corners of the minimum area rectangle
        # index values in numpy
        # draw the rectangle from these four corners on the subtracted image in line 61
        # show the output in imshow window
        for i,c in enumerate(contours):
            if cv2.contourArea(c) > 2000 or cv2.contourArea(c) < 35:
                continue
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            mg = cv2.drawContours(img,[box],0,(255,255,255),2)
            mg = cv2.imshow('Threshold',mg)

    # destroy all windows when function ends
    cv2.destroyAllWindows()

# Define mouse callback function for mask editing tool
def line():
    # Initiate drawing flag and coordinates
    drawing = False
    ix,iy = -1,-1

    # Line drawing function
    def draw_line(event,x,y,flags):
        global ix,iy,drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = False
            ix,iy = x,y

        elif event == cv2.EVENT_RBUTTONDOWN:
            drawing = False
            ix,iy = x,y

        # Left mouse button draws black lines
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.line(img,(ix,iy),(x,y),(0,0,0),4)

        # Right mouse button draws white lines
        elif event == cv2.EVENT_RBUTTONUP:
            drawing = False
            cv2.line(img,(ix,iy),(x,y),(255,255,255),4)

    img = cv2.imread('seg.png',0)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_line)

    # Draw lines and return image as variable
    while(1):
        cv2.imshow('image',img)
        if cv2.waitKey(20) & 0xFF == 13:
            cv2.destroyAllWindows()
            return img

# Reread original image and return copy as variable
def input():
    image = cv2.imread(args["image"], 0)
    img = image.copy()
    return img

# Draw contours same as before but for the edited mask
def cnts(img,thr):
    img = cv2.bitwise_and(img, thr)
    (x,y,w,h) = cv2.selectROI(img)
    img = img[y:y+h , x:x+w]
    cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours  = sorted(contours, key=cv2.contourArea, reverse=True)

    for i,c in enumerate(contours):
        if cv2.contourArea(c) > 2000 or cv2.contourArea(c) < 35:
            continue
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        img = cv2.drawContours(img,[box],0,(255,255,255),2)

        # Unpack center, size and angle from rect
        center, size, angle = rect[0], rect[1], rect[2]
        center, size = tuple(map(int, center)), tuple(map(int, size))

        # Unpack height and width
        height, width = img.shape[0], img.shape[1]

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1)

        # Warp to roation matrix to straighten chromosome
        img_rot = cv2.warpAffine(img, M, (width, height))

        # Crop the rotated image
        img_crop = cv2.getRectSubPix(img_rot, size, center)

        # Enlarge the cropped image
        img_crop = cv2.resize(img_crop, (0,0), fx=3.5, fy=3.5)

        # Write each chromosome to a separate .png file
        cv2.imwrite("chr"+str(i)+".png", img_crop)

main()
