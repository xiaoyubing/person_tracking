#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import cv2

# set to 1 for pipeline images
debug = 0


class Detectors(object):
    """Detectors class to detect objects in video frame
    Attributes:
        None
    """
    def __init__(self):
        """Initialize variables used by Detectors class
        Args:
            None
        Return:
            None
        """
        # self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=16, detectShadows=True)
        # self.fgbg = cv2.createBackgroundSubtractorKNN(history=42, dist2Threshold=400.0, detectShadows=True)

    def Detect(self, frame):
        """Detect objects in video frame using following pipeline
            - Convert captured frame from BGR to GRAY
            - Perform Background Subtraction
            - Detect edges using Canny Edge Detection
              http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
            - Retain only edges within the threshold
            - Find contours
            - Find centroids for each valid contours
        Args:
            frame: single video frame
        Return:
            centers: vector of object centroids in a frame
        """

        # Convert BGR to GRAY
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if (debug == 1):
            cv2.imshow('gray', gray)

        # Perform Background Subtraction
        fgmask = self.fgbg.apply(gray)

        if (debug == 0):
            cv2.imshow('bgsub', fgmask)

        ret, thresh = cv2.threshold(fgmask, 50, 255, 0)
        # 创建一个3*3的椭圆核
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # 腐蚀
        thresh = cv2.erode(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        # 膨胀
        thresh = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7)), iterations=3)

        cv2.imshow('膨胀', thresh)
        # 寻找视频中的轮廓
        im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centers = []
        for c in contours:
            perimeter = cv2.arcLength(c, True)
            if perimeter > 200:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), ( x + w, y + h), (0, 0, 255), 2)
                b = np.array([[x+w/2], [y+h/2]])
                centers.append(np.round(b))


        # # Detect edges
        # edges = cv2.Canny(fgmask, 50, 190, 3)
        #
        # if (debug == 0):
        #     cv2.imshow('Edges', edges)
        #
        # # Retain only edges within the threshold
        # ret, thresh = cv2.threshold(edges, 127, 255, 0)
        #
        # # Find contours
        # _, contours, hierarchy = cv2.findContours(thresh,
        #                                           cv2.RETR_EXTERNAL,
        #                                           cv2.CHAIN_APPROX_SIMPLE)
        #
        # if (debug == 0):
        #     cv2.imshow('thresh', thresh)
        #
        # centers = []  # vector of object centroids in a frame
        # # we only care about centroids with size of bug in this example
        # # recommended to be tunned based on expected object size for
        # # improved performance
        # blob_radius_thresh = 8
        # # Find centroid for each valid contours
        # for cnt in contours:
        #     try:
        #         # Calculate and draw circle
        #         (x, y), radius = cv2.minEnclosingCircle(cnt)
        #         centeroid = (int(x), int(y))
        #         radius = int(radius)
        #         if (radius > blob_radius_thresh):
        #             cv2.circle(frame, centeroid, radius, (0, 255, 0), 2)
        #             b = np.array([[x], [y]])
        #             centers.append(np.round(b))
        #     except ZeroDivisionError:
        #         pass
        #
        # # show contours of tracking objects
        # # cv2.imshow('Track Bugs', frame)

        return centers
