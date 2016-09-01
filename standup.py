#!/usr/bin/env python

'''
Standup Detector
====================

Usage
-----
standup.py [<video_source>]

Keys
----
ESC - exit
s   - slow
f   - fast
d   - stop
r   - restart
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import video
from common import anorm2, draw_str
from time import clock

# calcOpticalFlowPyrLK
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# goodFeaturesToTrack
feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.1,
                       minDistance = 7,
                       blockSize = 20 )


def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


def draw_hist(im):
    h = np.zeros((300,256,3))
    if len(im.shape)!=2:
        print("hist_lines applicable only for grayscale images")
        #print("so converting image to grayscale for representation"
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    hist_item = cv2.calcHist([im],[0],None,[256],[0,256])
    cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
    hist=np.int32(np.around(hist_item))
    for x,y in enumerate(hist):
        cv2.line(h,(x,0),(x,y),(255,255,255))
    y = np.flipud(h)
    return y


class App:
    def __init__(self, video_src):
        self.video_src = video_src
        self.track_len = 20
        self.detect_interval = 5 
        self.tracks = []
        self.cam = video.create_capture(self.video_src)
        self.frame_idx = 0
        
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.rect = []

        self.standup_distance = 15 # dis
        self.minLineNum = 2 # len(standup_tracks)
        self.motion_ratio = 0.2 # motion
        self.minFeaturesNum = 100 # point_cnt
        self.standup_time = 10 # standup_frame_cnt

        self.containsEnoughMotion = False
        self.detected  = False

    def run(self):
        
        standup_frame_cnt = 0
        sitdown_frame_cnt = 0
        #containsEnoughMotion = False
        #detected = False
        delayTime = 500
        
        while True:
            ret, frame = self.cam.read()
            if not ret:
                continue
            
            #########
            # Step 0: preprocessing
            #########
            frame = cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2) )
            vis = frame.copy()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            #cv2.imshow("before blur", frame_gray)
            frame_gray = cv2.medianBlur(frame_gray, 7)
            #cv2.imshow("after blur", frame_gray)
            
            
            #########
            # Step 1: BackgroundSubtractor
            #########
            fgmask = self.fgbg.apply(frame_gray, 0.7)
            #cv2.imshow("fgmask", fgmask)

            fgmask = cv2.medianBlur(fgmask, 7)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            closed = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
            cv2.imshow("close", closed)

            _, contours0, hierarchy = cv2.findContours( closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]    
            print(len(contours))

            maxLength = 0
            maxLength_index = 0
            for i in range(0, len(contours)):
                if len(contours[i]) > maxLength:
                    maxLength = len(contours[i])
                    maxLength_index = i

            #contourImg = np.zeros((closed.shape[0], closed.shape[1], 3), np.uint8)
            #cv2.drawContours( contourImg, contours, maxLength_index, (128,0,255), 3)
            #cv2.drawContours( contourImg, contours, -1, (255,255,255), 1)
            #cv2.imshow("contour", contourImg)

            
            self.frame_idx += 1
            self.prev_gray = frame_gray
           
            cv2.drawContours( vis, contours, maxLength_index, (128,0,255), 3)
            cv2.drawContours( vis, contours, -1, (255,255,255), 1)
           
            cv2.imshow('lk_track', vis)
            #cv2.imshow('mask', fgmask)

            ch = 0xFF & cv2.waitKey(delayTime) # 20
            if ch == 27:
                break
            if ch == ord('g'):
                delayTime = 1
            if ch == ord('f'):
                delayTime = 20
            if ch == ord('s'):
                delayTime = 500
            if ch == ord('r'):
                self.cam = video.create_capture(self.video_src)
            if ch == ord('d'):
                ch = 0xFF & cv2.waitKey(delayTime) 
                while ch != ord('d'):
                    ch = 0xFF & cv2.waitKey(delayTime)
                    continue;

def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    print(__doc__)
    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
