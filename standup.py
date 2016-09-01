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
         
        prev_contours = []
        prev_contourMask = []
        prev_centers = []
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

            
            #########
            # Step 2: morpology
            #########
            fgmask = cv2.medianBlur(fgmask, 7)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            closed = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)


            
            #########
            # Step 3: contour
            #########
            _, contours0, hierarchy = cv2.findContours( closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]    

            # filter contours
            minLength = 2
            minArea = 20
            new_contours=[]
            for c in contours:
                if len(c) > minLength and cv2.contourArea(c) > minArea:
                    new_contours.append(c)
            contours = new_contours

            # get contour mask in hull way
            hulls = []
            contourMask = np.zeros((closed.shape[0], closed.shape[1], 3), np.uint8)
            for contour in contours:
                hull = cv2.convexHull(contour)
                hulls.append(hull)
            cv2.drawContours( contourMask, hulls, -1, (255,255,255), 1)
           
            # get centers of contours
            centers = []
            for contour in contours:
                M = cv2.moments(contour)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                centers.append((cx, cy))

            '''
            maxLength = 0
            maxLength_index = 0
            for i in range(0, len(contours)):
                if len(contours[i]) > maxLength:
                    maxLength = len(contours[i])
                    maxLength_index = i
            '''

            # find relationship between contours and prev_contours
            new_contours = []
            if len(prev_contours) > 0:
                prev_flag = [0 for i in range(len(prev_contours))]
                isSplit = [False for i in range(len(contours))]
                relation = [0 for i in range(len(contours))] # index is contours, value is prev_contours
                # find one split into two case
                for i1, contour in enumerate(contours):
                    cx = centers[i1][0] 
                    cy = centers[i1][1]
                    if prev_contourMask[cy][cx][0] == 255:        
                        # find closest prev_contour and add 1
                        minDistance = 0
                        minI2 = 0
                        for i2, prev_center in enumerate(prev_centers):
                            distance = abs(prev_center[0]-cx) + abs(prev_center[1]-cy)
                            if distance < minDistance:
                                minDistance = distance 
                                minI2 = i2 
                        prev_flag[minI2] = prev_flag[minI2] + 1
                        if prev_flag[minI2] > 1:
                            isSplit[i1] = True
                        relation[i1] = minI2
                
                #new_contours = []
                haveRead = [False for i in range(len(prev_contours))]
                for i1, contour in enumerate(contours):
                    if isSplit[i1] is True:
                        if haveRead[relation[i1]] is False:
                            new_contours.append(contours[relation[i1]])
                            haveRead[relation[i1]] = True
                    else:
                        new_contours.append(contour)
                #contours = new_contours
            
            print(len(contours))
            print(len(new_contours))
            #cv2.imshow('mask', fgmask)
            #cv2.imshow("close", closed)
            
            hulls2 = []
            for contour in new_contours:
                hull = cv2.convexHull(contour)
                hulls2.append(hull)
            cv2.drawContours( contourMask, hulls2, -1, (255,0,255), 2)
            
            #cv2.imshow("contour", contourMask)

            #cv2.drawContours( vis, contours, maxLength_index, (128,0,255), 3)
            cv2.drawContours( vis, contours, -1, (255,255,255), 1)
            cv2.drawContours( vis, new_contours, -1, (255,0,255), 1)
            cv2.imshow('lk_track', vis)


            self.frame_idx += 1
            self.prev_gray = frame_gray
       
            prev_contours = len(new_contours) == 0 and contours or new_contours 
            prev_contourMask = contourMask
            prev_centers = centers


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
