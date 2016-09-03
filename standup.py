#!/usr/bin/env python

'''
Standup Detector
====================

Usage
-----
standup.py [video file or folder containing videos]

Keys
----
ESC - exit
s   - slow
f   - fast
d   - stop
r   - restart
right arrow  - move foreward
'''


# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import video
from common import anorm2, draw_str
from time import clock


_DEBUG=False

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


color = [(255,0,0),
         (0,255,0),
         (0,0,255),
         (255,255,0),
         (255,0,255),
         (0,255,255),
         (128,0,0),
         (139,0,0),
         (165,42,42),
         (178,34,34),
         (220,20,60),
         (255,0,0),
         (240,128,128),
         (233,150,122),
         (250,128,114),
         (255,160,122),
         (255,69,0),
         (255,140,0),
         (255,165,0),
         (255,215,0),
         (240,230,140),
         (128,128,0),
         (255,255,0),
         (154,205,50),
         (85,107,47),
         (107,142,35),
         (124,252,0),
         (127,255,0),
         (173,255,47),
         (0,100,0),
         (0,128,0),
         (34,139,34),
         (0,255,0),
         (50,205,50),
         (144,238,144),
         (152,251,152),
         (143,188,143),
         (0,250,154),
         (0,255,127),
         (46,139,87),
         (102,205,170),
         (60,179,113),
         (32,178,170),
         (47,79,79),
         (0,128,128),
         (0,139,139),
         (0,255,255),
         (0,255,255),
         (224,255,255),
         (0,206,209),
         (64,224,208),
         (72,209,204),
         (175,238,238),
         (127,255,212),
         (176,224,230),
         (95,158,160),
         (70,130,180),
         (100,149,237),
         (0,191,255),
         (30,144,255),
         (173,216,230),
         (135,206,235),
         (0,0,0)]


class App:
    def __init__(self, video_src):
        self.video_src = video_src
        self.track_len = 20
        self.detect_interval = 5 
        self.tracks = []
        self.cam = video.create_capture(self.video_src)
        self.frame_idx = 0
        self.ch = 0
        
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.rect = []

        self.standup_distance = 15 # dis
        self.minLineNum = 2 # len(standup_tracks)
        self.motion_ratio = 0.2 # motion
        self.minFeaturesNum = 100 # point_cnt
        self.standup_time = 10 # standup_frame_cnt

        self.containsEnoughMotion = False
        self.detected  = False

        self.prevHulls = []
        self.prevHullMask = 0
        self.prevCenters = []
        self.prevHullLabels = []

    def run(self):
        
        standup_frame_cnt = 0
        sitdown_frame_cnt = 0
        #containsEnoughMotion = False
        #detected = False
        delayTime = 1
        
        badframeCnt = 0
        while True:
            ret, frame = self.cam.read()
            if not ret:
                badframeCnt = badframeCnt + 1
                if badframeCnt > 3:
                    break
                else:
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
            minArea = 2
            new_contours=[]
            for index, c in enumerate(contours):
                if len(c) > minLength and cv2.contourArea(c) > minArea: # and hierarchy[index] is not None:
                    new_contours.append(c)
            contours = new_contours

            # get hulls
            hulls = []
            hullMask = np.zeros((closed.shape[0], closed.shape[1], 3), np.uint8)
            for contour in contours:
                hull = cv2.convexHull(contour)
                hulls.append(hull)
            cv2.drawContours( hullMask, hulls, -1, (255,255,255), -1)
             
            # get centers of contours
            centers = []
            for hull in hulls:
                M = cv2.moments(hull)
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
           
            # label hulls
            hullLables = [-1 for i in range(len(hulls))]
            if len(self.prevHulls) == 0:
                hullLables = [i for i in range(len(hulls))]
            else:
                for index, hull in enumerate(hulls):
                    cx = centers[index][0]
                    cy = centers[index][1]
                    if self.prevHullMask[cy][cx][0] != 0:
                        # find corresponding hull
                        minDist = 10000
                        prevIndex = 0

                        for i, c in enumerate(self.prevCenters):
                            dist = abs(int(c[0]) - int(cx)) + abs(int(c[1]) - int(cy))
                            if dist < minDist:
                                minDist = dist
                                prevIndex = i
                        hullLables[index] = self.prevHullLabels[prevIndex]
                    else:
                        label = 0
                        while True:
                            if label in self.prevHullLabels or label in hullLables:
                                label = label + 1
                            else:
                                hullLables[index] = label
                                break
          
            #print("hullLables", hullLables)
            for i, hull in enumerate(hulls):
                cv2.drawContours( vis, [hull], -1, color[hullLables[i]%len(color)], 2)
           
           
            '''
            prevHullMask = np.zeros((closed.shape[0], closed.shape[1], 3), np.uint8)
            if len(self.prevHulls) != 0:
                cv2.drawContours( prevHullMask, self.prevHulls, -1, (255, 255, 255), -1)
                for i, c in enumerate(centers):
                    cv2.drawContours(prevHullMask, [hulls[i]], 0, color[hullLables[i]], 2)
                    cv2.circle(prevHullMask, c, 1, color[hullLables[i]], 2)
                    draw_str(prevHullMask, c, "%d"%hullLables[i], color[hullLables[i]])
                for i, c in enumerate(self.prevCenters):
                    draw_str(prevHullMask, (c[0]+10, c[1]), "%d"%self.prevHullLabels[i])
            cv2.imshow("hulls", prevHullMask)
            '''


            #cv2.imshow('mask', fgmask)
            #cv2.imshow("close", closed)
            #cv2.imshow("contour", hullMask)
            
            #cv2.drawContours( vis, hulls, -1, (128,0,255), 2)
            #cv2.drawContours( vis, contours, -1, (255,255,255), 1)
            #cv2.drawContours( vis, new_contours, -1, (255,0,255), 1)
            
            cv2.imshow('lk_track', vis)

            self.frame_idx += 1
            self.prev_gray = frame_gray
      
            self.prevHulls = hulls
            self.prevHullMask = hullMask
            self.prevHullLabels = hullLables
            self.prevCenters = centers

            self.ch = 0xFF & cv2.waitKey(delayTime) # 20
            ch = self.ch
            # Esc
            if ch == 27:
                break
            # faster
            if ch == ord('g'):
                delayTime = 1
            # fast
            if ch == ord('f'):
                delayTime = 20
            if ch == ord('1'):
                delayTime = 100
            if ch == ord('2'):
                delayTime = 300  
            if ch == ord('3'):
                delayTime = 600
            # slow
            if ch == ord('s'):
                delayTime = 1000
            # replay
            if ch == ord('r'):
                self.cam = video.create_capture(self.video_src)
            # stop   
            if ch == ord('d'):
                ch = 0xFF & cv2.waitKey(delayTime) 
                while ch != ord('d'):
                    ch = 0xFF & cv2.waitKey(delayTime)
                    continue;
            # move foreward > 
            if ch == 83:
                for i in range(20):
                    self.cam.read()

def getVideofiles(directory):
    import os
    if directory == 0:
        return [str(0)]
    if not os.path.exists(directory):
        print("file or directory not exists")
    else:
        if os.path.isfile(directory):
            return [directory]
        elif os.path.isdir(directory):
            files = os.listdir(directory)
            videos = []
            for f in files:
                if f.find("mp4") > 0 and f.find("src") < 0:
                    videos.append(directory+"/"+f)
            return videos


def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    print(__doc__)
    
    videos = getVideofiles(video_src)

    for video in videos:
        print("current video:"+video)
        app = App(video)
        app.run()
        if app.ch == 27:
            break
    
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
