#!/usr/bin/env python

'''
It is a startup for opencv in python
====================

Usage
-----
startup.py [video file or folder containing videos]

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
from hog_svm import StatModel, SVM


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
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    hist_item = cv2.calcHist([im],[0],None,[256],[0,256])
    cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
    hist=np.int32(np.around(hist_item))
    for x,y in enumerate(hist):
        cv2.line(h,(x,0),(x,y),(255,255,255))
    y = np.flipud(h)
    return y


MHI_DURATION = 0.5
DEFAULT_THRESHOLD = 32
MAX_TIME_DELTA = 0.25
MIN_TIME_DELTA = 0.05


def draw_motion_comp(vis, (x, y, w, h), angle, color):
    draw_str(vis, (x,y), "%.1f"%angle)
    cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
    
    #r = min(w/2, h/2)
    #cx, cy = x+w/2, y+h/2
    #angle = angle*np.pi/180
    #cv2.circle(vis, (cx, cy), r, color, 3)
    #cv2.line(vis, (cx, cy), (int(cx+np.cos(angle)*r), int(cy+np.sin(angle)*r)), color, 3)

svm = 0

class App:
    def __init__(self, video_src):
        self.video_src = video_src
        self.ch = 0
        self.cam = video.create_capture(self.video_src)
        self.hogConf = "hog.xml"
        self.svmConf = "mhi_svm.dat"
        self.svm = svm
        # add other vars

    def run(self):
        # init local vars
        badframeCnt = 0
        delayTime = 20
        jump = 20

        # algorithm parameters
        thres = 32

        ret, frame = self.cam.read()
        frame = cv2.resize(frame, (frame.shape[1]/2 , frame.shape[0]/2))
        h, w = frame.shape[:2]
        prev_frame = frame.copy()
        mhi = np.zeros((h,w), np.float32)
        hsv = np.zeros((h,w,3), np.uint8)
        hsv[:,:,1] = 255
       
        hog = cv2.HOGDescriptor(self.hogConf)
        normalSize = (90, 66)

        while True:
            # read frame
            ret, frame = self.cam.read()
            if not ret:
                badframeCnt = badframeCnt + 1
                if badframeCnt > 3:
                    break
                else:
                    continue
            frame = cv2.resize(frame, (frame.shape[1]/2 , frame.shape[0]/2))

            ######################
            # Add your code here #
            ######################
            
            frame_diff = cv2.absdiff(frame, prev_frame)
            gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
            ret, motion_mask = cv2.threshold(gray_diff, thres, 1, cv2.THRESH_BINARY)
            timestamp = clock()
            vis = frame.copy()

            cv2.motempl.updateMotionHistory(motion_mask, mhi, timestamp, MHI_DURATION)
            mg_mask, mg_orient = cv2.motempl.calcMotionGradient( mhi, MAX_TIME_DELTA, MIN_TIME_DELTA, apertureSize=5 )
            seg_mask, seg_bounds = cv2.motempl.segmentMotion( mhi, timestamp, MAX_TIME_DELTA )
            mei = np.uint8(np.clip((mhi-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)
           
            maxArea = 64*2
            indexArea = 0
            for i, rect in enumerate([(0,0,w,h)] + list(seg_bounds)):
                x, y, rw, rh = rect
                area = rw * rh
                if area < 64*2 or rw < 16 or rh < 16:
                    continue
                silh_roi    = motion_mask   [y:y+rh, x:x+rw]
                orient_roi  = mg_orient     [y:y+rh, x:x+rw]
                mask_roi    = mg_mask       [y:y+rh, x:x+rw] 
                mhi_roi     = mhi           [y:y+rh, x:x+rw] 
                if cv2.norm(silh_roi, cv2.NORM_L1) < area*0.05:
                    continue
                angle = cv2.motempl.calcGlobalOrientation(orient_roi, mask_roi, mhi_roi, timestamp, MHI_DURATION)

                
                # compute hog
                mei_roi = mei[y:y+rh, x:x+rw] 
                mei_roi = cv2.resize(mei_roi, normalSize)
                print(mei_roi.shape)
                features = hog.compute(mei_roi, hog.blockStride, hog.cellSize, ((0,0),) )
                
                # classify using svm
                print(np.array(features).shape, np.array(features).dtype)
                result = self.svm.predict(np.array(features))
                print(result)
               
                color = ((255,0,0), (0,0,255))[result]
                draw_motion_comp(vis, rect, angle, color)

            # show result
            #cv2.imshow("motion energy image", mei)
            cv2.imshow("motion history image", mhi)
            cv2.imshow("vis", vis)

            prev_frame = frame.copy()
            
            ####################
            # keyboard control #
            ####################
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
            if ch == 82:
                jump = jump + 20
                print("jupm speed: ", jump)
            if ch == 84:
                jump = (jump - 20 > 0) and (jump - 20) or jump
                print("jupm speed: ", jump)
            if ch == 83:
                for i in range(jump):
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
                if f.find("mp4") > 0 and f.find("src") < 0 and f.find("txt") < 0:
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
    
    cv2.destroyAllWindows()

if __name__ == '__main__':

    import hog_svm as model
    global svm 
    svm = model.main()

    main()
