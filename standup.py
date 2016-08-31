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
        self.track_len = 20
        self.detect_interval = 5 
        self.tracks = []
        self.cam = video.create_capture(video_src)
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
            cv2.imshow("fgmask", fgmask)

            if len(self.tracks) > 0:
                
                #########
                # Step 3: Track goodFeatures 
                #########
                # track feature points using OpticalFlowPyrLK
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))
                
                draw_str(vis, (20, 80), 'motion: %f' % self.motion_ratio)
                draw_str(vis, (20, 100), 'minFeaturesNum: %d' % self.minFeaturesNum)
                draw_str(vis, (20, 120), 'standup_time: %d' % self.standup_time)
                
                #########
                # Step 4: Detect "stand up" pattern in self.tracks
                # 1. revise(2): revise distortion from depth(large near and small far)
                # 2. standup_distance(15): decide if it's standup feature lines
                # 3. minLineNum(2): standup region should contain enough standup feature lines
                # 4. minFeaturesNum(100): standup region should contain enough goodFeatures
                # 5. standup_time(10): standing up need some time
                #########
                
                # each in self.tracks is list, containing at most 20(self.track_len) points.

                # get standup_tracks and sitdown_tracks
                #standup_distance = 15
                standup_tracks = []
                sitdown_tracks = []
                region = (np.int(0.15*vis.shape[0]), np.int(0.8*vis.shape[0]))
                revise = 2
                for tr in self.tracks:
                    dis = tr[0][1]-tr[-1][1] # only consider vertical distance
                
                    # revise dis according to y (tr[0][1])
                    alpha = revise * abs(tr[0][1] - region[1])*1.0 / abs(region[0] - region[1])
                    dis *= (1+alpha)

                    # [standup]
                    if dis > self.standup_distance:
                        standup_tracks.append(tr)
                    
                    # [sitdown]
                    if -dis > self.standup_distance:
                        sitdown_tracks.append(tr)

                draw_str(vis, (20, 40), 'standup lines[%d]: %d' % (self.minLineNum, len(standup_tracks)))
                draw_str(vis, (20, 60), 'sitdown lines[%d]: %d' % (self.minLineNum, len(sitdown_tracks)))

                # judge standup region
                #minLineNum = 2
                if len(standup_tracks) > self.minLineNum:
                    cv2.polylines(vis, [np.int32(tr) for tr in standup_tracks], False, (255, 0, 0))

                    # step 1: get standup rectangle
                    center = [0,0]
                    point_sum = 0
                    for tr in standup_tracks:
                        for point in tr:
                            center[0] += point[0]
                            center[1] += point[1]
                            point_sum += 1
                    center[0] = np.int(1.0*center[0]/point_sum)
                    center[1] = np.int(1.0*center[1]/point_sum)
                    halfSize = 40
                    self.rect = [(center[0]-halfSize, center[1]-halfSize*2), (center[0]+halfSize, center[1]+halfSize*2)]

                    # step 2: rect should contain enough motion region
                    # but don't know how to add to result
                    #motion_ratio = 0.2
                    motion = 0
                    roi = fgmask[self.rect[0][1]:self.rect[1][1], self.rect[0][0]:self.rect[1][0]]
                    hist = cv2.calcHist([roi],[0],None,[256],[0,256])
                    for x,y in enumerate(hist):
                        if x > 100:
                            motion += y
                    if roi.size > 0:
                        motion = 1.0*motion/roi.size
                        if motion > self.motion_ratio:
                            self.containsEnoughMotion = True
                    #cv2.imshow('roi', roi)
                    #cv2.imshow('hist', draw_hist(roi)) 

                    # step 3: rect should contain enough feature points
                    #minFeaturesNum = 100
                    point_cnt = 0
                    for tr in self.tracks: 
                        # use self.tracks other than standup_tracks
                        # cause self.tracks is much more than standup_tracks and may be more rebust.
                        for point in tr:
                            if point[0] > self.rect[0][0] and point[0] < self.rect[1][0] and point[1] > self.rect[0][1] and point[1] < self.rect[1][1]: 
                                point_cnt += 1
                    if point_cnt > self.minFeaturesNum:
                        standup_frame_cnt = standup_frame_cnt + 1
                    else: 
                        standup_frame_cnt = 0

                    # Step 4: standup continues for some time
                    #standup_time = 10
                    if standup_frame_cnt > self.standup_time and self.containsEnoughMotion:
                        self.detected = True 
                    
                    cv2.rectangle(vis, self.rect[0], self.rect[1], (0, 255, 255), 1)
                    draw_str(vis, (self.rect[0][0], self.rect[1][1]+20), 'motion:%f' % (motion))
                    draw_str(vis, (self.rect[0][0], self.rect[1][1]+40), 'points:%d' % (point_cnt))
                    draw_str(vis, (self.rect[0][0], self.rect[1][1]+60), 'time:%d' % (standup_frame_cnt))


                minLineNum_sitdown = self.minLineNum  
                minFeaturesNum_sitdown= 3
                sitdown_time = self.standup_time
                if len(sitdown_tracks) > minLineNum_sitdown:
                    cv2.polylines(vis, [np.int32(tr) for tr in sitdown_tracks], False, (0, 0, 255))
                    
                    # sitdown_tracks points should be in detecting rect and has minFeaturesNum and 
                    # sitdown continues for some time
                    if self.detected:
                        point_cnt = 0
                        for tr in sitdown_tracks:
                            for point in tr:
                                if point[0] > self.rect[0][0] and point[0] < self.rect[1][0] and point[1] > self.rect[0][1] and point[1] < self.rect[1][1]: 
                                    point_cnt += 1
                        
                        #print("point_cnt in for sitdown: ", point_cnt)
                        #print("rect: ",self.rect)
                        if point_cnt > minFeaturesNum_sitdown:
                            sitdown_frame_cnt += 1
                        
                        if sitdown_frame_cnt > sitdown_time:
                            self.detected = False
                            self.containsEnoughMotion = False
                    else:
                        sitdown_frame_cnt = 0
                #
                # End Step 4
                ############


            if self.detected:
                cv2.rectangle(vis, self.rect[0], self.rect[1], (0, 255, 255), 5)
                    
        
            #########
            # Step 2: find goodFeatures 
            #########
            # find good points to track and saved in self.tracks
            if self.frame_idx % self.detect_interval == 0:

                # remove static points
                minDistance = 10
                update_tracks = []
                for tr in self.tracks:
                    if len(tr) > 5:
                        if abs(tr[0][0]-tr[-1][0]) + abs(tr[0][1]-tr[-1][1]) > minDistance:
                            update_tracks.append(tr)
                self.tracks = update_tracks

                # find feature points in fgmask
                p = cv2.goodFeaturesToTrack(frame_gray, mask = fgmask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])
            
            ## find student area
            #region = (0.15, 0.8)
            #for y in range(0, np.int(region[0]*vis.shape[0])):
            #    for x in range(vis.shape[1]):
            #        vis[y][x] = (0,0,0)

            #for y in range(np.int(region[1]*vis.shape[0]), vis.shape[0]):
            #    for x in range(vis.shape[1]):
            #        vis[y][x] = (0,0,0)
            
            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)
            #cv2.imshow('mask', fgmask)

            ch = 0xFF & cv2.waitKey(delayTime) # 20
            if ch == 27:
                break
            if ch == ord('f'):
                delayTime = 20
            if ch == ord('s'):
                delayTime = 500
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
