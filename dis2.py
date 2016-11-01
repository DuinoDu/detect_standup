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
import os

import sys
sys.path.insert(0, './mtcnn')
import demo_mtcnn

_DEBUG=False

roiNumber = 0

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


def draw_flow(img, flow, step=10):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    
    lines = np.int32(lines + 0.5)
    vis = img #cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for l in lines:
        #cv2.polylines(vis, lines, 0, (0, 255, 0))
        # since "lines = np.int32(lines+0.5)" 
        # most values are zero
        if l[0][1] < l[1][1]:
            cv2.polylines(vis, [l], 0, (0, 255, 0))
        elif l[0][1] > l[1][1]:
            cv2.polylines(vis, [l], 0, (0, 0, 255))

    for (x1, y1), (x2, y2) in lines:
        if y1 < y2:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        elif y1 > y2:
            cv2.circle(vis, (x1, y1), 1, (0, 0, 255), -1)
    return vis


def draw_flow_roi(img, flow,  roi, step=10):
    h, w = roi[3], roi[2] 
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    
    y, x = np.mgrid[roi[1]+step/2 : roi[1]+h : step, roi[0]+step/2 : roi[0]+w : step].reshape(2,-1).astype(int)
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = img #cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def drawface(img, faceRegion):
    for r in faceRegion:    
        cv2.rectangle(img, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (0,255,0), 3)
    return img

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

        self.fy_threshold = -0.5

        self.prevCenters = []
        self.prevContourLabels = []
        self.prevContourMask = 0
        self.prevLabelHist = []
        self.maxLabel = 200
        self.minTime = 5
        self.minFaceDistance = 10

    def run(self):
        
        standup_frame_cnt = 0
        sitdown_frame_cnt = 0
        #containsEnoughMotion = False
        #detected = False
        
        
        use_spatial_propagation = False
        use_temporal_propagation = True
        #inst = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOPTICAL_FLOW_PRESET_MEDIUM)
        #inst = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOPTICAL_FLOW_PRESET_FAST)
        inst = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        inst.setUseSpatialPropagation(use_spatial_propagation)
        flow = None
        self.prev_gray = None

        delayTime = 1
        jump = 20
        badframeCnt = 0

        facedetector = demo_mtcnn.initFaceDetector()

        standup_roi = [] # [[x1,y1,x2,y2], [], [] ..]
        labelFace = {} # key:label, value:boxes;  contour1[ box1[[t1],[t2],[t3]], box2[], box3[] ]

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
            # Step 1: calc dense optflow
            #########
            if self.prev_gray is not None:
                flow = inst.calc(self.prev_gray, frame_gray, None) 
                vis = draw_flow(vis, flow)

            #if flow is not None and use_temporal_propagation:
            #   #warp previous flow to get an initial approximation for the current flow:
            #    flow = inst.calc(self.prev_gray, frame_gray, warp_flow(flow,flow))
            #elif self.prev_gray is not None:
            #    flow = inst.calc(self.prev_gray, frame_gray, None)

                #########
                # Step 2: get region
                #########
                #h, w = flow.shape[:2]
                #scale = 10
                #y, x = np.mgrid[0:h:scale, 0:w:scale].astype(int)
                #fx, fy = flow[y,x].T
                h, w = flow.shape[:2]
                y, x = np.mgrid[0:h:1, 0:w:1].astype(int)
                fx, fy = flow[y,x].T
                fy = fy.T
                _, fy_up = cv2.threshold(fy, 0.5, 255, cv2.THRESH_BINARY)
                _, fy_down = cv2.threshold(fy, -0.5, 255, cv2.THRESH_BINARY_INV)
                fy = fy_up + fy_down


                #########
                # Step 3: get contours
                #########
                fy = np.uint8(fy)
                _, contours0, hierarchy = cv2.findContours( fy.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]    

                # filter contours
                minLength = 2
                minArea = 2
                new_contours=[]
                for index, c in enumerate(contours):
                    if len(c) > minLength and cv2.contourArea(c) > minArea: # and hierarchy[index] is not None:
                        new_contours.append(c)
                contours = new_contours
                #cv2.drawContours( vis, contours, -1, 255, 2)

                # get centers of contours
                centers = []
                for c in contours:
                    M = cv2.moments(c)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    centers.append((cx, cy))
    
                #########
                # Step 4: label contours
                #########

                # init contourLabels, like [-1, -1, ..] or []
                contourLabels = [-1 for i in range(len(contours))]
                labelHist = [0 for i in range(self.maxLabel)]

                # if it is the first frame
                if len(self.prevCenters) is 0: 
                    contourLabels = [i for i in range(len(contours))]
                else:
                    for index, c in enumerate(contours):
                        cx = centers[index][0]
                        cy = centers[index][1]

                        # if current contour has existed
                        if self.prevContourMask[cy][cx] != 0 :
                            # find corresponding contour
                            minDist = 10000
                            prevIndex = 0
                            for i, c in enumerate(self.prevCenters):
                                dist = abs(int(c[0]) - int(cx)) + abs(int(c[1]) - int(cy))
                                if dist < minDist:
                                    minDist = dist
                                    prevIndex = i
                            label = self.prevContourLabels[prevIndex]
                            contourLabels[index] = label
                            if label < self.maxLabel:
                                labelHist[label] = self.prevLabelHist[label] + 1
                            else:
                                print("error: label > maxLabel, reset maxLabel")

                        # if the current contour appears for the first time
                        else: 
                            label = 0
                            while True:
                                if label in self.prevContourLabels or label in contourLabels:
                                    label = label + 1
                                else:
                                    contourLabels[index] = label
                                    if label < self.maxLabel:
                                        labelHist[label] = self.prevLabelHist[label] + 1
                                    break
                
                self.prevCenters = centers
                self.prevContourLabels = contourLabels
                self.prevLabelHist = labelHist
                self.prevContourMask = fy
             
                # filtering using self.minTime
                longTimeLabels = [i for i,labelCnt in enumerate(labelHist) if labelCnt >= self.minTime]
                

                #########
                # Step 5: detect face (detect face in each contour)
                #########

                face_region = []

                for currentLabel in longTimeLabels:
                    # get index in contourLabels
                    longTimeLabelIndex = [i for i, label in enumerate(contourLabels) if label == currentLabel]
                    if len(longTimeLabelIndex) is not 0:
                        longTimeLabelIndex = longTimeLabelIndex[0]
                        x,y,w,h = cv2.boundingRect(contours[longTimeLabelIndex])
                        face, boxes = demo_mtcnn.haveFace(frame[y:y+h, x:x+w], facedetector)
                        if not face:
                            continue
                        else:
                            for i in range(boxes.shape[0]):
                                # change coordinate, from bbox to frame
                                boxes[i][0] += x
                                boxes[i][1] += y
                                boxes[i][2] += x
                                boxes[i][3] += y
                            boxes = boxes.tolist()
                            face_region = boxes # 2
                            print("boxes len", len(boxes))

                            if currentLabel in labelFace.keys():
                                for box in boxes:
                                    center = ((box[0] + box[2])/2, (box[1] + box[3])/2)
                                    minDist = sys.maxint
                                    nearBoxIndex = -1
                                    # find previous box 
                                    for boxIndex, prevBox in enumerate(labelFace[currentLabel]):
                                        prevCenter = ((prevBox[-1][0] + prevBox[-1][2])/2, (prevBox[-1][1] + prevBox[-1][3])/2)
                                        dist = np.abs(prevCenter[0]-center[0]) + np.abs(prevCenter[1]-center[1])
                                        if dist < minDist:
                                            minDist = dist
                                            nearBoxIndex = boxIndex
                                    # add box to labelFace
                                    labelFace[currentLabel][nearBoxIndex].append(box)
                            else:
                                # create a new key-value, saving (label, boxes)
                                labelFace[currentLabel] = [[i] for i in boxes]
                                print(labelFace[currentLabel])

                # run when contour disappears
                for label in labelFace.keys():
                    if label not in longTimeLabels:
                        # one labelFace[label] may contain a lot of boxSequences
                        for boxSeq in labelFace[label]:
                            box1 = boxSeq[0]
                            box2 = boxSeq[-1]
                            center1 = ((box1[0] + box1[2])/2, (box1[1] + box1[3])/2)
                            center2 = ((box2[0] + box2[2])/2, (box2[1] + box2[3])/2)
                            dist = np.abs(center1[0]-center2[0]) + np.abs(center1[1]-center2[1])
                            print("dist:{}".format(dist))

                            if dist > self.minFaceDistance: # self.minFaceDistance = 10
                                print("Standup or Sitdown: ", boxSeq[-1])
                                standup_roi.append(boxSeq[-1])
                        labelFace.pop(label)

                print("### labelFace  ###")
                numbox = 0
                for boxes in labelFace.values():
                    numbox = len(boxes)
                    for boxseq in boxes:
                        print("boxseq", boxseq)
                print("num of faces", numbox)
                print("contours containing face", len(labelFace))
               
                print("\n")
                ## calc face speed
                #face_speed = []
                #for r in face_region:
                #    center_x = (r[0] + r[2])/2
                #    center_y = (r[1] + r[3])/2
                #    speed = flow[center_y, center_x]
                #    face_speed.append(speed)
                ##print(face_speed)

                

                ## filter face using speed at y axis
                #newFaceRegion = []
                #face_standup_t = -0.5
                #face_sitdown_t = 0.5
                #for i, face in enumerate(face_region):
                #    if face_speed[i][1] < face_standup_t:
                #        newFaceRegion.append(face)
                #    elif face_speed[i][1] > face_sitdown_t:
                #        newFaceRegion.append(face)
                #face_region = newFaceRegion

                drawface(vis, face_region)

                for i, c in enumerate(contours):
                    cv2.drawContours( vis, [c], -1, color[contourLabels[i]%len(color)], 1) #labelHist[contourLabels[i]])

            for r in standup_roi:
                cv2.rectangle(vis, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (0,0,255), 4)
                

            cv2.imshow('test', vis)
            self.frame_idx += 1
            self.prev_gray = frame_gray

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
    
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
