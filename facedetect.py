#!/usr/bin/env python

'''
face detection using haar cascades

USAGE:
    facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

# local modules
from video import create_capture
from common import clock, draw_str


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import sys, getopt
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    #cascade_fn = args.get('--cascade', "haarcascades/haarcascade_eye_tree_eyeglasses.xml")
    #cascade_fn = args.get('--cascade', "haarcascades/haarcascade_eye.xml")
    #cascade_fn = args.get('--cascade', "haarcascades/haarcascade_frontalcatface_extended.xml")
    #cascade_fn = args.get('--cascade', "haarcascades/haarcascade_frontalcatface.xml")
    #cascade_fn = args.get('--cascade', "haarcascades/haarcascade_frontalface_alt2.xml")
    #cascade_fn = args.get('--cascade', "haarcascades/haarcascade_frontalface_alt_tree.xml")
    #cascade_fn = args.get('--cascade', "haarcascades/haarcascade_frontalface_alt.xml")
    #cascade_fn = args.get('--cascade', "haarcascades/haarcascade_frontalface_default.xml")
    #cascade_fn = args.get('--cascade', "haarcascades/haarcascade_fullbody.xml")
    #cascade_fn = args.get('--cascade', "haarcascades/haarcascade_lefteye_2splits.xml")
    #cascade_fn = args.get('--cascade', "haarcascades/haarcascade_licence_plate_rus_16stages.xml")
    #cascade_fn = args.get('--cascade', "haarcascades/haarcascade_lowerbody.xml")
    #cascade_fn = args.get('--cascade', "haarcascades/haarcascade_profileface.xml")
    #cascade_fn = args.get('--cascade', "haarcascades/haarcascade_righteye_2splits.xml")
    #cascade_fn = args.get('--cascade', "haarcascades/haarcascade_russian_plate_number.xml")
    #cascade_fn = args.get('--cascade', "haarcascades/haarcascade_smile.xml")
    #cascade_fn = args.get('--cascade', "haarcascades/haarcascade_upperbody.xml")

    cascade = cv2.CascadeClassifier(cascade_fn)
    #nested_fn  = args.get('--nested-cascade', "haarcascades/haarcascade_eye.xml")
    #nested = cv2.CascadeClassifier(nested_fn)

    cam = create_capture(video_src, fallback='synth:bg=../data/lena.jpg:noise=0.05')

    while True:
        ret, img = cam.read()
        if ret:
            img = cv2.resize(img, (img.shape[1]/2, img.shape[0]/2))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            t = clock()
            rects = detect(gray, cascade)
            vis = img.copy()
            draw_rects(vis, rects, (0, 255, 0))
           
            '''
            if not nested.empty():
                for x1, y1, x2, y2 in rects:
                    roi = gray[y1:y2, x1:x2]
                    vis_roi = vis[y1:y2, x1:x2]
                    subrects = detect(roi.copy(), nested)
                    draw_rects(vis_roi, subrects, (255, 0, 0))
            '''
            dt = clock() - t


            draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
            cv2.imshow('facedetect', vis)

        if 0xFF & cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()

