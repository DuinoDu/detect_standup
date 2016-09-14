#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
OK, the problem is, I have image samples, including both positive and negative.
Now, I need to use hog_svm to classify the samples.

Why use hog_svm?
Because for motion history image, silh is important to represent motion information.
Hog is a pretty good method to represent boundary information.
And hog+svm is the most used combination.

How?
1. For each sample in train image set, compute hog features and store them.
2. Construct trainset (feature, label).
3. svm.train
4. svm.validation
'''

# Python 2/3 compatibility
from __future__ import print_function
import os
import pickle
import numpy as np
import cv2

_DEBUG = False

def constructSampleSet(path):
    """ construct sample set from image sample

    :path: dirctory containing 'positive/*.jpg' and 'negative'
    :returns: sampleSet[hogFeatures, label]
    
    1. get image path
    2. compute hog for each image
    3. store feature and label
    4. save
    """
    positiveDir = path + "/positive" 
    negativeDir = path + "/negative" 
    if not os.path.exists(positiveDir) or not os.path.exists(negativeDir):
        print("Not find positive and negative file")
        return None
   
    if not os.path.exists(path+ '/sampleSet.pkl'):
        pSamples = os.listdir(positiveDir)
        nSamples = os.listdir(negativeDir)
        hog = cv2.HOGDescriptor()
        normalSize = (90, 66)
        sampleSet = {'features':[], 'label':[]}

        print("compute hog from positive samples")
        for s in pSamples:
            imgPath = positiveDir +'/'+ s
            img = cv2.imread(imgPath, 0)
            img = cv2.resize(img, normalSize)
            
            if _DEBUG:
                print(img.shape)
                cv2.imwrite(path + '/normal/p_'+s, img)

            features = hog.compute(img, hog.blockStride, hog.cellSize, ((0,0),) ) # why (0,0)?? what does location mean?
            sampleSet['features'].append(features)
            sampleSet['label'].append(1)

        print("compute hog from negative samples")
        for s in nSamples:
            imgPath = negativeDir +'/'+ s
            img = cv2.imread(imgPath, 0)
            img = cv2.resize(img, normalSize)
            
            if _DEBUG:
                print(img.shape)
                cv2.imwrite(path + '/normal/n_'+s, img)

            features = hog.compute(img, hog.blockStride, hog.cellSize, ((0,0),) )
            sampleSet['features'].append(features)
            sampleSet['label'].append(0)

        #output = open( path+'/sampleSet.pkl', 'wb')
        #pickle.dump(sampleSet, output)
        #output.close()
        #print("Save to sampleSet.pkl")
    else:
        print("Read from sampleSet.pkl")
        dataFile = open( path+'/sampleSet.pkl', 'rb')
        sampleSet = pickle.load(dataFile)
        dataFile.close()

    print("features length: ", len(sampleSet['features']))
    return sampleSet



def splitSampleSet(sampleSet):
    """ divide sample set into train set and valid set.

    :sampleSet: [features, label]
    :returns: trainSet, validSet

    """
    pass 


def train(trainSet):
    """ train svm on the trainSet

    :trainSet: [features, label]
    :returns: None

    """
    pass


def valid(validSet):
    """ valid svm model on the validSet

    :validSet: [features, label]
    :returns: None

    """
    pass


if __name__ == '__main__':

    import sys
 
    sampleSet = constructSampleSet(sys.argv[1])
    #trainSet, validSet = splitSampleSet(sampleSet)
    #train(trainSet)
    #valid(validSet)

