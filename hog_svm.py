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

##################
# Classify Model #
##################
class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  # Known bug: https://github.com/opencv/opencv/issues/4969
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()



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

        
        sampleSet['features'] = np.array(sampleSet['features'])
        sampleSet['label'] = np.array(sampleSet['label'])

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
    ratio = 0.7
    pSum = len([i for i in sampleSet['label'] if i == 1]) 
    nSum = len([i for i in sampleSet['label'] if i == 0]) 
    pTrainSum = int(pSum*ratio)
    nTrainSum = int(nSum*ratio)

    trainSet = {'features':[], 'label':[]}
    validSet = {'features':[], 'label':[]}
    
    [a,b,c,d] = np.split(sampleSet["features"], [pTrainSum, pSum, pSum + nTrainSum])
    trainSet["features"] = np.concatenate((a,c))
    validSet["features"] = np.concatenate((b,d))
    [a,b,c,d] = np.split(sampleSet["label"], [pTrainSum, pSum, pSum + nTrainSum])
    trainSet["label"] = np.concatenate((a,c))
    validSet["label"] = np.concatenate((b,d))
    
    return trainSet, validSet



def train(model, trainSet):
    """ train svm on the trainSet

    :trainSet: [features, label], features and label should be np.array
    :returns: None

    """
    print('training SVM...')
    model.train(trainSet["features"], trainSet["label"])
    print('saving SVM as "mhi_svm.dat"...')
    model.save('mhi_svm.dat')


def valid(model, validSet):
    """ valid svm model on the validSet

    :validSet: [features, label]
    :returns: None

    """
    resp = model.predict(validSet["features"])
    err = (validSet["label"] != resp).mean()
    print('error: %.2f %%' % (err*100))

def main():
    import sys
    sampleSet = constructSampleSet(sys.argv[1])
    trainSet, validSet = splitSampleSet(sampleSet)
    model = SVM(C=2.67, gamma=5.383)
    train(model, trainSet)
    valid(model, validSet)
    
    return model
    

if __name__ == '__main__':
    main()

