#coding=utf-8
import numpy as np

class FeatureData:
    def __init__(self,test=False):
        self.trainX=None
        self.trainY=None

        self.testX=None
        self.testY=None

        self.testMode=test

    def constructData(self,s1,s2,label):
        #print(s1.shape,s2.shape)
        #print(label.shape)
        #print(type(s1),type(s2),type(label))
        Y = np.array(list(label), dtype=np.int)
        s_f=np.concatenate((s1,s2),axis=1)

        s_f=np.asarray(s_f)

        if self.testMode:
            self.testX=s_f
            self.testY=Y

        else:
            self.trainX=s_f
            self.trainY=Y

