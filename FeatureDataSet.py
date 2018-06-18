import pandas as pd
import numpy as np

class FeatureData:
    def __init__(self,test=False):
        self.trainX=None
        self.trainY=None
        self.validateX=None
        self.validateY=None
        self.testX=None
        self.testY=None
        self.ratio=0.8
        self.testMode=test
    def constructData(self,s1,s2,label):

        s_f = []
        for i in range(len(s1)):
            s_f.append(list(s1[i]) + list(s2[i]))

        Y = np.array(label, dtype=np.int)
        dataseize = int(self.ratio * len(Y))


        if self.testMode:
            self.testX=np.array(s_f[dataseize:],dtype=np.float)
            self.testY=np.array(label[dataseize:],dtype=np.int)

        else:
            self.trainX=s_f[:dataseize]
            self.trainY=Y[:dataseize]
            self.validateX=s_f[dataseize:]
            self.validateY=Y[dataseize:]
