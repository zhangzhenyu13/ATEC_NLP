#coding=utf-8
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
        print(s1.shape,s2.shape)
        print(label.shape)
        #print(type(s1),type(s2),type(label))
        Y = np.array(list(label), dtype=np.int)
        s_f=[]
        for i in range(len(Y)):
            x1,x2=s1[i],s2[i]
            #print(len(x1),x1)
            #print(len(x2),x2)
            #print()

            x=np.concatenate((x1,x2))
            s_f.append(x)
        s_f=np.asarray(s_f)

        dataseize = int(self.ratio * len(Y))

        if self.testMode:
            self.testX=s_f[dataseize:]
            self.testY=Y[dataseize:]

        else:
            self.trainX=s_f[:dataseize]
            self.trainY=Y[:dataseize]
            self.validateX=s_f[dataseize:]
            self.validateY=Y[dataseize:]
