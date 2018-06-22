#coding=utf-8
import numpy as np
import pandas as pd

class DocDataSet:
    def __init__(self,testMode):
        self.dataX=None
        self.dataY=None
        self.simY=None
        self.docdata=None
        self.testMode=testMode

    def loadDocsData(self,inputfile):

        with open(inputfile,"r") as f:
            records=[]
            for line in f:
                record=line.replace("\n","").replace("\r", "").split("\t")
                records.append(record)
        if self.testMode:
            self.docdata=pd.DataFrame(data=records,columns=["no","s1","s2"])
        else:
            self.docdata=pd.DataFrame(data=records,columns=["no","s1","s2","label"])

        print("loaded %d records"%len(self.docdata["no"]))

    def getAllDocs(self):

        s1, s2 = self.docdata["s1"], self.docdata["s2"]
        docs = s1.append(s2)
        return docs

    def constructData(self,s1,s2,labels):
        #labels are reversed

        #print("before",collections.Counter(Y))
        labels=1-labels
        #print("after",collections.Counter(Y))

        s_f=np.concatenate((s1,s2),axis=1)

        s_f=np.asarray(s_f)

        self.dataX=s_f
        self.dataY=labels

        self.getSimValue(s1,s2)

    def getSimValue(self,s1,s2,force=False):
        if self.simY is not None and force==False:
            return self.simY

        def cos_sim(vector_a, vector_b):

            num=np.dot(vector_a,vector_b)
            denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            cos = num / denom
            sim = 0.5 + 0.5 * cos
            return sim

        s1=np.array(s1)
        s2=np.array(s2)
        s=np.zeros(len(s1))
        for i in range(len(s)):
            s[i]=cos_sim(s1[i],s2[i])

        self.simY=s

        return self.simY
