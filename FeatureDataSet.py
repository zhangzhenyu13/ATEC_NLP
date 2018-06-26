#coding=utf-8
import numpy as np
import pandas as pd

class NLPDataSet:
    def __init__(self,testMode):

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

    def constructData(self,s1,s2,em1,em2,labels):
        """

        :param s1:
        :param s2:
        :param labels:
        :return:
        """

        labels = 1 - labels
        self.dataS1 = s1
        self.dataS2 = s2
        self.dataEm1=em1
        self.dataEm2=em2
        self.dataY = labels
        self.computeSim(s1,s2)

    def computeSim(self,s1,s2):

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

        self.simY=np.reshape(s,newshape=(len(s),1))

