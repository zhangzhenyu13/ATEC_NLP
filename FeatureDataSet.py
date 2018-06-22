#coding=utf-8
import numpy as np
import pandas as pd

class DocDataSet:
    def __init__(self,testMode):
        self.dataX=None
        self.dataY=None
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

