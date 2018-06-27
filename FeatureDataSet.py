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

    def constructData(self,em1,em2,labels):
        """

        :param s1:
        :param s2:
        :param labels:
        :return:
        """

        self.dataEm1=em1
        self.dataEm2=em2
        self.dataY = labels



