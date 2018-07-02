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

    def getFold(self,fold_num=5):
        from sklearn import model_selection
        kfold=model_selection.KFold(n_splits=fold_num,shuffle=True)
        dataList=[]
        Y=np.reshape(self.dataY,newshape=(len(self.dataY),1))
        for train_index, test_index in kfold.split(Y):
            data_train=NLPDataSet(False)
            data_test=NLPDataSet(False)
            data_train.dataEm1,data_train.dataEm2,data_train.dataY=\
                self.dataEm1[train_index],self.dataEm2[train_index],self.dataY[train_index]
            data_test.dataEm1,data_test.dataEm2,data_test.dataY=\
                self.dataEm1[test_index],self.dataEm2[test_index],self.dataY[test_index]

            dataList.append([data_train,data_test])

        return dataList