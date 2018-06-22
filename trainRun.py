# coding=utf-8

from DocData import DocDatapreprocessing
import numpy as np
from FeatureDataSet import DocDataSet
from XGboostBinaryClassifier import XGBoostClassifier
from DNNBinClassifier import DNNCLassifier
import initConfig

#train phase
def trainPhase():
    print("\n ===============tuning models============== \n")
    #data
    docdata=DocDataSet(testMode=False)
    docdata.loadDocsData("./data/train.csv")
    docModel=DocDatapreprocessing()
    docs=docdata.getAllDocs()

    if initConfig.config["build-docmodel"]==1:
        docModel.trainDocModel(docs)
        docModel.saveModel()
    else:
        docModel.loadModel()

    px=docModel.transformDoc2Vec(docs)

    n_count=len(px)
    s1=px[:n_count//2]
    s2=px[n_count//2:]
    labels=np.array(docdata.docdata["label"],dtype=np.int)

    docdata.constructData(s1,s2,labels)

    #sim classifier
    classifier=Models[modeltype]()
    if initConfig.config["build-classifier"]==1:
        print("train mode")
        classifier.trainModel(docdata)
        classifier.saveModel()


if __name__ == '__main__':

    modeltype = initConfig.config["modeltype"]
    Models = {
        1: DNNCLassifier,
        2: XGBoostClassifier
    }

    trainPhase()