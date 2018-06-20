# coding=utf-8
import collections
import argparse
from DocData import DocDatapreprocessing
import numpy as np
from FeatureDataSet import FeatureData
from XGboostBinaryClassifier import XGBoostClassifier
from ExtreesClassifier import TreeClassifier
from DNNBinClassifier import DNNCLassifier
from sklearn import metrics
import initConfig
# parse input and output file path
inputfile = "./data/train_nlp_data.csv"
outputfile = "./data/results.csv"


parser = argparse.ArgumentParser(description="feed input and output files")
parser.add_argument('--input', dest="input",default=inputfile, help='path of input file')
parser.add_argument('--output', dest="output",default=outputfile, help='path of output file')

args = parser.parse_args()
inputfile = args.input
outputfile = args.output


modeltype=initConfig.config["modeltype"]
Models={
    1:DNNCLassifier,
    2:TreeClassifier,
    3:XGBoostClassifier
}

#pipeline

#train phase
def trainPhase():
    print("\n ===============tuning models============== \n")
    #data
    docdata=DocDatapreprocessing("./data/train_nlp_data.csv")
    docdata.loadDocsData()
    if initConfig.config["build-docmodel"]==1:
        docdata.trainDocModel()
        docdata.saveModel()
    else:
        docdata.loadModel()
    px=docdata.transformDoc2Vec(None)

    s1=px[0:len(px):2]
    s2=px[1:len(px):2]
    labels=np.array(docdata.docdata["label"],dtype=np.int)

    dataSet=FeatureData()
    dataSet.constructData(s1,s2,labels)

    #sim classifier
    classifier=Models[modeltype]()
    if initConfig.config["build-classifier"]==1:
        print("train mode")
        classifier.trainModel(dataSet)
        classifier.saveModel()

#test phase
def testPhase():
    print("\n ============begin to test=========== \n")
    docdata=DocDatapreprocessing(inputfile)
    docdata.loadDocsData()

    docdata.loadModel()
    px=docdata.transformDoc2Vec(None)

    s1=px[0:len(px):2]
    s2=px[1:len(px):2]
    labels=np.zeros(shape=len(s1),dtype=np.int)

    dataSet=FeatureData(True)
    dataSet.constructData(s1,s2,labels)

    classifier=Models[modeltype]()
    classifier.loadModel()

    results=classifier.predict(dataSet.testX)
    print(collections.Counter(results),collections.Counter(dataSet.testY))
    print("f1-score",metrics.f1_score(dataSet.testY,results))
    print("accuracy",metrics.accuracy_score(dataSet.testY,results))

    with open(outputfile,"w") as f:
        for i in range(len(results)):
            f.write(str(i)+"\t"+str(results[i])+"\n")

#trainPhase()
testPhase()