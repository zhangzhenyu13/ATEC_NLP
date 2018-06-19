# coding=utf-8

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

#print inputfile, outputfile
# preprocessing

docdata=DocDatapreprocessing(inputfile,outputfile)
docdata.loadDocsData()
if initConfig.config["docmodel"]!=1:
    docdata.trainDocModel()
docdata.loadModel()
px1,px2=docdata.transformDoc2Vec()

s1=px1[0:len(px1):2]
s2=px2[1:len(px2):2]
labels=np.array(docdata.docdata["label"],dtype=np.int)

dataSet=FeatureData()
dataSet.constructData(s1,s2,labels)

classifier=XGBoostClassifier()
if initConfig.config["test"]!=1:
    classifier.trainModel(dataSet)
    classifier.saveModel()
else:
    print("test mode")
classifier.loadModel()
dataSet=FeatureData(True)
dataSet.constructData(s1,s2,labels)
results=classifier.predict(dataSet.testX)
print(metrics.f1_score(dataSet.testY,results))

with open(outputfile,"w") as f:
    for i in range(len(results)):
        f.write(str(i)+"\t"+str(results[i])+"\n")
