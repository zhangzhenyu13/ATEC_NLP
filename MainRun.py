# coding=utf-8
import collections
import argparse
from DocData import DocDatapreprocessing
import numpy as np
from FeatureDataSet import DocDataSet
from XGboostBinaryClassifier import XGBoostClassifier
from DNNBinClassifier import DNNCLassifier
import initConfig
from utilityFiles import testScore

#test phase
def testPhase():
    print("\n ============begin to test=========== \n")
    docdata=DocDataSet(testMode=True)
    docdata.loadDocsData(inputfile)
    docs=docdata.getAllDocs()

    docModel=DocDatapreprocessing()
    docModel.loadModel()
    px=docModel.transformDoc2Vec(docs)

    n_count = len(px)
    s1 = px[:n_count // 2]
    s2 = px[n_count // 2:]
    labels=np.zeros(shape=len(s1),dtype=np.int)

    docdata.constructData(s1,s2,labels)

    classifier=Models[modeltype]()
    classifier.loadModel()

    results=classifier.predict(docdata.dataX)

    with open(outputfile,"w") as f:
        for i in range(len(results)):
            results[i]=1-results[i]

            f.write(str(i)+"\t"+str(results[i])+"\n")

if __name__ == '__main__':
    # parse input and output file path
    inputfile = "./data/validate.csv"
    outputfile = "./data/results.csv"


    parser = argparse.ArgumentParser(description="feed input and output files")
    parser.add_argument('--input', dest="input", default=inputfile, help='path of input file')
    parser.add_argument('--output', dest="output", default=outputfile, help='path of output file')

    args = parser.parse_args()
    inputfile = args.input
    outputfile = args.output

    modeltype = initConfig.config["modeltype"]
    Models = {
        1: DNNCLassifier,
        2: XGBoostClassifier
    }

    #run test

    testPhase()
