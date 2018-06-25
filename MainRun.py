# coding=utf-8
import argparse
from WordModel import WordEmbedding
import numpy as np
from FeatureDataSet import WordDataSet
from LSTMDNN import TwoInDNNModel

#test phase
def testPhase():
    print("\n ============begin to test=========== \n")
    data=WordDataSet(testMode=True)
    data.loadDocsData(inputfile)
    docs=data.getAllDocs()

    docModel=WordEmbedding()
    docModel.loadModel()
    px=docModel.transformDoc2Vec(docs)

    n_count = len(px)
    s1 = px[:n_count // 2]
    s2 = px[n_count // 2:]
    labels=np.zeros(shape=len(s1),dtype=np.int)

    data.constructData(s1,s2,labels)

    classifier=TwoInDNNModel()

    classifier.loadModel()

    results=classifier.predict([data.dataX1,data.dataX2])


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


    #run test

    testPhase()
