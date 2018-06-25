# coding=utf-8
<<<<<<< HEAD
import argparse
from WordModel import WordEmbedding
import numpy as np
from FeatureDataSet import WordDataSet
from LSTMDNN import TwoInDNNModel
=======
import collections
import argparse
from WordModel import WordEmbedding
import numpy as np
from FeatureDataSet import DocDataSet
import initConfig
>>>>>>> 9785dac91bb11fde2f45de06f1cad56ddd806f13

#test phase
def testPhase():
    print("\n ============begin to test=========== \n")
<<<<<<< HEAD
    data=WordDataSet(testMode=True)
    data.loadDocsData(inputfile)
    docs=data.getAllDocs()

    docModel=WordEmbedding()
=======
    docdata=DocDataSet(testMode=True)
    docdata.loadDocsData(inputfile)
    docs=docdata.getAllDocs()

    docModel=DocDatapreprocessing()
>>>>>>> 9785dac91bb11fde2f45de06f1cad56ddd806f13
    docModel.loadModel()
    px=docModel.transformDoc2Vec(docs)

    n_count = len(px)
    s1 = px[:n_count // 2]
    s2 = px[n_count // 2:]
    labels=np.zeros(shape=len(s1),dtype=np.int)

<<<<<<< HEAD
    data.constructData(s1,s2,labels)

    classifier=TwoInDNNModel()

    classifier.loadModel()

    results=classifier.predict([data.dataX1,data.dataX2])
=======
    docdata.constructData(s1,s2,labels)

    classifier=
    classifier.loadModel()

    results=classifier.predict(docdata.dataX)
>>>>>>> 9785dac91bb11fde2f45de06f1cad56ddd806f13


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

<<<<<<< HEAD
=======
    modeltype = initConfig.config["modeltype"]

>>>>>>> 9785dac91bb11fde2f45de06f1cad56ddd806f13

    #run test

    testPhase()
