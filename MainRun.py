# coding=utf-8
import argparse
from WordModel import WordEmbedding
from FeatureDataSet import NLPDataSet
import numpy as np
from LSTMClassifier import LSTMModel
from CNNClassifier import CNNModel
import initConfig
#test phase
def testPhase():
    print("\n ============begin to test=========== \n")

    data = NLPDataSet(testMode=True)
    data.loadDocsData(inputfile)
    docs = data.getAllDocs()

    # embedding words
    emModel = WordEmbedding()
    emModel.loadModel()

    embeddings = emModel.transformDoc2Vec(docs)


    n_count = len(embeddings)
    em1 = embeddings[:n_count // 2]
    em2 = embeddings[n_count // 2:]

    labels = np.zeros(shape=n_count,dtype=np.int)

    data.constructData(em1=em1, em2=em2, labels=labels)

    classifier = CNNModel()

    classifier.loadModel()

    results=classifier.predict(data)
    no=data.docdata["no"]

    results=np.array(results,dtype=np.int)
    no=np.array(no,dtype=np.int)

    with open(outputfile,"w") as f:
        for i in range(len(results)):

            f.write(str(no[i])+"\t"+str(results[i])+"\n")

def ensemBleTest():
    print("\n ============begin to test=========== \n")

    data = NLPDataSet(testMode=True)
    data.loadDocsData(inputfile)
    docs = data.getAllDocs()

    # embedding words
    emModel = WordEmbedding()
    emModel.loadModel()

    embeddings = emModel.transformDoc2Vec(docs)

    n_count = len(embeddings)
    em1 = embeddings[:n_count // 2]
    em2 = embeddings[n_count // 2:]

    labels = np.zeros(shape=n_count, dtype=np.int)

    data.constructData(em1=em1, em2=em2, labels=labels)

    classifiers=[]
    for i in range(initConfig.config["cnnNum"]):
        classifier = CNNModel()
        classifier.name+=str(i)
        classifier.loadModel()
        classifiers.append(classifier)
    for i in range(initConfig.config["lstmNum"]):
        classifier=LSTMModel()
        classifier.name+=str(i)
        classifier.loadModel()
        classifiers.append(classifier)

    resultsList=[]
    for predictor in classifiers:

        predicts = predictor.predict(data)
        resultsList.append(predicts)

    no = data.docdata["no"]
    results=np.sum(resultsList,axis=0)
    vote_c=1+(initConfig.config["lstmNum"]+initConfig.config["cnnNum"])//2
    results = np.array(results>vote_c, dtype=np.int)
    no = np.array(no, dtype=np.int)

    with open(outputfile, "w") as f:
        for i in range(len(results)):
            f.write(str(no[i]) + "\t" + str(results[i]) + "\n")


if __name__ == '__main__':
    # parse input and output file path
    inputfile = "../data/validateX.csv"
    outputfile = "../data/results.csv"

    parser = argparse.ArgumentParser(description="feed input and output files")
    parser.add_argument('--input', dest="input", default=inputfile, help='path of input file')
    parser.add_argument('--output', dest="output", default=outputfile, help='path of output file')

    args = parser.parse_args()
    inputfile = args.input
    outputfile = args.output

    #run test
    ensemBleTest()
